# CompileMetal.cmake
# CMake module for compiling Metal shader files (.metal) to Metal libraries (.metallib)

# Function to compile Metal shaders
# Usage: compile_metal_shaders(TARGET target_name SOURCES source1.metal source2.metal ...)
function(compile_metal_shaders)
    cmake_parse_arguments(METAL "" "TARGET;OUTPUT" "SOURCES" ${ARGN})

    if(NOT METAL_TARGET)
        message(FATAL_ERROR "compile_metal_shaders: TARGET argument is required")
    endif()

    if(NOT METAL_SOURCES)
        message(FATAL_ERROR "compile_metal_shaders: SOURCES argument is required")
    endif()

    # Find metal compiler (xcrun metal)
    find_program(METAL_COMPILER xcrun)
    if(NOT METAL_COMPILER)
        message(FATAL_ERROR "Metal compiler (xcrun) not found. Ensure Xcode is installed.")
    endif()

    # Find metal library tool (xcrun metallib)
    find_program(METAL_LINKER xcrun)
    if(NOT METAL_LINKER)
        message(FATAL_ERROR "Metal linker (xcrun metallib) not found. Ensure Xcode is installed.")
    endif()

    set(METAL_STANDARD "-std=metal3.0")
    set(METAL_FLAGS ${METAL_STANDARD})

    # Add optimization flags based on build type
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        list(APPEND METAL_FLAGS "-gline-tables-only" "-frecord-sources")
    else()
        list(APPEND METAL_FLAGS "-O3")
    endif()

    # Output directory for compiled shaders
    set(METAL_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/metal_shaders")
    file(MAKE_DIRECTORY ${METAL_OUTPUT_DIR})

    set(AIR_FILES)
    set(METALLIB_OUTPUT "${METAL_OUTPUT_DIR}/${METAL_TARGET}.metallib")

    # Compile each .metal file to .air (Apple Intermediate Representation)
    foreach(METAL_SOURCE ${METAL_SOURCES})
        get_filename_component(METAL_BASENAME ${METAL_SOURCE} NAME_WE)
        set(AIR_FILE "${METAL_OUTPUT_DIR}/${METAL_BASENAME}.air")

        add_custom_command(
            OUTPUT ${AIR_FILE}
            COMMAND ${METAL_COMPILER} metal ${METAL_FLAGS} -c ${CMAKE_CURRENT_SOURCE_DIR}/${METAL_SOURCE} -o ${AIR_FILE}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${METAL_SOURCE}
            COMMENT "Compiling Metal shader: ${METAL_SOURCE}"
            VERBATIM
        )

        list(APPEND AIR_FILES ${AIR_FILE})
    endforeach()

    # Link all .air files into a single .metallib
    add_custom_command(
        OUTPUT ${METALLIB_OUTPUT}
        COMMAND ${METAL_LINKER} metallib ${AIR_FILES} -o ${METALLIB_OUTPUT}
        DEPENDS ${AIR_FILES}
        COMMENT "Linking Metal library: ${METAL_TARGET}.metallib"
        VERBATIM
    )

    # Create a custom target for the Metal library
    add_custom_target(${METAL_TARGET}_shaders ALL DEPENDS ${METALLIB_OUTPUT})

    # Set output variable if provided
    if(METAL_OUTPUT)
        set(${METAL_OUTPUT} ${METALLIB_OUTPUT} PARENT_SCOPE)
    endif()

    # Make the .metallib available to the parent scope
    set(${METAL_TARGET}_METALLIB ${METALLIB_OUTPUT} PARENT_SCOPE)

    message(STATUS "Metal shaders will be compiled to: ${METALLIB_OUTPUT}")
endfunction()

# Function to add Metal library to a target
# Usage: target_add_metal_library(target_name metal_library_path)
function(target_add_metal_library TARGET METALLIB)
    # Copy the .metallib to the output directory at build time
    add_custom_command(
        TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${METALLIB} $<TARGET_FILE_DIR:${TARGET}>
        COMMENT "Copying Metal library to output directory"
    )

    # Install the .metallib alongside the executable
    install(FILES ${METALLIB} DESTINATION bin)
endfunction()
