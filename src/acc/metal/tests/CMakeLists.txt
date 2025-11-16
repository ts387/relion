# Metal Kernel Validation Tests
#
# Build with: cmake -DMETAL=ON -DBUILD_METAL_TESTS=ON ..
# Run: ./bin/test_metal_kernels

if(METAL AND BUILD_METAL_TESTS)
    message(STATUS "Building Metal kernel validation tests")

    add_executable(test_metal_kernels
        test_metal_kernels.cpp
    )

    target_include_directories(test_metal_kernels PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(test_metal_kernels
        relion_lib
        ${METAL_LIBRARIES}
    )

    target_compile_definitions(test_metal_kernels PRIVATE
        _METAL_ENABLED
    )

    if(APPLE)
        target_link_libraries(test_metal_kernels
            "-framework Metal"
            "-framework Foundation"
            "-framework Accelerate"
        )
    endif()

    set_target_properties(test_metal_kernels PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    # Add to CTest
    add_test(NAME metal_kernel_validation COMMAND test_metal_kernels)

endif()
