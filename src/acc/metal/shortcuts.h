#ifndef METAL_SHORTCUTS_H_
#define METAL_SHORTCUTS_H_

#include "src/acc/metal/metal_settings.h"

// Type definitions for Metal backend
#ifdef __OBJC__
    typedef id<MTLDevice> deviceStream_t;
    typedef id<MTLCommandQueue> commandQueue_t;
    typedef id<MTLCommandBuffer> commandBuffer_t;
    typedef id<MTLComputeCommandEncoder> computeEncoder_t;
    typedef id<MTLBuffer> deviceBuffer_t;
    typedef id<MTLComputePipelineState> pipelineState_t;
#else
    // Forward declarations for C++ compatibility
    typedef void* deviceStream_t;
    typedef void* commandQueue_t;
    typedef void* commandBuffer_t;
    typedef void* computeEncoder_t;
    typedef void* deviceBuffer_t;
    typedef void* pipelineState_t;
#endif

// Memory copy macros
#ifdef __OBJC__
    #define METAL_MEMCPY_TO_DEVICE(dst, src, size, queue) \
        do { \
            memcpy([dst contents], src, size); \
            [dst didModifyRange:NSMakeRange(0, size)]; \
        } while(0)

    #define METAL_MEMCPY_FROM_DEVICE(dst, src, size, queue) \
        do { \
            memcpy(dst, [src contents], size); \
        } while(0)

    #define METAL_MEMCPY_DEVICE_TO_DEVICE(dst, src, size, queue) \
        do { \
            id<MTLCommandBuffer> cmdBuf = [queue commandBuffer]; \
            id<MTLBlitCommandEncoder> blitEncoder = [cmdBuf blitCommandEncoder]; \
            [blitEncoder copyFromBuffer:src sourceOffset:0 \
                               toBuffer:dst destinationOffset:0 size:size]; \
            [blitEncoder endEncoding]; \
            [cmdBuf commit]; \
            [cmdBuf waitUntilCompleted]; \
        } while(0)
#else
    #define METAL_MEMCPY_TO_DEVICE(dst, src, size, queue)
    #define METAL_MEMCPY_FROM_DEVICE(dst, src, size, queue)
    #define METAL_MEMCPY_DEVICE_TO_DEVICE(dst, src, size, queue)
#endif

// Synchronization macros
#ifdef __OBJC__
    #define METAL_SYNC() \
        do { /* Metal operations are inherently asynchronous, sync via command buffer */ } while(0)

    #define METAL_QUEUE_SYNC(queue) \
        do { \
            id<MTLCommandBuffer> cmdBuf = [queue commandBuffer]; \
            [cmdBuf commit]; \
            [cmdBuf waitUntilCompleted]; \
        } while(0)
#else
    #define METAL_SYNC()
    #define METAL_QUEUE_SYNC(queue)
#endif

// Error checking macros
#define METAL_ERROR_CHECK(err) HANDLE_ERROR(err)
#define METAL_LAUNCH_CHECK(err) LAUNCH_HANDLE_ERROR(err)

// Threadgroup (workgroup) size calculation helpers
#define METAL_CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define METAL_THREADGROUPS(total, threadsPerGroup) METAL_CEIL_DIV(total, threadsPerGroup)

// Precision settings
#ifdef ACC_DOUBLE_PRECISION
    typedef double XFLOAT;
    #define METAL_XFLOAT_TYPE float  // Metal doesn't support double natively on all GPUs
    #warning "Metal backend may not fully support double precision on all hardware"
#else
    typedef float XFLOAT;
    #define METAL_XFLOAT_TYPE float
#endif

// Complex number helper (Metal uses packed types)
struct metal_complex {
    XFLOAT real;
    XFLOAT imag;
};

#endif /* METAL_SHORTCUTS_H_ */
