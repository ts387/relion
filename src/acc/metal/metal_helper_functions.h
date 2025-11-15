#ifndef METAL_HELPER_FUNCTIONS_H_
#define METAL_HELPER_FUNCTIONS_H_

#include "src/acc/metal/metal_settings.h"
#include "src/acc/metal/shortcuts.h"

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

namespace MetalHelper {

// Get the default Metal device
inline id<MTLDevice> getDefaultDevice() {
    static id<MTLDevice> device = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        // Prefer high-performance discrete GPU, fall back to integrated
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        for (id<MTLDevice> dev in devices) {
            if (!dev.isLowPower && !dev.isRemovable) {
                device = dev;
                break;
            }
        }
        if (device == nil && devices.count > 0) {
            device = devices[0];
        }
        if (device == nil) {
            device = MTLCreateSystemDefaultDevice();
        }
    });
    return device;
}

// Get all available Metal devices
inline NSArray<id<MTLDevice>>* getAllDevices() {
    return MTLCopyAllDevices();
}

// Create a command queue
inline id<MTLCommandQueue> createCommandQueue(id<MTLDevice> device) {
    return [device newCommandQueue];
}

// Allocate a Metal buffer
inline id<MTLBuffer> allocateBuffer(id<MTLDevice> device, size_t size, MTLResourceOptions options) {
    return [device newBufferWithLength:size options:options];
}

// Allocate a Metal buffer with initial data
inline id<MTLBuffer> allocateBufferWithData(id<MTLDevice> device, const void* data, size_t size, MTLResourceOptions options) {
    return [device newBufferWithBytes:data length:size options:options];
}

// Get buffer contents pointer
inline void* getBufferContents(id<MTLBuffer> buffer) {
    return [buffer contents];
}

// Copy host to device
inline void copyHostToDevice(id<MTLBuffer> buffer, const void* hostPtr, size_t size, size_t offset = 0) {
    memcpy(static_cast<char*>([buffer contents]) + offset, hostPtr, size);
    [buffer didModifyRange:NSMakeRange(offset, size)];
}

// Copy device to host
inline void copyDeviceToHost(void* hostPtr, id<MTLBuffer> buffer, size_t size, size_t offset = 0) {
    memcpy(hostPtr, static_cast<const char*>([buffer contents]) + offset, size);
}

// Synchronous device-to-device copy using blit encoder
inline void copyDeviceToDevice(id<MTLBuffer> dst, id<MTLBuffer> src, size_t size, id<MTLCommandQueue> queue) {
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blitEncoder = [cmdBuf blitCommandEncoder];
        [blitEncoder copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:size];
        [blitEncoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }
}

// Load Metal library from file
inline id<MTLLibrary> loadLibraryFromFile(id<MTLDevice> device, const char* filepath, NSError** error) {
    NSString* path = [NSString stringWithUTF8String:filepath];
    NSURL* url = [NSURL fileURLWithPath:path];
    return [device newLibraryWithURL:url error:error];
}

// Load Metal library from source
inline id<MTLLibrary> loadLibraryFromSource(id<MTLDevice> device, const char* source, NSError** error) {
    NSString* sourceString = [NSString stringWithUTF8String:source];
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    options.languageVersion = MTLLanguageVersion3_0;
    return [device newLibraryWithSource:sourceString options:options error:error];
}

// Create compute pipeline state
inline id<MTLComputePipelineState> createComputePipeline(id<MTLDevice> device, id<MTLLibrary> library, const char* functionName, NSError** error) {
    NSString* funcName = [NSString stringWithUTF8String:functionName];
    id<MTLFunction> function = [library newFunctionWithName:funcName];
    if (!function) {
        if (error) {
            *error = [NSError errorWithDomain:@"MetalHelper"
                                        code:-1
                                    userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithFormat:@"Function '%@' not found in library", funcName]}];
        }
        return nil;
    }
    return [device newComputePipelineStateWithFunction:function error:error];
}

// Get optimal threadgroup size for a pipeline
inline MTLSize getOptimalThreadgroupSize(id<MTLComputePipelineState> pipeline, MTLSize gridSize) {
    NSUInteger maxThreadsPerGroup = pipeline.maxTotalThreadsPerThreadgroup;
    NSUInteger threadExecutionWidth = pipeline.threadExecutionWidth;

    // Try to use a multiple of the thread execution width (SIMD width)
    NSUInteger threadsPerGroup = threadExecutionWidth;
    while (threadsPerGroup * 2 <= maxThreadsPerGroup && threadsPerGroup < 1024) {
        threadsPerGroup *= 2;
    }

    return MTLSizeMake(threadsPerGroup, 1, 1);
}

// Dispatch compute kernel with automatic threadgroup sizing
inline void dispatchCompute(id<MTLComputeCommandEncoder> encoder,
                            id<MTLComputePipelineState> pipeline,
                            size_t totalThreads,
                            MTLSize* outThreadgroupSize = nullptr) {
    MTLSize gridSize = MTLSizeMake(totalThreads, 1, 1);
    MTLSize threadgroupSize = getOptimalThreadgroupSize(pipeline, gridSize);

    if (outThreadgroupSize) {
        *outThreadgroupSize = threadgroupSize;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
}

// Get device memory info
inline void getDeviceMemoryInfo(id<MTLDevice> device, size_t* total, size_t* free) {
    // Metal doesn't provide direct memory queries like CUDA
    // We use approximate values based on system memory (unified memory architecture)
    if (device.hasUnifiedMemory) {
        // For Apple Silicon, GPU shares system RAM
        // Get available physical memory as approximation
        *total = device.recommendedMaxWorkingSetSize;
        *free = device.currentAllocatedSize > *total ? 0 : (*total - device.currentAllocatedSize);
    } else {
        // For discrete GPUs, use recommended working set
        *total = device.recommendedMaxWorkingSetSize;
        *free = device.currentAllocatedSize > *total ? 0 : (*total - device.currentAllocatedSize);
    }
}

// Check if device supports feature
inline bool supportsFeature(id<MTLDevice> device, MTLFeatureSet featureSet) {
    return [device supportsFeatureSet:featureSet];
}

} // namespace MetalHelper

#endif // __OBJC__

#endif /* METAL_HELPER_FUNCTIONS_H_ */
