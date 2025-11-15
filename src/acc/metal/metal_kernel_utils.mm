#import "src/acc/metal/metal_kernel_utils.h"
#import "src/acc/metal/metal_helper_functions.h"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <map>
#include <string>

namespace MetalKernels {

class KernelLauncher::Impl {
public:
    id<MTLDevice> device;
    id<MTLLibrary> library;
    id<MTLCommandQueue> commandQueue;
    std::map<std::string, id<MTLComputePipelineState>> pipelineCache;
};

KernelLauncher::KernelLauncher(void* dev, const char* libraryPath) {
    @autoreleasepool {
        if (dev) {
            device = dev;
        } else {
            device = MetalHelper::getDefaultDevice();
        }

        // Load Metal library
        NSError* error = nil;
        if (libraryPath) {
            NSString* path = [NSString stringWithUTF8String:libraryPath];
            NSURL* url = [NSURL fileURLWithPath:path];
            library = (__bridge_retained void*)[(__bridge id<MTLDevice>)device newLibraryWithURL:url error:&error];
        } else {
            // Try to load default library from bundle
            library = (__bridge_retained void*)[(__bridge id<MTLDevice>)device newDefaultLibrary];
        }

        if (!library) {
            if (error) {
                fprintf(stderr, "Metal ERROR: Failed to load library: %s\n",
                        [[error localizedDescription] UTF8String]);
            }
        }

        // Create command queue
        commandQueue = MetalHelper::createCommandQueue((__bridge id<MTLDevice>)device);
    }
}

KernelLauncher::~KernelLauncher() {
    @autoreleasepool {
        if (library) {
            id<MTLLibrary> lib = (__bridge_transfer id<MTLLibrary>)library;
            lib = nil;
        }
        if (commandQueue) {
            // Command queue is not bridged_retained, so just clear
            commandQueue = nullptr;
        }
    }
}

void* KernelLauncher::getPipelineState(const char* kernelName) {
    @autoreleasepool {
        id<MTLLibrary> lib = (__bridge id<MTLLibrary>)library;
        if (!lib) {
            fprintf(stderr, "Metal ERROR: No library loaded\n");
            return nullptr;
        }

        NSString* funcName = [NSString stringWithUTF8String:kernelName];
        id<MTLFunction> function = [lib newFunctionWithName:funcName];
        if (!function) {
            fprintf(stderr, "Metal ERROR: Function '%s' not found in library\n", kernelName);
            return nullptr;
        }

        NSError* error = nil;
        id<MTLComputePipelineState> pipeline =
            [(__bridge id<MTLDevice>)device newComputePipelineStateWithFunction:function error:&error];

        if (!pipeline || error) {
            fprintf(stderr, "Metal ERROR: Failed to create pipeline for '%s': %s\n",
                    kernelName, [[error localizedDescription] UTF8String]);
            return nullptr;
        }

        return (__bridge_retained void*)pipeline;
    }
}

void KernelLauncher::launch1D(const char* kernelName,
                              size_t totalThreads,
                              void** buffers,
                              size_t* bufferSizes,
                              int numBuffers) {
    @autoreleasepool {
        void* pipelinePtr = getPipelineState(kernelName);
        if (!pipelinePtr) return;

        id<MTLComputePipelineState> pipeline = (__bridge_transfer id<MTLComputePipelineState>)pipelinePtr;
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)commandQueue;

        id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];

        // Bind buffers
        for (int i = 0; i < numBuffers; i++) {
            id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buffers[i];
            [encoder setBuffer:buffer offset:0 atIndex:i];
        }

        // Calculate thread configuration
        MTLSize gridSize = MTLSizeMake(totalThreads, 1, 1);
        NSUInteger threadGroupSize = pipeline.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > 256) threadGroupSize = 256;
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
    }
}

void KernelLauncher::launch2D(const char* kernelName,
                              size_t width, size_t height,
                              void** buffers,
                              size_t* bufferSizes,
                              int numBuffers) {
    @autoreleasepool {
        void* pipelinePtr = getPipelineState(kernelName);
        if (!pipelinePtr) return;

        id<MTLComputePipelineState> pipeline = (__bridge_transfer id<MTLComputePipelineState>)pipelinePtr;
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)commandQueue;

        id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];

        // Bind buffers
        for (int i = 0; i < numBuffers; i++) {
            id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buffers[i];
            [encoder setBuffer:buffer offset:0 atIndex:i];
        }

        // Calculate thread configuration
        MTLSize gridSize = MTLSizeMake(width, height, 1);
        NSUInteger w = pipeline.threadExecutionWidth;
        NSUInteger h = pipeline.maxTotalThreadsPerThreadgroup / w;
        MTLSize threadgroupSize = MTLSizeMake(w, h, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
    }
}

void KernelLauncher::launch3D(const char* kernelName,
                              size_t width, size_t height, size_t depth,
                              void** buffers,
                              size_t* bufferSizes,
                              int numBuffers) {
    @autoreleasepool {
        void* pipelinePtr = getPipelineState(kernelName);
        if (!pipelinePtr) return;

        id<MTLComputePipelineState> pipeline = (__bridge_transfer id<MTLComputePipelineState>)pipelinePtr;
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)commandQueue;

        id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];

        // Bind buffers
        for (int i = 0; i < numBuffers; i++) {
            id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buffers[i];
            [encoder setBuffer:buffer offset:0 atIndex:i];
        }

        // Calculate thread configuration
        MTLSize gridSize = MTLSizeMake(width, height, depth);
        NSUInteger maxThreads = pipeline.maxTotalThreadsPerThreadgroup;
        NSUInteger threadGroupSize = 8; // 8x8x8 = 512 threads
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, threadGroupSize, threadGroupSize);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
    }
}

void KernelLauncher::synchronize() {
    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)commandQueue;
        id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
    }
}

// Utility kernel dispatchers (simplified implementations)
void launchMultiplyKernel(void* input, void* output, float multiplier, size_t size, void* queue) {
    // TODO: Implement using metal_test_kernel or create dedicated multiply kernel
}

void launchExponentiateKernel(void* array, float add, size_t size, void* queue) {
    // TODO: Implement exponentiate kernel
}

void launchSoftMaskKernel(void* volume, size_t volSize, size_t xdim, size_t ydim, size_t zdim,
                          size_t xinit, size_t yinit, size_t zinit,
                          float radius, float cosine_width, void* queue) {
    // TODO: Implement soft mask kernel
}

} // namespace MetalKernels
