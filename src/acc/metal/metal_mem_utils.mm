#import "src/acc/metal/metal_mem_utils.h"
#import "src/acc/metal/metal_helper_functions.h"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstring>
#include <mutex>
#include <unordered_map>

// ============================================================================
// Buffer Registry: Maps data pointers (buffer.contents) to MTLBuffer objects
// ============================================================================
// This is critical because RELION passes around data pointers (XFLOAT*) to kernels,
// but Metal's encoder.setBuffer() requires MTLBuffer objects. We maintain a registry
// to look up the MTLBuffer from its contents pointer.

static std::unordered_map<const void*, id<MTLBuffer>> g_buffer_registry;
static std::mutex g_buffer_registry_mutex;

// Register a buffer in the registry (called after allocation)
void metalRegisterBuffer(MTLBufferPtr buffer) {
    if (!buffer) return;

    @autoreleasepool {
        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
        void* contents = mtlBuffer.contents;

        if (contents) {
            std::lock_guard<std::mutex> lock(g_buffer_registry_mutex);
            g_buffer_registry[contents] = mtlBuffer;
        }
    }
}

// Unregister a buffer from the registry (called before deallocation)
void metalUnregisterBuffer(MTLBufferPtr buffer) {
    if (!buffer) return;

    @autoreleasepool {
        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
        void* contents = mtlBuffer.contents;

        if (contents) {
            std::lock_guard<std::mutex> lock(g_buffer_registry_mutex);
            g_buffer_registry.erase(contents);
        }
    }
}

// Look up MTLBuffer from a data pointer
MTLBufferPtr metalGetBufferFromPointer(const void* dataPtr) {
    if (!dataPtr) return nullptr;

    std::lock_guard<std::mutex> lock(g_buffer_registry_mutex);
    auto it = g_buffer_registry.find(dataPtr);
    if (it != g_buffer_registry.end()) {
        return (__bridge MTLBufferPtr)it->second;
    }

    // Not found - this is a programming error
    fprintf(stderr, "Metal ERROR: Buffer not found in registry for pointer %p\n", dataPtr);
    fprintf(stderr, "             This likely means the pointer was not allocated via metalAllocateDevice\n");
    fprintf(stderr, "             or the buffer was freed before use.\n");
    return nullptr;
}

// Convert MetalMemoryMode to MTLResourceOptions
static MTLResourceOptions getMTLResourceOptions(MetalMemoryMode mode) {
    switch (mode) {
        case METAL_MEM_SHARED:
            return MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache;
        case METAL_MEM_PRIVATE:
            return MTLResourceStorageModePrivate;
        case METAL_MEM_MANAGED:
#if TARGET_OS_OSX
            return MTLResourceStorageModeManaged;
#else
            // iOS doesn't support managed mode, use shared instead
            return MTLResourceStorageModeShared;
#endif
        default:
            return MTLResourceStorageModeShared;
    }
}

extern "C" {

void* metalAllocateDevice(MTLDevicePtr device, size_t size, MetalMemoryMode mode) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        if (!mtlDevice) {
            fprintf(stderr, "Metal ERROR: Invalid device in metalAllocateDevice\n");
            return nullptr;
        }

        MTLResourceOptions options = getMTLResourceOptions(mode);
        id<MTLBuffer> buffer = [mtlDevice newBufferWithLength:size options:options];

        if (!buffer) {
            fprintf(stderr, "Metal ERROR: Failed to allocate buffer of size %zu\n", size);
            return nullptr;
        }

        // Register buffer in the pointer-to-buffer registry
        // This allows kernel launchers to look up the MTLBuffer from data pointers
        void* result = (__bridge_retained void*)buffer;
        metalRegisterBuffer((MTLBufferPtr)result);

        return result;
    }
}

void metalFreeDevice(MTLBufferPtr buffer) {
    @autoreleasepool {
        if (buffer) {
            // Unregister from the pointer-to-buffer registry before freeing
            metalUnregisterBuffer(buffer);

            // Transfer ownership back and release
            id<MTLBuffer> mtlBuffer = (__bridge_transfer id<MTLBuffer>)buffer;
            mtlBuffer = nil;
        }
    }
}

void metalCopyHostToDevice(MTLBufferPtr buffer, const void* hostPtr, size_t size, size_t offset) {
    @autoreleasepool {
        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
        if (!mtlBuffer || !hostPtr) {
            fprintf(stderr, "Metal ERROR: Invalid parameters in metalCopyHostToDevice\n");
            return;
        }

        void* bufferPtr = mtlBuffer.contents;
        if (!bufferPtr) {
            fprintf(stderr, "Metal ERROR: Buffer has no CPU-accessible contents (may be private storage)\n");
            return;
        }

        memcpy(static_cast<char*>(bufferPtr) + offset, hostPtr, size);

#if TARGET_OS_OSX
        // Notify Metal that we modified the buffer (important for managed storage)
        if (mtlBuffer.storageMode == MTLStorageModeManaged) {
            [mtlBuffer didModifyRange:NSMakeRange(offset, size)];
        }
#endif
    }
}

void metalCopyDeviceToHost(void* hostPtr, MTLBufferPtr buffer, size_t size, size_t offset) {
    @autoreleasepool {
        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
        if (!mtlBuffer || !hostPtr) {
            fprintf(stderr, "Metal ERROR: Invalid parameters in metalCopyDeviceToHost\n");
            return;
        }

        const void* bufferPtr = mtlBuffer.contents;
        if (!bufferPtr) {
            fprintf(stderr, "Metal ERROR: Buffer has no CPU-accessible contents (may be private storage)\n");
            return;
        }

        memcpy(hostPtr, static_cast<const char*>(bufferPtr) + offset, size);
    }
}

void metalCopyDeviceToDevice(MTLBufferPtr dst, MTLBufferPtr src, size_t size, MTLCommandQueuePtr queue) {
    @autoreleasepool {
        id<MTLBuffer> dstBuffer = (__bridge id<MTLBuffer>)dst;
        id<MTLBuffer> srcBuffer = (__bridge id<MTLBuffer>)src;
        id<MTLCommandQueue> cmdQueue = (__bridge id<MTLCommandQueue>)queue;

        if (!dstBuffer || !srcBuffer || !cmdQueue) {
            fprintf(stderr, "Metal ERROR: Invalid parameters in metalCopyDeviceToDevice\n");
            return;
        }

        id<MTLCommandBuffer> cmdBuffer = [cmdQueue commandBuffer];
        id<MTLBlitCommandEncoder> blitEncoder = [cmdBuffer blitCommandEncoder];

        [blitEncoder copyFromBuffer:srcBuffer
                       sourceOffset:0
                           toBuffer:dstBuffer
                  destinationOffset:0
                               size:size];

        [blitEncoder endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
    }
}

void metalMemsetDevice(MTLBufferPtr buffer, int value, size_t size, MTLCommandQueuePtr queue) {
    @autoreleasepool {
        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
        id<MTLCommandQueue> cmdQueue = (__bridge id<MTLCommandQueue>)queue;

        if (!mtlBuffer) {
            fprintf(stderr, "Metal ERROR: Invalid buffer in metalMemsetDevice\n");
            return;
        }

        // For shared/managed buffers, we can use CPU memset
        if (mtlBuffer.storageMode == MTLStorageModeShared
#if TARGET_OS_OSX
            || mtlBuffer.storageMode == MTLStorageModeManaged
#endif
        ) {
            void* bufferPtr = mtlBuffer.contents;
            if (bufferPtr) {
                memset(bufferPtr, value, size);
#if TARGET_OS_OSX
                if (mtlBuffer.storageMode == MTLStorageModeManaged) {
                    [mtlBuffer didModifyRange:NSMakeRange(0, size)];
                }
#endif
                return;
            }
        }

        // For private buffers, we'd need to use a compute kernel or blit encoder
        // For now, just log a warning
        if (mtlBuffer.storageMode == MTLStorageModePrivate) {
            fprintf(stderr, "Metal WARNING: metalMemsetDevice on private buffer requires compute kernel (not implemented)\n");
        }
    }
}

void* metalGetBufferPointer(MTLBufferPtr buffer) {
    @autoreleasepool {
        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
        if (!mtlBuffer) {
            return nullptr;
        }
        return mtlBuffer.contents;
    }
}

size_t metalGetBufferSize(MTLBufferPtr buffer) {
    @autoreleasepool {
        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
        if (!mtlBuffer) {
            return 0;
        }
        return mtlBuffer.length;
    }
}

void metalSynchronizeBuffer(MTLBufferPtr buffer) {
    @autoreleasepool {
#if TARGET_OS_OSX
        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
        if (mtlBuffer && mtlBuffer.storageMode == MTLStorageModeManaged) {
            // For managed buffers, we've already called didModifyRange in copy operations
            // No additional synchronization needed here
        }
#endif
        // For shared buffers, no synchronization needed (coherent)
    }
}

} // extern "C"

// C++ wrapper implementation
namespace Metal {

void* MemoryManager::allocate(void* device, size_t size, MetalMemoryMode mode) {
    return metalAllocateDevice((MTLDevicePtr)device, size, mode);
}

void MemoryManager::deallocate(void* buffer) {
    metalFreeDevice((MTLBufferPtr)buffer);
}

void MemoryManager::copyToDevice(void* buffer, const void* hostPtr, size_t size, size_t offset) {
    metalCopyHostToDevice((MTLBufferPtr)buffer, hostPtr, size, offset);
}

void MemoryManager::copyFromDevice(void* hostPtr, void* buffer, size_t size, size_t offset) {
    metalCopyDeviceToHost(hostPtr, (MTLBufferPtr)buffer, size, offset);
}

void MemoryManager::copyDeviceToDevice(void* dst, void* src, size_t size, void* queue) {
    metalCopyDeviceToDevice((MTLBufferPtr)dst, (MTLBufferPtr)src, size, (MTLCommandQueuePtr)queue);
}

void MemoryManager::memset(void* buffer, int value, size_t size, void* queue) {
    metalMemsetDevice((MTLBufferPtr)buffer, value, size, (MTLCommandQueuePtr)queue);
}

void* MemoryManager::getPointer(void* buffer) {
    return metalGetBufferPointer((MTLBufferPtr)buffer);
}

size_t MemoryManager::getSize(void* buffer) {
    return metalGetBufferSize((MTLBufferPtr)buffer);
}

} // namespace Metal
