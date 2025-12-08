#ifndef METAL_MEM_UTILS_H_
#define METAL_MEM_UTILS_H_

#include "src/acc/metal/metal_settings.h"
#include "src/acc/metal/shortcuts.h"

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Memory allocation modes
typedef enum {
    METAL_MEM_SHARED = 0,      // MTLStorageModeShared - CPU and GPU can both access
    METAL_MEM_PRIVATE = 1,     // MTLStorageModePrivate - GPU only
    METAL_MEM_MANAGED = 2      // MTLStorageModeManaged - synchronized between CPU/GPU
} MetalMemoryMode;

// Forward declarations for Objective-C types when not in Objective-C context
#ifndef __OBJC__
typedef void* MTLDevicePtr;
typedef void* MTLBufferPtr;
typedef void* MTLCommandQueuePtr;
#else
typedef id<MTLDevice> MTLDevicePtr;
typedef id<MTLBuffer> MTLBufferPtr;
typedef id<MTLCommandQueue> MTLCommandQueuePtr;
#endif

// Allocate device buffer (returns MTLBuffer as void*)
// The returned pointer is the MTLBuffer object, NOT the data pointer
void* metalAllocateDevice(MTLDevicePtr device, size_t size, MetalMemoryMode mode);

// Look up MTLBuffer from a data pointer (buffer.contents)
// This is essential for kernel launchers that receive data pointers
// Returns NULL if the pointer is not found in the registry
MTLBufferPtr metalGetBufferFromPointer(const void* dataPtr);

// Register a buffer in the pointer-to-buffer registry
// Called automatically by metalAllocateDevice for shared/managed buffers
void metalRegisterBuffer(MTLBufferPtr buffer);

// Unregister a buffer from the pointer-to-buffer registry
// Called automatically by metalFreeDevice
void metalUnregisterBuffer(MTLBufferPtr buffer);

// Free device buffer
void metalFreeDevice(MTLBufferPtr buffer);

// Copy host to device
void metalCopyHostToDevice(MTLBufferPtr buffer, const void* hostPtr, size_t size, size_t offset);

// Copy device to host
void metalCopyDeviceToHost(void* hostPtr, MTLBufferPtr buffer, size_t size, size_t offset);

// Copy device to device
void metalCopyDeviceToDevice(MTLBufferPtr dst, MTLBufferPtr src, size_t size, MTLCommandQueuePtr queue);

// Zero out device memory
void metalMemsetDevice(MTLBufferPtr buffer, int value, size_t size, MTLCommandQueuePtr queue);

// Get buffer pointer (for shared memory only)
void* metalGetBufferPointer(MTLBufferPtr buffer);

// Get buffer size
size_t metalGetBufferSize(MTLBufferPtr buffer);

// Synchronize buffer (for managed memory)
void metalSynchronizeBuffer(MTLBufferPtr buffer);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
// C++ wrapper class for easier usage
namespace Metal {

class MemoryManager {
public:
    static void* allocate(void* device, size_t size, MetalMemoryMode mode = METAL_MEM_SHARED);
    static void deallocate(void* buffer);
    static void copyToDevice(void* buffer, const void* hostPtr, size_t size, size_t offset = 0);
    static void copyFromDevice(void* hostPtr, void* buffer, size_t size, size_t offset = 0);
    static void copyDeviceToDevice(void* dst, void* src, size_t size, void* queue);
    static void memset(void* buffer, int value, size_t size, void* queue);
    static void* getPointer(void* buffer);
    static size_t getSize(void* buffer);
};

} // namespace Metal
#endif

#endif /* METAL_MEM_UTILS_H_ */
