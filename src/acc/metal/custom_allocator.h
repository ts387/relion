#ifndef METAL_CUSTOM_ALLOCATOR_H_
#define METAL_CUSTOM_ALLOCATOR_H_

#include "src/acc/metal/metal_settings.h"
#include "src/acc/metal/metal_mem_utils.h"
#include <vector>
#include <map>
#include <cstddef>

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#endif

#ifndef __OBJC__
typedef void* MTLDevicePtr;
typedef void* MTLBufferPtr;
typedef void* MTLEventPtr;
#else
typedef id<MTLDevice> MTLDevicePtr;
typedef id<MTLBuffer> MTLBufferPtr;
typedef id<MTLEvent> MTLEventPtr;
#endif

class MetalCustomAllocator {
public:
    // Allocation record
    class Alloc {
    public:
        MTLBufferPtr buffer;
        size_t size;
        bool inUse;
        MTLEventPtr event;  // For event-based deallocation

        Alloc() : buffer(nullptr), size(0), inUse(false), event(nullptr) {}
        Alloc(MTLBufferPtr buf, size_t sz) : buffer(buf), size(sz), inUse(true), event(nullptr) {}

        ~Alloc();

        void free();
        void* getPtr();
        size_t getSize() const { return size; }
        bool isInUse() const { return inUse; }
        void markInUse() { inUse = true; }
        void markFree() { inUse = false; }
    };

private:
    MTLDevicePtr device;
    std::vector<Alloc*> allocations;
    std::map<size_t, std::vector<Alloc*>> freeBlocks;  // Size bucket -> free allocations

    size_t totalAllocated;
    size_t totalFreed;
    size_t peakUsage;

    bool enableCaching;
    size_t cacheSizeLimit;  // Maximum size of cached buffers

    // Find a suitable free block
    Alloc* findFreeBlock(size_t size);

    // Add a block to the free list
    void addToFreeList(Alloc* alloc);

    // Remove a block from the free list
    void removeFromFreeList(Alloc* alloc);

    // Allocate a new buffer from Metal
    Alloc* allocateNew(size_t size, MetalMemoryMode mode);

public:
    MetalCustomAllocator(MTLDevicePtr dev, bool enableCache = true);
    ~MetalCustomAllocator();

    // Allocate memory
    Alloc* alloc(size_t size, MetalMemoryMode mode = METAL_MEM_SHARED);

    // Free memory (returns to pool if caching enabled)
    void free(Alloc* allocation);

    // Actually deallocate a buffer (bypass cache)
    void deallocate(Alloc* allocation);

    // Clear all cached free blocks
    void clearCache();

    // Synchronize with an event (for async operations)
    void attachEvent(Alloc* allocation, MTLEventPtr event);

    // Get statistics
    size_t getTotalAllocated() const { return totalAllocated; }
    size_t getTotalFreed() const { return totalFreed; }
    size_t getCurrentUsage() const { return totalAllocated - totalFreed; }
    size_t getPeakUsage() const { return peakUsage; }
    size_t getCachedMemory() const;

    // Enable/disable caching
    void setCachingEnabled(bool enabled) { enableCaching = enabled; }
    bool isCachingEnabled() const { return enableCaching; }

    // Set cache size limit
    void setCacheSizeLimit(size_t limit) { cacheSizeLimit = limit; }
    size_t getCacheSizeLimit() const { return cacheSizeLimit; }

    // Get device
    MTLDevicePtr getDevice() const { return device; }
};

#endif /* METAL_CUSTOM_ALLOCATOR_H_ */
