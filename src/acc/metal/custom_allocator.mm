#import "src/acc/metal/custom_allocator.h"
#import "src/acc/metal/metal_mem_utils.h"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <algorithm>
#include <cstdio>

// Alloc implementation
MetalCustomAllocator::Alloc::~Alloc() {
    free();
}

void MetalCustomAllocator::Alloc::free() {
    if (buffer) {
        metalFreeDevice(buffer);
        buffer = nullptr;
    }
    if (event) {
        id<MTLEvent> mtlEvent = (__bridge_transfer id<MTLEvent>)event;
        mtlEvent = nil;
        event = nullptr;
    }
    size = 0;
    inUse = false;
}

void* MetalCustomAllocator::Alloc::getPtr() {
    return metalGetBufferPointer(buffer);
}

// MetalCustomAllocator implementation
MetalCustomAllocator::MetalCustomAllocator(MTLDevicePtr dev, bool enableCache)
    : device(dev)
    , totalAllocated(0)
    , totalFreed(0)
    , peakUsage(0)
    , enableCaching(enableCache)
    , cacheSizeLimit(512 * 1024 * 1024)  // 512MB default cache limit
{
    if (!device) {
        fprintf(stderr, "Metal ERROR: Invalid device in MetalCustomAllocator constructor\n");
    }
}

MetalCustomAllocator::~MetalCustomAllocator() {
    clearCache();

    // Free all allocations
    for (Alloc* alloc : allocations) {
        if (alloc) {
            delete alloc;
        }
    }
    allocations.clear();
}

MetalCustomAllocator::Alloc* MetalCustomAllocator::findFreeBlock(size_t size) {
    if (!enableCaching) {
        return nullptr;
    }

    // Look for an exact size match first
    auto it = freeBlocks.find(size);
    if (it != freeBlocks.end() && !it->second.empty()) {
        Alloc* alloc = it->second.back();
        it->second.pop_back();
        if (it->second.empty()) {
            freeBlocks.erase(it);
        }
        return alloc;
    }

    // Look for a larger block (find smallest block that fits)
    Alloc* bestFit = nullptr;
    size_t bestFitSize = SIZE_MAX;

    for (auto& pair : freeBlocks) {
        if (pair.first >= size && pair.first < bestFitSize && !pair.second.empty()) {
            bestFit = pair.second.back();
            bestFitSize = pair.first;
        }
    }

    if (bestFit) {
        removeFromFreeList(bestFit);
        return bestFit;
    }

    return nullptr;
}

void MetalCustomAllocator::addToFreeList(Alloc* alloc) {
    if (!alloc) return;

    alloc->markFree();
    freeBlocks[alloc->getSize()].push_back(alloc);
}

void MetalCustomAllocator::removeFromFreeList(Alloc* alloc) {
    if (!alloc) return;

    auto it = freeBlocks.find(alloc->getSize());
    if (it != freeBlocks.end()) {
        auto& vec = it->second;
        vec.erase(std::remove(vec.begin(), vec.end(), alloc), vec.end());
        if (vec.empty()) {
            freeBlocks.erase(it);
        }
    }
}

MetalCustomAllocator::Alloc* MetalCustomAllocator::allocateNew(size_t size, MetalMemoryMode mode) {
    MTLBufferPtr buffer = (MTLBufferPtr)metalAllocateDevice(device, size, mode);
    if (!buffer) {
        fprintf(stderr, "Metal ERROR: Failed to allocate buffer of size %zu\n", size);
        return nullptr;
    }

    Alloc* alloc = new Alloc(buffer, size);
    allocations.push_back(alloc);

    totalAllocated += size;
    if (getCurrentUsage() > peakUsage) {
        peakUsage = getCurrentUsage();
    }

#ifdef DEBUG_METAL
    fprintf(stderr, "Metal ALLOC: Allocated %zu bytes (total: %zu, peak: %zu)\n",
            size, getCurrentUsage(), peakUsage);
#endif

    return alloc;
}

MetalCustomAllocator::Alloc* MetalCustomAllocator::alloc(size_t size, MetalMemoryMode mode) {
    if (size == 0) {
        fprintf(stderr, "Metal WARNING: Attempt to allocate 0 bytes\n");
        return nullptr;
    }

    // Try to find a free block in the cache
    Alloc* alloc = findFreeBlock(size);

    if (alloc) {
        // Found a cached block, reuse it
        alloc->markInUse();
#ifdef DEBUG_METAL
        fprintf(stderr, "Metal ALLOC: Reused cached buffer of size %zu\n", alloc->getSize());
#endif
        return alloc;
    }

    // No suitable cached block, allocate new
    return allocateNew(size, mode);
}

void MetalCustomAllocator::free(Alloc* allocation) {
    if (!allocation) {
        return;
    }

    if (!allocation->isInUse()) {
        fprintf(stderr, "Metal WARNING: Attempt to free already-freed allocation\n");
        return;
    }

    totalFreed += allocation->getSize();

#ifdef DEBUG_METAL
    fprintf(stderr, "Metal FREE: Freed %zu bytes (current usage: %zu)\n",
            allocation->getSize(), getCurrentUsage());
#endif

    // If caching is enabled and we're under the cache limit, add to free list
    if (enableCaching && getCachedMemory() < cacheSizeLimit) {
        addToFreeList(allocation);
    } else {
        // Otherwise, actually deallocate
        deallocate(allocation);
    }
}

void MetalCustomAllocator::deallocate(Alloc* allocation) {
    if (!allocation) {
        return;
    }

    // Remove from free list if it's there
    if (!allocation->isInUse()) {
        removeFromFreeList(allocation);
    }

    // Remove from allocations list
    allocations.erase(std::remove(allocations.begin(), allocations.end(), allocation), allocations.end());

    // Actually free the buffer
    delete allocation;
}

void MetalCustomAllocator::clearCache() {
    std::vector<Alloc*> toDelete;

    // Collect all free blocks
    for (auto& pair : freeBlocks) {
        toDelete.insert(toDelete.end(), pair.second.begin(), pair.second.end());
    }

    // Clear the free blocks map
    freeBlocks.clear();

    // Deallocate all free blocks
    for (Alloc* alloc : toDelete) {
        deallocate(alloc);
    }

#ifdef DEBUG_METAL
    fprintf(stderr, "Metal CACHE: Cleared cache, deallocated %zu blocks\n", toDelete.size());
#endif
}

void MetalCustomAllocator::attachEvent(Alloc* allocation, MTLEventPtr event) {
    if (allocation && event) {
        allocation->event = event;
    }
}

size_t MetalCustomAllocator::getCachedMemory() const {
    size_t cached = 0;
    for (const auto& pair : freeBlocks) {
        cached += pair.first * pair.second.size();
    }
    return cached;
}
