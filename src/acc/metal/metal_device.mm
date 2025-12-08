#import "src/acc/metal/metal_device.h"
#import "src/acc/metal/metal_helper_functions.h"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <vector>
#include <string>
#include <mutex>
#include <set>

// Global state
static NSArray<id<MTLDevice>>* g_devices = nil;
static int g_currentDevice = 0;
static bool g_initialized = false;

// Track active command queues for synchronization
static std::set<id<MTLCommandQueue>> g_activeQueues;
static std::mutex g_queuesMutex;

extern "C" {

void metalInitialize() {
    if (g_initialized) {
        return;
    }

    @autoreleasepool {
        g_devices = MTLCopyAllDevices();

        if (!g_devices || g_devices.count == 0) {
            // Fall back to system default device
            id<MTLDevice> defaultDev = MTLCreateSystemDefaultDevice();
            if (defaultDev) {
                g_devices = @[defaultDev];
            } else {
                fprintf(stderr, "Metal ERROR: No Metal devices found on this system\n");
                g_devices = @[];
            }
        }

        g_currentDevice = 0;
        g_initialized = true;

        fprintf(stderr, "Metal INFO: Found %lu Metal device(s)\n", (unsigned long)g_devices.count);
        for (NSUInteger i = 0; i < g_devices.count; i++) {
            id<MTLDevice> dev = g_devices[i];
            fprintf(stderr, "  Device %lu: %s%s%s\n",
                    (unsigned long)i,
                    [dev.name UTF8String],
                    dev.isLowPower ? " (Low Power)" : "",
                    dev.hasUnifiedMemory ? " (Unified Memory)" : "");
        }
    }
}

void metalFinalize() {
    @autoreleasepool {
        g_devices = nil;
        g_initialized = false;
    }
}

int metalGetDeviceCount() {
    if (!g_initialized) {
        metalInitialize();
    }
    return (int)g_devices.count;
}

void metalSetDevice(int deviceId) {
    if (!g_initialized) {
        metalInitialize();
    }

    if (deviceId < 0 || deviceId >= (int)g_devices.count) {
        fprintf(stderr, "Metal ERROR: Invalid device ID %d (valid range: 0-%lu)\n",
                deviceId, (unsigned long)g_devices.count - 1);
        return;
    }

    g_currentDevice = deviceId;
}

int metalGetDevice() {
    if (!g_initialized) {
        metalInitialize();
    }
    return g_currentDevice;
}

void* metalGetDeviceHandle(int deviceId) {
    if (!g_initialized) {
        metalInitialize();
    }

    if (deviceId < 0 || deviceId >= (int)g_devices.count) {
        fprintf(stderr, "Metal ERROR: Invalid device ID %d\n", deviceId);
        return nullptr;
    }

    return (__bridge void*)g_devices[deviceId];
}

void* metalGetCurrentDeviceHandle() {
    return metalGetDeviceHandle(g_currentDevice);
}

const char* metalGetDeviceName(int deviceId) {
    if (!g_initialized) {
        metalInitialize();
    }

    if (deviceId < 0 || deviceId >= (int)g_devices.count) {
        return "Invalid Device";
    }

    @autoreleasepool {
        id<MTLDevice> device = g_devices[deviceId];
        // Use thread_local to avoid race conditions when multiple threads
        // request device names simultaneously
        thread_local char nameBuf[256];
        strncpy(nameBuf, [device.name UTF8String], sizeof(nameBuf) - 1);
        nameBuf[sizeof(nameBuf) - 1] = '\0';
        return nameBuf;
    }
}

size_t metalGetDeviceTotalMemory(int deviceId) {
    if (!g_initialized) {
        metalInitialize();
    }

    if (deviceId < 0 || deviceId >= (int)g_devices.count) {
        return 0;
    }

    @autoreleasepool {
        id<MTLDevice> device = g_devices[deviceId];
        return (size_t)device.recommendedMaxWorkingSetSize;
    }
}

size_t metalGetDeviceFreeMemory(int deviceId) {
    if (!g_initialized) {
        metalInitialize();
    }

    if (deviceId < 0 || deviceId >= (int)g_devices.count) {
        return 0;
    }

    @autoreleasepool {
        id<MTLDevice> device = g_devices[deviceId];
        size_t total = device.recommendedMaxWorkingSetSize;
        size_t used = device.currentAllocatedSize;
        return (used < total) ? (total - used) : 0;
    }
}

bool metalDeviceSupportsFeature(int deviceId, const char* featureName) {
    if (!g_initialized) {
        metalInitialize();
    }

    if (deviceId < 0 || deviceId >= (int)g_devices.count) {
        return false;
    }

    @autoreleasepool {
        id<MTLDevice> device = g_devices[deviceId];

        // Check various features
        if (strcmp(featureName, "unified_memory") == 0) {
            return device.hasUnifiedMemory;
        } else if (strcmp(featureName, "metal3") == 0) {
            if (@available(macOS 13.0, *)) {
                return [device supportsFamily:MTLGPUFamilyMetal3];
            }
            return false;
        } else if (strcmp(featureName, "apple_gpu") == 0) {
            if (@available(macOS 13.0, *)) {
                return [device supportsFamily:MTLGPUFamilyApple7] ||
                       [device supportsFamily:MTLGPUFamilyApple8] ||
                       [device supportsFamily:MTLGPUFamilyApple9];
            }
            return false;
        }

        return false;
    }
}

void* metalCreateCommandQueue(void* device) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice;

        if (device == nullptr) {
            mtlDevice = (__bridge id<MTLDevice>)metalGetCurrentDeviceHandle();
        } else {
            mtlDevice = (__bridge id<MTLDevice>)device;
        }

        if (!mtlDevice) {
            fprintf(stderr, "Metal ERROR: Invalid device in metalCreateCommandQueue\n");
            return nullptr;
        }

        id<MTLCommandQueue> queue = [mtlDevice newCommandQueue];
        if (!queue) {
            fprintf(stderr, "Metal ERROR: Failed to create command queue\n");
            return nullptr;
        }

        // Track this queue for global synchronization
        {
            std::lock_guard<std::mutex> lock(g_queuesMutex);
            g_activeQueues.insert(queue);
        }

        return (__bridge_retained void*)queue;
    }
}

void metalDestroyCommandQueue(void* queue) {
    @autoreleasepool {
        if (queue) {
            id<MTLCommandQueue> mtlQueue = (__bridge_transfer id<MTLCommandQueue>)queue;

            // Remove from tracking set
            {
                std::lock_guard<std::mutex> lock(g_queuesMutex);
                g_activeQueues.erase(mtlQueue);
            }

            mtlQueue = nil;
        }
    }
}

void metalDeviceSynchronize() {
    // Synchronize all tracked command queues by submitting and waiting
    // on empty command buffers for each queue
    @autoreleasepool {
        std::lock_guard<std::mutex> lock(g_queuesMutex);

        for (id<MTLCommandQueue> queue : g_activeQueues) {
            if (queue) {
                id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
                if (cmdBuffer) {
                    [cmdBuffer commit];
                    [cmdBuffer waitUntilCompleted];
                }
            }
        }
    }
}

void metalQueueSynchronize(void* queue) {
    @autoreleasepool {
        if (!queue) {
            return;
        }

        id<MTLCommandQueue> mtlQueue = (__bridge id<MTLCommandQueue>)queue;

        // Submit a dummy command buffer and wait for it
        id<MTLCommandBuffer> cmdBuffer = [mtlQueue commandBuffer];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
    }
}

} // extern "C"

// C++ wrapper implementation
namespace Metal {

int DeviceManager::currentDevice = 0;
bool DeviceManager::initialized = false;

void DeviceManager::initialize() {
    metalInitialize();
    initialized = true;
}

void DeviceManager::finalize() {
    metalFinalize();
    initialized = false;
}

int DeviceManager::getDeviceCount() {
    return metalGetDeviceCount();
}

void DeviceManager::setDevice(int deviceId) {
    metalSetDevice(deviceId);
    currentDevice = deviceId;
}

int DeviceManager::getCurrentDevice() {
    return metalGetDevice();
}

void* DeviceManager::getDevice(int deviceId) {
    return metalGetDeviceHandle(deviceId);
}

void* DeviceManager::getCurrentDeviceHandle() {
    return metalGetCurrentDeviceHandle();
}

DeviceInfo DeviceManager::getDeviceInfo(int deviceId) {
    DeviceInfo info;

    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)metalGetDeviceHandle(deviceId);
        if (device) {
            info.name = [device.name UTF8String];
            info.totalMemory = device.recommendedMaxWorkingSetSize;
            info.isLowPower = device.isLowPower;
            info.isRemovable = device.isRemovable;
            info.hasUnifiedMemory = device.hasUnifiedMemory;
        }
    }

    return info;
}

std::string DeviceManager::getDeviceName(int deviceId) {
    return std::string(metalGetDeviceName(deviceId));
}

size_t DeviceManager::getTotalMemory(int deviceId) {
    return metalGetDeviceTotalMemory(deviceId);
}

size_t DeviceManager::getFreeMemory(int deviceId) {
    return metalGetDeviceFreeMemory(deviceId);
}

void* DeviceManager::createCommandQueue(void* device) {
    return metalCreateCommandQueue(device);
}

void DeviceManager::destroyCommandQueue(void* queue) {
    metalDestroyCommandQueue(queue);
}

void DeviceManager::synchronize() {
    metalDeviceSynchronize();
}

void DeviceManager::synchronizeQueue(void* queue) {
    metalQueueSynchronize(queue);
}

} // namespace Metal
