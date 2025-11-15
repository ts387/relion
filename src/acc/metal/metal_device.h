#ifndef METAL_DEVICE_H_
#define METAL_DEVICE_H_

#include "src/acc/metal/metal_settings.h"
#include "src/acc/metal/shortcuts.h"
#include <vector>
#include <string>

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Device management functions
int metalGetDeviceCount();
void metalSetDevice(int deviceId);
int metalGetDevice();
void* metalGetDeviceHandle(int deviceId);
void* metalGetCurrentDeviceHandle();

// Device properties
const char* metalGetDeviceName(int deviceId);
size_t metalGetDeviceTotalMemory(int deviceId);
size_t metalGetDeviceFreeMemory(int deviceId);
bool metalDeviceSupportsFeature(int deviceId, const char* featureName);

// Command queue management
void* metalCreateCommandQueue(void* device);
void metalDestroyCommandQueue(void* queue);

// Synchronization
void metalDeviceSynchronize();
void metalQueueSynchronize(void* queue);

// Initialization and cleanup
void metalInitialize();
void metalFinalize();

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
// C++ wrapper class
namespace Metal {

struct DeviceInfo {
    std::string name;
    size_t totalMemory;
    bool isLowPower;
    bool isRemovable;
    bool hasUnifiedMemory;
    bool supportsFamily(int familyVersion);
};

class DeviceManager {
public:
    static void initialize();
    static void finalize();

    static int getDeviceCount();
    static void setDevice(int deviceId);
    static int getCurrentDevice();

    static void* getDevice(int deviceId);
    static void* getCurrentDeviceHandle();

    static DeviceInfo getDeviceInfo(int deviceId);
    static std::string getDeviceName(int deviceId);
    static size_t getTotalMemory(int deviceId);
    static size_t getFreeMemory(int deviceId);

    static void* createCommandQueue(void* device = nullptr);
    static void destroyCommandQueue(void* queue);

    static void synchronize();
    static void synchronizeQueue(void* queue);

private:
    static int currentDevice;
    static bool initialized;
};

} // namespace Metal
#endif

#endif /* METAL_DEVICE_H_ */
