#ifndef METAL_DEVICE_STUBS_H_
#define METAL_DEVICE_STUBS_H_

#include "src/acc/metal/metal_settings.h"
#include "src/acc/metal/shortcuts.h"

#ifdef __cplusplus
extern "C" {
#endif

// Device stream type
typedef deviceStream_t streamType;

// Stream/queue creation and destruction
streamType createStream();
void destroyStream(streamType stream);

// Device synchronization
void syncDevice();
void syncStream(streamType stream);

// Device memory management
void* allocateDeviceMemory(size_t size);
void freeDeviceMemory(void* ptr);

// Memory transfer operations
void copyToDevice(void* dst, const void* src, size_t size, streamType stream);
void copyFromDevice(void* dst, const void* src, size_t size, streamType stream);
void copyDeviceToDevice(void* dst, const void* src, size_t size, streamType stream);

// Device information
int getDeviceCount();
void setDevice(int device);
int getDevice();
size_t getDeviceFreeMemory();
size_t getDeviceTotalMemory();

// Device properties
const char* getDeviceName(int device);
bool deviceSupportsDoublePrecision(int device);

#ifdef __cplusplus
}
#endif

#endif /* METAL_DEVICE_STUBS_H_ */
