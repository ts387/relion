#ifndef METAL_KERNEL_UTILS_H_
#define METAL_KERNEL_UTILS_H_

#include "src/acc/metal/metal_settings.h"
#include "src/acc/metal/metal_helper_functions.h"

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#endif

#ifdef __cplusplus

namespace MetalKernels {

// Kernel launcher helper
class KernelLauncher {
private:
    void* device;
    void* library;
    void* commandQueue;

public:
    KernelLauncher(void* dev, const char* libraryPath);
    ~KernelLauncher();

    // Get pipeline state for a kernel
    void* getPipelineState(const char* kernelName);

    // Launch a 1D kernel
    void launch1D(const char* kernelName,
                  size_t totalThreads,
                  void** buffers,
                  size_t* bufferSizes,
                  int numBuffers);

    // Launch a 2D kernel
    void launch2D(const char* kernelName,
                  size_t width, size_t height,
                  void** buffers,
                  size_t* bufferSizes,
                  int numBuffers);

    // Launch a 3D kernel
    void launch3D(const char* kernelName,
                  size_t width, size_t height, size_t depth,
                  void** buffers,
                  size_t* bufferSizes,
                  int numBuffers);

    // Synchronize
    void synchronize();

    // Get command queue
    void* getCommandQueue() { return commandQueue; }
};

// Utility kernel dispatchers
void launchMultiplyKernel(void* input, void* output, float multiplier, size_t size, void* queue);
void launchExponentiateKernel(void* array, float add, size_t size, void* queue);
void launchSoftMaskKernel(void* volume, size_t volSize, size_t xdim, size_t ydim, size_t zdim,
                          size_t xinit, size_t yinit, size_t zinit,
                          float radius, float cosine_width, void* queue);

} // namespace MetalKernels

#endif // __cplusplus

#endif /* METAL_KERNEL_UTILS_H_ */
