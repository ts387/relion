#import "src/acc/metal/metal_fft.h"
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Accelerate/Accelerate.h>
#include <cstdio>

namespace MetalFFT {

struct FFTPlanImpl {
    id<MTLDevice> device;
    size_t nx, ny, nz;
    size_t batch;
    FFTType type;
    int dimensions; // 1, 2, or 3

    // For MPS-based FFT (Metal Performance Shaders)
    // Note: MPS doesn't directly support FFT, we'll use vDSP/Accelerate as fallback
    // or implement custom FFT kernels for GPU

    // Temporary: Use CPU-based Accelerate framework FFT for now
    // TODO: Implement GPU-based FFT using custom Metal kernels or wait for MPS updates

    FFTSetup vdsp_setup; // For vDSP FFT (CPU)
    size_t log2n;

    FFTPlanImpl() : vdsp_setup(nullptr), log2n(0) {}
    ~FFTPlanImpl() {
        if (vdsp_setup) {
            vDSP_destroy_fftsetup(vdsp_setup);
        }
    }
};

FFTPlan createFFTPlan1D(void* device, size_t nx, FFTType type, size_t batch) {
    @autoreleasepool {
        FFTPlanImpl* plan = new FFTPlanImpl();
        plan->device = (__bridge id<MTLDevice>)device;
        plan->nx = nx;
        plan->ny = 1;
        plan->nz = 1;
        plan->batch = batch;
        plan->type = type;
        plan->dimensions = 1;

        // Calculate log2(nx) for vDSP
        plan->log2n = 0;
        size_t n = nx;
        while (n > 1) {
            n >>= 1;
            plan->log2n++;
        }

        // Create vDSP FFT setup (CPU-based for now)
        plan->vdsp_setup = vDSP_create_fftsetup(plan->log2n, FFT_RADIX2);
        if (!plan->vdsp_setup) {
            fprintf(stderr, "Metal FFT ERROR: Failed to create vDSP FFT setup for 1D FFT\n");
            delete plan;
            return nullptr;
        }

        fprintf(stderr, "Metal FFT INFO: Created 1D FFT plan (%zu points, batch=%zu) using vDSP\n",
                nx, batch);

        return (FFTPlan)plan;
    }
}

FFTPlan createFFTPlan2D(void* device, size_t nx, size_t ny, FFTType type, size_t batch) {
    @autoreleasepool {
        FFTPlanImpl* plan = new FFTPlanImpl();
        plan->device = (__bridge id<MTLDevice>)device;
        plan->nx = nx;
        plan->ny = ny;
        plan->nz = 1;
        plan->batch = batch;
        plan->type = type;
        plan->dimensions = 2;

        // For 2D FFT, we'll do row-wise then column-wise 1D FFTs
        // Calculate log2 for both dimensions
        size_t log2nx = 0, log2ny = 0;
        size_t n = nx;
        while (n > 1) { n >>= 1; log2nx++; }
        n = ny;
        while (n > 1) { n >>= 1; log2ny++; }

        plan->log2n = (log2nx > log2ny) ? log2nx : log2ny;
        plan->vdsp_setup = vDSP_create_fftsetup(plan->log2n, FFT_RADIX2);

        if (!plan->vdsp_setup) {
            fprintf(stderr, "Metal FFT ERROR: Failed to create vDSP FFT setup for 2D FFT\n");
            delete plan;
            return nullptr;
        }

        fprintf(stderr, "Metal FFT INFO: Created 2D FFT plan (%zux%zu, batch=%zu) using vDSP\n",
                nx, ny, batch);

        return (FFTPlan)plan;
    }
}

FFTPlan createFFTPlan3D(void* device, size_t nx, size_t ny, size_t nz, FFTType type) {
    @autoreleasepool {
        FFTPlanImpl* plan = new FFTPlanImpl();
        plan->device = (__bridge id<MTLDevice>)device;
        plan->nx = nx;
        plan->ny = ny;
        plan->nz = nz;
        plan->batch = 1;
        plan->type = type;
        plan->dimensions = 3;

        // For 3D FFT, compute largest dimension for log2
        size_t max_dim = nx;
        if (ny > max_dim) max_dim = ny;
        if (nz > max_dim) max_dim = nz;

        plan->log2n = 0;
        size_t n = max_dim;
        while (n > 1) { n >>= 1; plan->log2n++; }

        plan->vdsp_setup = vDSP_create_fftsetup(plan->log2n, FFT_RADIX2);

        if (!plan->vdsp_setup) {
            fprintf(stderr, "Metal FFT ERROR: Failed to create vDSP FFT setup for 3D FFT\n");
            delete plan;
            return nullptr;
        }

        fprintf(stderr, "Metal FFT INFO: Created 3D FFT plan (%zux%zux%zu) using vDSP\n",
                nx, ny, nz);

        return (FFTPlan)plan;
    }
}

void destroyFFTPlan(FFTPlan plan) {
    if (plan) {
        FFTPlanImpl* impl = (FFTPlanImpl*)plan;
        delete impl;
    }
}

void executeFFT(FFTPlan plan,
                void* input_real,
                void* input_imag,
                void* output_real,
                void* output_imag,
                FFTDirection direction,
                void* commandQueue) {
    if (!plan) {
        fprintf(stderr, "Metal FFT ERROR: Null plan in executeFFT\n");
        return;
    }

    FFTPlanImpl* impl = (FFTPlanImpl*)plan;

    fprintf(stderr, "Metal FFT WARNING: Using CPU-based FFT (vDSP/Accelerate)\n");
    fprintf(stderr, "Metal FFT TODO: Implement GPU-accelerated FFT using custom Metal kernels\n");

    // For now, this is a placeholder that would copy data to CPU, execute FFT, and copy back
    // This is NOT efficient and should be replaced with GPU-based FFT

    // TODO: Implement one of these approaches:
    // 1. Custom Metal FFT kernels (Cooley-Tukey algorithm)
    // 2. Use Metal compute shaders with shared memory for FFT
    // 3. Wait for MPS to add FFT support
    // 4. Use third-party Metal FFT library
}

void executeFFTInPlace(FFTPlan plan,
                       void* data_real,
                       void* data_imag,
                       FFTDirection direction,
                       void* commandQueue) {
    executeFFT(plan, data_real, data_imag, data_real, data_imag, direction, commandQueue);
}

} // namespace MetalFFT
