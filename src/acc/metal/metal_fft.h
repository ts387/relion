#ifndef METAL_FFT_H_
#define METAL_FFT_H_

#include "src/acc/metal/metal_settings.h"
#include <cstddef>

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

#ifdef __cplusplus

namespace MetalFFT {

enum FFTDirection {
    FFT_FORWARD = 0,
    FFT_INVERSE = 1
};

enum FFTType {
    FFT_R2C = 0,    // Real to complex
    FFT_C2R = 1,    // Complex to real
    FFT_C2C = 2     // Complex to complex
};

// FFT plan (opaque handle)
typedef void* FFTPlan;

// Create FFT plan for 1D transform
FFTPlan createFFTPlan1D(void* device, size_t nx, FFTType type, size_t batch = 1);

// Create FFT plan for 2D transform
FFTPlan createFFTPlan2D(void* device, size_t nx, size_t ny, FFTType type, size_t batch = 1);

// Create FFT plan for 3D transform
FFTPlan createFFTPlan3D(void* device, size_t nx, size_t ny, size_t nz, FFTType type);

// Destroy FFT plan
void destroyFFTPlan(FFTPlan plan);

// Execute FFT
void executeFFT(FFTPlan plan,
                void* input_real,
                void* input_imag,
                void* output_real,
                void* output_imag,
                FFTDirection direction,
                void* commandQueue);

// Execute in-place FFT
void executeFFTInPlace(FFTPlan plan,
                       void* data_real,
                       void* data_imag,
                       FFTDirection direction,
                       void* commandQueue);

} // namespace MetalFFT

#endif // __cplusplus

#endif /* METAL_FFT_H_ */
