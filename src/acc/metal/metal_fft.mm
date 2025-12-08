#import "src/acc/metal/metal_fft.h"
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Accelerate/Accelerate.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

namespace MetalFFT {

// Helper function to compute log2
static inline uint computeLog2(size_t n) {
    uint log2n = 0;
    size_t val = n;
    while (val > 1) {
        val >>= 1;
        log2n++;
    }
    return log2n;
}

// Check if n is power of 2
static inline bool isPowerOfTwo(size_t n) {
    return (n > 0) && ((n & (n - 1)) == 0);
}

struct FFTPlanImpl {
    id<MTLDevice> device;
    id<MTLLibrary> library;
    id<MTLCommandQueue> commandQueue;

    size_t nx, ny, nz;
    size_t batch;
    FFTType type;
    int dimensions; // 1, 2, or 3

    // Compute pipeline states for FFT kernels
    id<MTLComputePipelineState> bitReversalPipeline;
    id<MTLComputePipelineState> butterflyPipeline;
    id<MTLComputePipelineState> scalePipeline;
    id<MTLComputePipelineState> copyPipeline;
    id<MTLComputePipelineState> packRealPipeline;
    id<MTLComputePipelineState> extractRealPipeline;
    id<MTLComputePipelineState> r2cPostprocessPipeline;
    id<MTLComputePipelineState> c2rPreprocessPipeline;

    // 2D FFT pipelines
    id<MTLComputePipelineState> bitReversalRowsPipeline;
    id<MTLComputePipelineState> bitReversalColsPipeline;
    id<MTLComputePipelineState> butterflyRowsPipeline;
    id<MTLComputePipelineState> butterflyColsPipeline;

    // 3D FFT pipelines
    id<MTLComputePipelineState> butterflyZPipeline;

    // Temporary buffers
    id<MTLBuffer> tempReal;
    id<MTLBuffer> tempImag;

    uint log2nx, log2ny, log2nz;

    // Fallback to CPU for non-power-of-2 sizes
    bool useCPUFallback;
    FFTSetup vdsp_setup;

    // DFT setups for arbitrary sizes (non-power-of-2)
    vDSP_DFT_Setup dft_forward;
    vDSP_DFT_Setup dft_inverse;

    FFTPlanImpl() : library(nil), commandQueue(nil),
                    bitReversalPipeline(nil), butterflyPipeline(nil),
                    scalePipeline(nil), copyPipeline(nil),
                    tempReal(nil), tempImag(nil),
                    useCPUFallback(false), vdsp_setup(nullptr),
                    dft_forward(nullptr), dft_inverse(nullptr) {}

    ~FFTPlanImpl() {
        if (vdsp_setup) {
            vDSP_destroy_fftsetup(vdsp_setup);
        }
        if (dft_forward) {
            vDSP_DFT_DestroySetup(dft_forward);
        }
        if (dft_inverse) {
            vDSP_DFT_DestroySetup(dft_inverse);
        }
    }

    bool initializePipelines() {
        NSError* error = nil;

        // Load the default library (compiled shaders)
        library = [device newDefaultLibrary];
        if (!library) {
            fprintf(stderr, "Metal FFT ERROR: Failed to load default Metal library\n");
            return false;
        }

        // Create command queue
        commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            fprintf(stderr, "Metal FFT ERROR: Failed to create command queue\n");
            return false;
        }

        // Create pipeline states for each kernel
        auto createPipeline = [&](const char* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
            if (!function) {
                fprintf(stderr, "Metal FFT WARNING: Kernel '%s' not found\n", name);
                return nil;
            }
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            if (error) {
                fprintf(stderr, "Metal FFT ERROR: Failed to create pipeline for '%s': %s\n",
                        name, [[error localizedDescription] UTF8String]);
                return nil;
            }
            return pipeline;
        };

        // Create 1D FFT pipelines
        bitReversalPipeline = createPipeline("metal_kernel_fft_bit_reversal");
        butterflyPipeline = createPipeline("metal_kernel_fft_butterfly");
        scalePipeline = createPipeline("metal_kernel_fft_scale");
        copyPipeline = createPipeline("metal_kernel_fft_copy");
        packRealPipeline = createPipeline("metal_kernel_fft_pack_real_to_complex");
        extractRealPipeline = createPipeline("metal_kernel_fft_extract_real");
        r2cPostprocessPipeline = createPipeline("metal_kernel_fft_r2c_postprocess");
        c2rPreprocessPipeline = createPipeline("metal_kernel_fft_c2r_preprocess");

        // Create 2D FFT pipelines
        bitReversalRowsPipeline = createPipeline("metal_kernel_fft2d_bit_reversal_rows");
        bitReversalColsPipeline = createPipeline("metal_kernel_fft2d_bit_reversal_cols");
        butterflyRowsPipeline = createPipeline("metal_kernel_fft2d_rows_butterfly");
        butterflyColsPipeline = createPipeline("metal_kernel_fft2d_cols_butterfly");

        // Create 3D FFT pipelines
        butterflyZPipeline = createPipeline("metal_kernel_fft3d_z_butterfly");

        // Verify critical pipelines
        if (!bitReversalPipeline || !butterflyPipeline || !scalePipeline) {
            fprintf(stderr, "Metal FFT ERROR: Failed to create critical FFT pipelines\n");
            return false;
        }

        return true;
    }

    void allocateTemporaryBuffers() {
        size_t totalSize = nx * ny * nz * batch;

        tempReal = [device newBufferWithLength:totalSize * sizeof(float)
                                       options:MTLResourceStorageModeShared];
        tempImag = [device newBufferWithLength:totalSize * sizeof(float)
                                       options:MTLResourceStorageModeShared];

        if (!tempReal || !tempImag) {
            fprintf(stderr, "Metal FFT WARNING: Failed to allocate temporary buffers\n");
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
        plan->log2nx = computeLog2(nx);
        plan->log2ny = 0;
        plan->log2nz = 0;

        // Check if we can use GPU FFT (requires power of 2)
        if (!isPowerOfTwo(nx)) {
            fprintf(stderr, "Metal FFT INFO: Non-power-of-2 size (%zu), using vDSP DFT fallback\n", nx);
            plan->useCPUFallback = true;

            // Create DFT setups for arbitrary-length transforms
            // vDSP_DFT supports any length, not just power-of-2
            plan->dft_forward = vDSP_DFT_zop_CreateSetup(nullptr, nx, vDSP_DFT_FORWARD);
            plan->dft_inverse = vDSP_DFT_zop_CreateSetup(nullptr, nx, vDSP_DFT_INVERSE);

            if (!plan->dft_forward || !plan->dft_inverse) {
                fprintf(stderr, "Metal FFT ERROR: Failed to create vDSP DFT setup for size %zu\n", nx);
                delete plan;
                return nullptr;
            }
        } else {
            plan->useCPUFallback = false;
            if (!plan->initializePipelines()) {
                fprintf(stderr, "Metal FFT WARNING: Failed to initialize GPU pipelines, using CPU fallback\n");
                plan->useCPUFallback = true;
                plan->vdsp_setup = vDSP_create_fftsetup(plan->log2nx, FFT_RADIX2);
            } else {
                plan->allocateTemporaryBuffers();
                fprintf(stderr, "Metal FFT INFO: Created 1D GPU FFT plan (%zu points, batch=%zu)\n", nx, batch);
            }
        }

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
        plan->log2nx = computeLog2(nx);
        plan->log2ny = computeLog2(ny);
        plan->log2nz = 0;

        if (!isPowerOfTwo(nx) || !isPowerOfTwo(ny)) {
            fprintf(stderr, "Metal FFT INFO: Non-power-of-2 2D size (%zux%zu), using vDSP DFT fallback\n", nx, ny);
            plan->useCPUFallback = true;

            // For 2D FFT, we'll do row and column FFTs separately
            // Create DFT setup for the larger dimension (we'll reuse for both)
            size_t maxDim = (nx > ny) ? nx : ny;
            plan->dft_forward = vDSP_DFT_zop_CreateSetup(nullptr, maxDim, vDSP_DFT_FORWARD);
            plan->dft_inverse = vDSP_DFT_zop_CreateSetup(nullptr, maxDim, vDSP_DFT_INVERSE);

            if (!plan->dft_forward || !plan->dft_inverse) {
                fprintf(stderr, "Metal FFT ERROR: Failed to create vDSP DFT setup for 2D size %zux%zu\n", nx, ny);
                delete plan;
                return nullptr;
            }
        } else {
            plan->useCPUFallback = false;
            if (!plan->initializePipelines()) {
                plan->useCPUFallback = true;
                size_t max_log2 = (plan->log2nx > plan->log2ny) ? plan->log2nx : plan->log2ny;
                plan->vdsp_setup = vDSP_create_fftsetup(max_log2, FFT_RADIX2);
            } else {
                plan->allocateTemporaryBuffers();
                fprintf(stderr, "Metal FFT INFO: Created 2D GPU FFT plan (%zux%zu, batch=%zu)\n", nx, ny, batch);
            }
        }

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
        plan->log2nx = computeLog2(nx);
        plan->log2ny = computeLog2(ny);
        plan->log2nz = computeLog2(nz);

        if (!isPowerOfTwo(nx) || !isPowerOfTwo(ny) || !isPowerOfTwo(nz)) {
            fprintf(stderr, "Metal FFT INFO: Non-power-of-2 3D size (%zux%zux%zu), using vDSP DFT fallback\n",
                    nx, ny, nz);
            plan->useCPUFallback = true;

            // Create DFT setup for the largest dimension
            size_t max_dim = nx;
            if (ny > max_dim) max_dim = ny;
            if (nz > max_dim) max_dim = nz;

            plan->dft_forward = vDSP_DFT_zop_CreateSetup(nullptr, max_dim, vDSP_DFT_FORWARD);
            plan->dft_inverse = vDSP_DFT_zop_CreateSetup(nullptr, max_dim, vDSP_DFT_INVERSE);

            if (!plan->dft_forward || !plan->dft_inverse) {
                fprintf(stderr, "Metal FFT ERROR: Failed to create vDSP DFT setup for 3D size %zux%zux%zu\n",
                        nx, ny, nz);
                delete plan;
                return nullptr;
            }
        } else {
            plan->useCPUFallback = false;
            if (!plan->initializePipelines()) {
                plan->useCPUFallback = true;
                size_t max_dim = nx;
                if (ny > max_dim) max_dim = ny;
                if (nz > max_dim) max_dim = nz;
                plan->vdsp_setup = vDSP_create_fftsetup(computeLog2(max_dim), FFT_RADIX2);
            } else {
                plan->allocateTemporaryBuffers();
                fprintf(stderr, "Metal FFT INFO: Created 3D GPU FFT plan (%zux%zux%zu)\n", nx, ny, nz);
            }
        }

        return (FFTPlan)plan;
    }
}

void destroyFFTPlan(FFTPlan plan) {
    if (plan) {
        FFTPlanImpl* impl = (FFTPlanImpl*)plan;
        delete impl;
    }
}

// ============================================================================
// CPU Fallback FFT using vDSP for non-power-of-2 sizes
// ============================================================================

// Execute 1D DFT on CPU for non-power-of-2 sizes
static void execute1DCPUFFT(FFTPlanImpl* impl,
                            id<MTLBuffer> inReal, id<MTLBuffer> inImag,
                            id<MTLBuffer> outReal, id<MTLBuffer> outImag,
                            FFTDirection direction) {
    size_t n = impl->nx;

    // Get pointers to buffer contents (shared memory allows CPU access)
    float* srcReal = (float*)inReal.contents;
    float* srcImag = (float*)inImag.contents;
    float* dstReal = (float*)outReal.contents;
    float* dstImag = (float*)outImag.contents;

    if (!srcReal || !srcImag || !dstReal || !dstImag) {
        fprintf(stderr, "Metal FFT ERROR: Buffer contents not accessible for CPU FFT\n");
        return;
    }

    // Select the appropriate DFT setup
    vDSP_DFT_Setup dftSetup = (direction == FFT_FORWARD) ? impl->dft_forward : impl->dft_inverse;

    // Process each batch
    for (size_t b = 0; b < impl->batch; b++) {
        size_t offset = b * n;

        // Execute the DFT
        vDSP_DFT_Execute(dftSetup,
                         srcReal + offset, srcImag + offset,
                         dstReal + offset, dstImag + offset);

        // Scale for inverse FFT (vDSP doesn't auto-scale)
        if (direction == FFT_INVERSE) {
            float scale = 1.0f / (float)n;
            vDSP_vsmul(dstReal + offset, 1, &scale, dstReal + offset, 1, n);
            vDSP_vsmul(dstImag + offset, 1, &scale, dstImag + offset, 1, n);
        }
    }
}

// Execute 2D DFT on CPU (row-column decomposition)
static void execute2DCPUFFT(FFTPlanImpl* impl,
                            id<MTLBuffer> inReal, id<MTLBuffer> inImag,
                            id<MTLBuffer> outReal, id<MTLBuffer> outImag,
                            FFTDirection direction) {
    size_t nx = impl->nx;
    size_t ny = impl->ny;

    float* srcReal = (float*)inReal.contents;
    float* srcImag = (float*)inImag.contents;
    float* dstReal = (float*)outReal.contents;
    float* dstImag = (float*)outImag.contents;

    if (!srcReal || !srcImag || !dstReal || !dstImag) {
        fprintf(stderr, "Metal FFT ERROR: Buffer contents not accessible for CPU FFT\n");
        return;
    }

    // Allocate temporary buffers for row/column processing
    std::vector<float> tempReal(nx * ny);
    std::vector<float> tempImag(nx * ny);
    std::vector<float> rowReal(nx);
    std::vector<float> rowImag(nx);
    std::vector<float> colReal(ny);
    std::vector<float> colImag(ny);

    // Create separate setups for row and column transforms if sizes differ
    vDSP_DFT_Setup rowSetupFwd = vDSP_DFT_zop_CreateSetup(nullptr, nx, vDSP_DFT_FORWARD);
    vDSP_DFT_Setup rowSetupInv = vDSP_DFT_zop_CreateSetup(nullptr, nx, vDSP_DFT_INVERSE);
    vDSP_DFT_Setup colSetupFwd = vDSP_DFT_zop_CreateSetup(nullptr, ny, vDSP_DFT_FORWARD);
    vDSP_DFT_Setup colSetupInv = vDSP_DFT_zop_CreateSetup(nullptr, ny, vDSP_DFT_INVERSE);

    vDSP_DFT_Setup rowSetup = (direction == FFT_FORWARD) ? rowSetupFwd : rowSetupInv;
    vDSP_DFT_Setup colSetup = (direction == FFT_FORWARD) ? colSetupFwd : colSetupInv;

    for (size_t b = 0; b < impl->batch; b++) {
        size_t batchOffset = b * nx * ny;

        // Step 1: FFT along rows
        for (size_t y = 0; y < ny; y++) {
            size_t rowOffset = batchOffset + y * nx;

            // Copy row to temp buffers
            memcpy(rowReal.data(), srcReal + rowOffset, nx * sizeof(float));
            memcpy(rowImag.data(), srcImag + rowOffset, nx * sizeof(float));

            // Execute row DFT
            vDSP_DFT_Execute(rowSetup, rowReal.data(), rowImag.data(),
                            tempReal.data() + y * nx, tempImag.data() + y * nx);
        }

        // Step 2: FFT along columns
        for (size_t x = 0; x < nx; x++) {
            // Extract column
            for (size_t y = 0; y < ny; y++) {
                colReal[y] = tempReal[y * nx + x];
                colImag[y] = tempImag[y * nx + x];
            }

            // Execute column DFT
            std::vector<float> colOutReal(ny), colOutImag(ny);
            vDSP_DFT_Execute(colSetup, colReal.data(), colImag.data(),
                            colOutReal.data(), colOutImag.data());

            // Store back
            for (size_t y = 0; y < ny; y++) {
                dstReal[batchOffset + y * nx + x] = colOutReal[y];
                dstImag[batchOffset + y * nx + x] = colOutImag[y];
            }
        }

        // Scale for inverse FFT
        if (direction == FFT_INVERSE) {
            float scale = 1.0f / (float)(nx * ny);
            vDSP_vsmul(dstReal + batchOffset, 1, &scale, dstReal + batchOffset, 1, nx * ny);
            vDSP_vsmul(dstImag + batchOffset, 1, &scale, dstImag + batchOffset, 1, nx * ny);
        }
    }

    // Cleanup
    vDSP_DFT_DestroySetup(rowSetupFwd);
    vDSP_DFT_DestroySetup(rowSetupInv);
    vDSP_DFT_DestroySetup(colSetupFwd);
    vDSP_DFT_DestroySetup(colSetupInv);
}

// Execute 3D DFT on CPU
static void execute3DCPUFFT(FFTPlanImpl* impl,
                            id<MTLBuffer> inReal, id<MTLBuffer> inImag,
                            id<MTLBuffer> outReal, id<MTLBuffer> outImag,
                            FFTDirection direction) {
    size_t nx = impl->nx;
    size_t ny = impl->ny;
    size_t nz = impl->nz;
    size_t nxy = nx * ny;
    size_t nxyz = nx * ny * nz;

    float* srcReal = (float*)inReal.contents;
    float* srcImag = (float*)inImag.contents;
    float* dstReal = (float*)outReal.contents;
    float* dstImag = (float*)outImag.contents;

    if (!srcReal || !srcImag || !dstReal || !dstImag) {
        fprintf(stderr, "Metal FFT ERROR: Buffer contents not accessible for CPU FFT\n");
        return;
    }

    // Allocate temporary buffers
    std::vector<float> temp1Real(nxyz), temp1Imag(nxyz);
    std::vector<float> temp2Real(nxyz), temp2Imag(nxyz);

    // Create DFT setups for each dimension
    vDSP_DFT_Setup xSetup = (direction == FFT_FORWARD) ?
        vDSP_DFT_zop_CreateSetup(nullptr, nx, vDSP_DFT_FORWARD) :
        vDSP_DFT_zop_CreateSetup(nullptr, nx, vDSP_DFT_INVERSE);
    vDSP_DFT_Setup ySetup = (direction == FFT_FORWARD) ?
        vDSP_DFT_zop_CreateSetup(nullptr, ny, vDSP_DFT_FORWARD) :
        vDSP_DFT_zop_CreateSetup(nullptr, ny, vDSP_DFT_INVERSE);
    vDSP_DFT_Setup zSetup = (direction == FFT_FORWARD) ?
        vDSP_DFT_zop_CreateSetup(nullptr, nz, vDSP_DFT_FORWARD) :
        vDSP_DFT_zop_CreateSetup(nullptr, nz, vDSP_DFT_INVERSE);

    std::vector<float> lineReal, lineImag, lineOutReal, lineOutImag;

    // Step 1: FFT along x (rows)
    lineReal.resize(nx);
    lineImag.resize(nx);
    lineOutReal.resize(nx);
    lineOutImag.resize(nx);

    for (size_t z = 0; z < nz; z++) {
        for (size_t y = 0; y < ny; y++) {
            size_t offset = z * nxy + y * nx;
            memcpy(lineReal.data(), srcReal + offset, nx * sizeof(float));
            memcpy(lineImag.data(), srcImag + offset, nx * sizeof(float));

            vDSP_DFT_Execute(xSetup, lineReal.data(), lineImag.data(),
                            lineOutReal.data(), lineOutImag.data());

            memcpy(temp1Real.data() + offset, lineOutReal.data(), nx * sizeof(float));
            memcpy(temp1Imag.data() + offset, lineOutImag.data(), nx * sizeof(float));
        }
    }

    // Step 2: FFT along y (columns)
    lineReal.resize(ny);
    lineImag.resize(ny);
    lineOutReal.resize(ny);
    lineOutImag.resize(ny);

    for (size_t z = 0; z < nz; z++) {
        for (size_t x = 0; x < nx; x++) {
            // Extract column
            for (size_t y = 0; y < ny; y++) {
                lineReal[y] = temp1Real[z * nxy + y * nx + x];
                lineImag[y] = temp1Imag[z * nxy + y * nx + x];
            }

            vDSP_DFT_Execute(ySetup, lineReal.data(), lineImag.data(),
                            lineOutReal.data(), lineOutImag.data());

            // Store back
            for (size_t y = 0; y < ny; y++) {
                temp2Real[z * nxy + y * nx + x] = lineOutReal[y];
                temp2Imag[z * nxy + y * nx + x] = lineOutImag[y];
            }
        }
    }

    // Step 3: FFT along z
    lineReal.resize(nz);
    lineImag.resize(nz);
    lineOutReal.resize(nz);
    lineOutImag.resize(nz);

    for (size_t y = 0; y < ny; y++) {
        for (size_t x = 0; x < nx; x++) {
            // Extract z-line
            for (size_t z = 0; z < nz; z++) {
                lineReal[z] = temp2Real[z * nxy + y * nx + x];
                lineImag[z] = temp2Imag[z * nxy + y * nx + x];
            }

            vDSP_DFT_Execute(zSetup, lineReal.data(), lineImag.data(),
                            lineOutReal.data(), lineOutImag.data());

            // Store to output
            for (size_t z = 0; z < nz; z++) {
                dstReal[z * nxy + y * nx + x] = lineOutReal[z];
                dstImag[z * nxy + y * nx + x] = lineOutImag[z];
            }
        }
    }

    // Scale for inverse FFT
    if (direction == FFT_INVERSE) {
        float scale = 1.0f / (float)nxyz;
        vDSP_vsmul(dstReal, 1, &scale, dstReal, 1, nxyz);
        vDSP_vsmul(dstImag, 1, &scale, dstImag, 1, nxyz);
    }

    // Cleanup
    vDSP_DFT_DestroySetup(xSetup);
    vDSP_DFT_DestroySetup(ySetup);
    vDSP_DFT_DestroySetup(zSetup);
}

// Execute 1D FFT on GPU
static void execute1DGPUFFT(FFTPlanImpl* impl,
                            id<MTLBuffer> inReal, id<MTLBuffer> inImag,
                            id<MTLBuffer> outReal, id<MTLBuffer> outImag,
                            FFTDirection direction) {
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [impl->commandQueue commandBuffer];

        uint n = (uint)impl->nx;
        uint log2n = impl->log2nx;
        int dir = (direction == FFT_FORWARD) ? 1 : -1;

        for (size_t batch_idx = 0; batch_idx < impl->batch; batch_idx++) {
            uint batchIdx = (uint)batch_idx;

            // Step 1: Bit-reversal permutation
            {
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                [encoder setComputePipelineState:impl->bitReversalPipeline];
                [encoder setBuffer:inReal offset:0 atIndex:0];
                [encoder setBuffer:inImag offset:0 atIndex:1];
                [encoder setBuffer:outReal offset:0 atIndex:2];
                [encoder setBuffer:outImag offset:0 atIndex:3];
                [encoder setBytes:&n length:sizeof(uint) atIndex:4];
                [encoder setBytes:&log2n length:sizeof(uint) atIndex:5];
                [encoder setBytes:&batchIdx length:sizeof(uint) atIndex:6];

                MTLSize gridSize = MTLSizeMake(n, 1, 1);
                MTLSize threadgroupSize = MTLSizeMake(MIN(n, 256u), 1, 1);
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];
            }

            // Step 2: Butterfly operations for each stage
            for (uint stage = 0; stage < log2n; stage++) {
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                [encoder setComputePipelineState:impl->butterflyPipeline];
                [encoder setBuffer:outReal offset:0 atIndex:0];
                [encoder setBuffer:outImag offset:0 atIndex:1];
                [encoder setBytes:&n length:sizeof(uint) atIndex:2];
                [encoder setBytes:&stage length:sizeof(uint) atIndex:3];
                [encoder setBytes:&dir length:sizeof(int) atIndex:4];
                [encoder setBytes:&batchIdx length:sizeof(uint) atIndex:5];

                uint numButterflies = n / 2;
                MTLSize gridSize = MTLSizeMake(numButterflies, 1, 1);
                MTLSize threadgroupSize = MTLSizeMake(MIN(numButterflies, 256u), 1, 1);
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];
            }

            // Step 3: Scale for inverse FFT
            if (direction == FFT_INVERSE) {
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                [encoder setComputePipelineState:impl->scalePipeline];
                [encoder setBuffer:outReal offset:0 atIndex:0];
                [encoder setBuffer:outImag offset:0 atIndex:1];
                [encoder setBytes:&n length:sizeof(uint) atIndex:2];
                float scale = 1.0f / (float)n;
                [encoder setBytes:&scale length:sizeof(float) atIndex:3];
                [encoder setBytes:&batchIdx length:sizeof(uint) atIndex:4];

                MTLSize gridSize = MTLSizeMake(n, 1, 1);
                MTLSize threadgroupSize = MTLSizeMake(MIN(n, 256u), 1, 1);
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];
            }
        }

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

// Execute 2D FFT on GPU
static void execute2DGPUFFT(FFTPlanImpl* impl,
                            id<MTLBuffer> inReal, id<MTLBuffer> inImag,
                            id<MTLBuffer> outReal, id<MTLBuffer> outImag,
                            FFTDirection direction) {
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [impl->commandQueue commandBuffer];

        uint nx = (uint)impl->nx;
        uint ny = (uint)impl->ny;
        uint log2nx = impl->log2nx;
        uint log2ny = impl->log2ny;
        int dir = (direction == FFT_FORWARD) ? 1 : -1;

        for (size_t batch_idx = 0; batch_idx < impl->batch; batch_idx++) {
            uint batchIdx = (uint)batch_idx;

            // Step 1: Bit-reversal for rows
            {
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                [encoder setComputePipelineState:impl->bitReversalRowsPipeline];
                [encoder setBuffer:inReal offset:0 atIndex:0];
                [encoder setBuffer:inImag offset:0 atIndex:1];
                [encoder setBuffer:impl->tempReal offset:0 atIndex:2];
                [encoder setBuffer:impl->tempImag offset:0 atIndex:3];
                [encoder setBytes:&nx length:sizeof(uint) atIndex:4];
                [encoder setBytes:&ny length:sizeof(uint) atIndex:5];
                [encoder setBytes:&log2nx length:sizeof(uint) atIndex:6];
                [encoder setBytes:&batchIdx length:sizeof(uint) atIndex:7];

                MTLSize gridSize = MTLSizeMake(nx, ny, 1);
                MTLSize threadgroupSize = MTLSizeMake(MIN(nx, 16u), MIN(ny, 16u), 1);
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];
            }

            // Step 2: FFT on rows
            for (uint stage = 0; stage < log2nx; stage++) {
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                [encoder setComputePipelineState:impl->butterflyRowsPipeline];
                [encoder setBuffer:impl->tempReal offset:0 atIndex:0];
                [encoder setBuffer:impl->tempImag offset:0 atIndex:1];
                [encoder setBytes:&nx length:sizeof(uint) atIndex:2];
                [encoder setBytes:&ny length:sizeof(uint) atIndex:3];
                [encoder setBytes:&stage length:sizeof(uint) atIndex:4];
                [encoder setBytes:&dir length:sizeof(int) atIndex:5];
                [encoder setBytes:&batchIdx length:sizeof(uint) atIndex:6];

                uint numButterflies = nx / 2;
                MTLSize gridSize = MTLSizeMake(numButterflies, ny, 1);
                MTLSize threadgroupSize = MTLSizeMake(MIN(numButterflies, 16u), MIN(ny, 16u), 1);
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];
            }

            // Step 3: Bit-reversal for columns
            {
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                [encoder setComputePipelineState:impl->bitReversalColsPipeline];
                [encoder setBuffer:impl->tempReal offset:0 atIndex:0];
                [encoder setBuffer:impl->tempImag offset:0 atIndex:1];
                [encoder setBuffer:outReal offset:0 atIndex:2];
                [encoder setBuffer:outImag offset:0 atIndex:3];
                [encoder setBytes:&nx length:sizeof(uint) atIndex:4];
                [encoder setBytes:&ny length:sizeof(uint) atIndex:5];
                [encoder setBytes:&log2ny length:sizeof(uint) atIndex:6];
                [encoder setBytes:&batchIdx length:sizeof(uint) atIndex:7];

                MTLSize gridSize = MTLSizeMake(nx, ny, 1);
                MTLSize threadgroupSize = MTLSizeMake(MIN(nx, 16u), MIN(ny, 16u), 1);
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];
            }

            // Step 4: FFT on columns
            for (uint stage = 0; stage < log2ny; stage++) {
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                [encoder setComputePipelineState:impl->butterflyColsPipeline];
                [encoder setBuffer:outReal offset:0 atIndex:0];
                [encoder setBuffer:outImag offset:0 atIndex:1];
                [encoder setBytes:&nx length:sizeof(uint) atIndex:2];
                [encoder setBytes:&ny length:sizeof(uint) atIndex:3];
                [encoder setBytes:&stage length:sizeof(uint) atIndex:4];
                [encoder setBytes:&dir length:sizeof(int) atIndex:5];
                [encoder setBytes:&batchIdx length:sizeof(uint) atIndex:6];

                uint numButterflies = ny / 2;
                MTLSize gridSize = MTLSizeMake(nx, numButterflies, 1);
                MTLSize threadgroupSize = MTLSizeMake(MIN(nx, 16u), MIN(numButterflies, 16u), 1);
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];
            }

            // Step 5: Scale for inverse FFT
            if (direction == FFT_INVERSE) {
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                [encoder setComputePipelineState:impl->scalePipeline];
                [encoder setBuffer:outReal offset:0 atIndex:0];
                [encoder setBuffer:outImag offset:0 atIndex:1];
                uint totalSize = nx * ny;
                [encoder setBytes:&totalSize length:sizeof(uint) atIndex:2];
                float scale = 1.0f / (float)totalSize;
                [encoder setBytes:&scale length:sizeof(float) atIndex:3];
                [encoder setBytes:&batchIdx length:sizeof(uint) atIndex:4];

                MTLSize gridSize = MTLSizeMake(totalSize, 1, 1);
                MTLSize threadgroupSize = MTLSizeMake(MIN(totalSize, 256u), 1, 1);
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];
            }
        }

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

// Execute 3D FFT on GPU
static void execute3DGPUFFT(FFTPlanImpl* impl,
                            id<MTLBuffer> inReal, id<MTLBuffer> inImag,
                            id<MTLBuffer> outReal, id<MTLBuffer> outImag,
                            FFTDirection direction) {
    @autoreleasepool {
        // 3D FFT = 2D FFT on xy-planes + 1D FFT on z-dimension
        // First do 2D FFT on each xy-plane

        id<MTLCommandBuffer> commandBuffer = [impl->commandQueue commandBuffer];

        uint nx = (uint)impl->nx;
        uint ny = (uint)impl->ny;
        uint nz = (uint)impl->nz;
        uint log2nx = impl->log2nx;
        uint log2ny = impl->log2ny;
        uint log2nz = impl->log2nz;
        int dir = (direction == FFT_FORWARD) ? 1 : -1;
        uint batchIdx = 0;

        // Process each z-slice as 2D FFT
        for (uint z = 0; z < nz; z++) {
            // Bit-reversal for rows of this slice
            {
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                [encoder setComputePipelineState:impl->bitReversalRowsPipeline];
                size_t offset = z * ny * nx * sizeof(float);
                [encoder setBuffer:inReal offset:offset atIndex:0];
                [encoder setBuffer:inImag offset:offset atIndex:1];
                [encoder setBuffer:impl->tempReal offset:offset atIndex:2];
                [encoder setBuffer:impl->tempImag offset:offset atIndex:3];
                [encoder setBytes:&nx length:sizeof(uint) atIndex:4];
                [encoder setBytes:&ny length:sizeof(uint) atIndex:5];
                [encoder setBytes:&log2nx length:sizeof(uint) atIndex:6];
                [encoder setBytes:&batchIdx length:sizeof(uint) atIndex:7];

                MTLSize gridSize = MTLSizeMake(nx, ny, 1);
                MTLSize threadgroupSize = MTLSizeMake(MIN(nx, 16u), MIN(ny, 16u), 1);
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];
            }

            // Row FFTs
            for (uint stage = 0; stage < log2nx; stage++) {
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                [encoder setComputePipelineState:impl->butterflyRowsPipeline];
                size_t offset = z * ny * nx * sizeof(float);
                [encoder setBuffer:impl->tempReal offset:offset atIndex:0];
                [encoder setBuffer:impl->tempImag offset:offset atIndex:1];
                [encoder setBytes:&nx length:sizeof(uint) atIndex:2];
                [encoder setBytes:&ny length:sizeof(uint) atIndex:3];
                [encoder setBytes:&stage length:sizeof(uint) atIndex:4];
                [encoder setBytes:&dir length:sizeof(int) atIndex:5];
                [encoder setBytes:&batchIdx length:sizeof(uint) atIndex:6];

                uint numButterflies = nx / 2;
                MTLSize gridSize = MTLSizeMake(numButterflies, ny, 1);
                MTLSize threadgroupSize = MTLSizeMake(MIN(numButterflies, 16u), MIN(ny, 16u), 1);
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];
            }

            // Bit-reversal for columns
            {
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                [encoder setComputePipelineState:impl->bitReversalColsPipeline];
                size_t offset = z * ny * nx * sizeof(float);
                [encoder setBuffer:impl->tempReal offset:offset atIndex:0];
                [encoder setBuffer:impl->tempImag offset:offset atIndex:1];
                [encoder setBuffer:outReal offset:offset atIndex:2];
                [encoder setBuffer:outImag offset:offset atIndex:3];
                [encoder setBytes:&nx length:sizeof(uint) atIndex:4];
                [encoder setBytes:&ny length:sizeof(uint) atIndex:5];
                [encoder setBytes:&log2ny length:sizeof(uint) atIndex:6];
                [encoder setBytes:&batchIdx length:sizeof(uint) atIndex:7];

                MTLSize gridSize = MTLSizeMake(nx, ny, 1);
                MTLSize threadgroupSize = MTLSizeMake(MIN(nx, 16u), MIN(ny, 16u), 1);
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];
            }

            // Column FFTs
            for (uint stage = 0; stage < log2ny; stage++) {
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                [encoder setComputePipelineState:impl->butterflyColsPipeline];
                size_t offset = z * ny * nx * sizeof(float);
                [encoder setBuffer:outReal offset:offset atIndex:0];
                [encoder setBuffer:outImag offset:offset atIndex:1];
                [encoder setBytes:&nx length:sizeof(uint) atIndex:2];
                [encoder setBytes:&ny length:sizeof(uint) atIndex:3];
                [encoder setBytes:&stage length:sizeof(uint) atIndex:4];
                [encoder setBytes:&dir length:sizeof(int) atIndex:5];
                [encoder setBytes:&batchIdx length:sizeof(uint) atIndex:6];

                uint numButterflies = ny / 2;
                MTLSize gridSize = MTLSizeMake(nx, numButterflies, 1);
                MTLSize threadgroupSize = MTLSizeMake(MIN(nx, 16u), MIN(numButterflies, 16u), 1);
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];
            }
        }

        // Now do FFT along z-dimension
        // First bit-reverse along z
        // Since we don't have a dedicated kernel, we'll implement inline

        // Bit-reversal in z-dimension (copy to temp)
        for (uint z = 0; z < nz; z++) {
            uint rev_z = 0;
            uint temp_z = z;
            for (uint i = 0; i < log2nz; i++) {
                rev_z = (rev_z << 1) | (temp_z & 1);
                temp_z >>= 1;
            }
            if (z < rev_z) {
                // Swap slices z and rev_z
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                [encoder setComputePipelineState:impl->copyPipeline];

                size_t sliceSize = nx * ny;
                size_t offset_z = z * sliceSize * sizeof(float);
                size_t offset_rev_z = rev_z * sliceSize * sizeof(float);

                // Copy z to temp
                [encoder setBuffer:outReal offset:offset_z atIndex:0];
                [encoder setBuffer:outImag offset:offset_z atIndex:1];
                [encoder setBuffer:impl->tempReal offset:0 atIndex:2];
                [encoder setBuffer:impl->tempImag offset:0 atIndex:3];
                uint sliceSizeU = (uint)sliceSize;
                [encoder setBytes:&sliceSizeU length:sizeof(uint) atIndex:4];

                MTLSize gridSize = MTLSizeMake(sliceSize, 1, 1);
                MTLSize threadgroupSize = MTLSizeMake(MIN(sliceSize, 256u), 1, 1);
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];

                // Copy rev_z to z
                encoder = [commandBuffer computeCommandEncoder];
                [encoder setComputePipelineState:impl->copyPipeline];
                [encoder setBuffer:outReal offset:offset_rev_z atIndex:0];
                [encoder setBuffer:outImag offset:offset_rev_z atIndex:1];
                [encoder setBuffer:outReal offset:offset_z atIndex:2];
                [encoder setBuffer:outImag offset:offset_z atIndex:3];
                [encoder setBytes:&sliceSizeU length:sizeof(uint) atIndex:4];
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];

                // Copy temp to rev_z
                encoder = [commandBuffer computeCommandEncoder];
                [encoder setComputePipelineState:impl->copyPipeline];
                [encoder setBuffer:impl->tempReal offset:0 atIndex:0];
                [encoder setBuffer:impl->tempImag offset:0 atIndex:1];
                [encoder setBuffer:outReal offset:offset_rev_z atIndex:2];
                [encoder setBuffer:outImag offset:offset_rev_z atIndex:3];
                [encoder setBytes:&sliceSizeU length:sizeof(uint) atIndex:4];
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];
            }
        }

        // Z-dimension butterflies
        for (uint stage = 0; stage < log2nz; stage++) {
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:impl->butterflyZPipeline];
            [encoder setBuffer:outReal offset:0 atIndex:0];
            [encoder setBuffer:outImag offset:0 atIndex:1];
            [encoder setBytes:&nx length:sizeof(uint) atIndex:2];
            [encoder setBytes:&ny length:sizeof(uint) atIndex:3];
            [encoder setBytes:&nz length:sizeof(uint) atIndex:4];
            [encoder setBytes:&stage length:sizeof(uint) atIndex:5];
            [encoder setBytes:&dir length:sizeof(int) atIndex:6];

            uint numButterflies = nz / 2;
            MTLSize gridSize = MTLSizeMake(nx, ny, numButterflies);
            MTLSize threadgroupSize = MTLSizeMake(MIN(nx, 8u), MIN(ny, 8u), MIN(numButterflies, 4u));
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];
        }

        // Scale for inverse FFT
        if (direction == FFT_INVERSE) {
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:impl->scalePipeline];
            [encoder setBuffer:outReal offset:0 atIndex:0];
            [encoder setBuffer:outImag offset:0 atIndex:1];
            uint totalSize = nx * ny * nz;
            [encoder setBytes:&totalSize length:sizeof(uint) atIndex:2];
            float scale = 1.0f / (float)totalSize;
            [encoder setBytes:&scale length:sizeof(float) atIndex:3];
            [encoder setBytes:&batchIdx length:sizeof(uint) atIndex:4];

            MTLSize gridSize = MTLSizeMake(totalSize, 1, 1);
            MTLSize threadgroupSize = MTLSizeMake(MIN(totalSize, 256u), 1, 1);
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];
        }

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
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

    // Cast buffers
    id<MTLBuffer> inReal = (__bridge id<MTLBuffer>)input_real;
    id<MTLBuffer> inImag = (__bridge id<MTLBuffer>)input_imag;
    id<MTLBuffer> outReal = (__bridge id<MTLBuffer>)output_real;
    id<MTLBuffer> outImag = (__bridge id<MTLBuffer>)output_imag;

    if (impl->useCPUFallback) {
        // Execute FFT using vDSP CPU implementation
        // This is used for non-power-of-2 sizes where GPU radix-2 FFT doesn't work
        if (impl->dimensions == 1) {
            execute1DCPUFFT(impl, inReal, inImag, outReal, outImag, direction);
        } else if (impl->dimensions == 2) {
            execute2DCPUFFT(impl, inReal, inImag, outReal, outImag, direction);
        } else if (impl->dimensions == 3) {
            execute3DCPUFFT(impl, inReal, inImag, outReal, outImag, direction);
        }
        return;
    }

    // GPU FFT path for power-of-2 sizes
    if (impl->dimensions == 1) {
        execute1DGPUFFT(impl, inReal, inImag, outReal, outImag, direction);
    } else if (impl->dimensions == 2) {
        execute2DGPUFFT(impl, inReal, inImag, outReal, outImag, direction);
    } else if (impl->dimensions == 3) {
        execute3DGPUFFT(impl, inReal, inImag, outReal, outImag, direction);
    }
}

void executeFFTInPlace(FFTPlan plan,
                       void* data_real,
                       void* data_imag,
                       FFTDirection direction,
                       void* commandQueue) {
    executeFFT(plan, data_real, data_imag, data_real, data_imag, direction, commandQueue);
}

} // namespace MetalFFT
