#import "src/acc/metal/metal_kernels.h"
#import "src/acc/metal/metal_device.h"
#import "src/acc/metal/metal_kernel_utils.h"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#ifdef _METAL_ENABLED

namespace MetalKernels {

// Global Metal resources
static id<MTLDevice> g_device = nil;
static id<MTLLibrary> g_library = nil;
static id<MTLCommandQueue> g_queue = nil;
static NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* g_pipelines = nil;

// Initialize Metal infrastructure
static void initMetalIfNeeded() {
    if (g_device == nil) {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            CRITICAL(ERRGPUKERN);
        }

        g_queue = [g_device newCommandQueue];
        g_library = [g_device newDefaultLibrary];
        g_pipelines = [[NSMutableDictionary alloc] init];

        fprintf(stderr, "Metal INFO: Initialized Metal device: %s\n",
                [[g_device name] UTF8String]);
    }
}

// Get or create pipeline state for kernel
static id<MTLComputePipelineState> getPipeline(const char* kernelName) {
    initMetalIfNeeded();

    NSString* name = [NSString stringWithUTF8String:kernelName];
    id<MTLComputePipelineState> pipeline = g_pipelines[name];

    if (!pipeline) {
        id<MTLFunction> function = [g_library newFunctionWithName:name];
        if (!function) {
            fprintf(stderr, "Metal ERROR: Kernel '%s' not found\n", kernelName);
            CRITICAL(ERRGPUKERN);
        }

        NSError* error = nil;
        pipeline = [g_device newComputePipelineStateWithFunction:function error:&error];
        if (error) {
            fprintf(stderr, "Metal ERROR: Failed to create pipeline for '%s': %s\n",
                    kernelName, [[error localizedDescription] UTF8String]);
            CRITICAL(ERRGPUKERN);
        }

        g_pipelines[name] = pipeline;
    }

    return pipeline;
}

// Helper to create ProjectorParams buffer
static id<MTLBuffer> createProjectorParamsBuffer(AccProjectorKernel &projector) {
    struct ProjectorParams {
        int mdlX, mdlXY, mdlZ;
        int imgX, imgY, imgZ;
        int mdlInitY, mdlInitZ;
        int maxR, maxR2, maxR2_padded;
        float padding_factor;
    };

    ProjectorParams params;
    params.mdlX = projector.mdlX;
    params.mdlXY = projector.mdlXY;
    params.mdlZ = projector.mdlZ;
    params.imgX = projector.imgX;
    params.imgY = projector.imgY;
    params.imgZ = projector.imgZ;
    params.mdlInitY = projector.mdlInitY;
    params.mdlInitZ = projector.mdlInitZ;
    params.maxR = projector.maxR;
    params.maxR2 = projector.maxR2;
    params.maxR2_padded = projector.maxR2_padded;
    params.padding_factor = projector.padding_factor;

    return [g_device newBufferWithBytes:&params
                                 length:sizeof(ProjectorParams)
                                options:MTLResourceStorageModeShared];
}

// ============================================================================
// diff2_coarse Implementation
// ============================================================================

template<bool REF3D, bool DATA3D>
void diff2_coarse(
    unsigned long grid_size,
    int block_size,
    XFLOAT *g_eulers,
    XFLOAT *trans_x,
    XFLOAT *trans_y,
    XFLOAT *trans_z,
    XFLOAT *g_real,
    XFLOAT *g_imag,
    XFLOAT *mdlReal,
    XFLOAT *mdlImag,
    AccProjectorKernel &projector,
    XFLOAT *g_corr,
    XFLOAT *g_diff2s,
    unsigned long translation_num,
    unsigned long image_size,
    int eulers_per_block,
    int prefetch_fraction,
    deviceStream_t stream)
{
    @autoreleasepool {
        initMetalIfNeeded();

        // Select kernel based on dimensionality
        const char* kernelName = DATA3D ? "metal_kernel_diff2_coarse_3D"
                                         : "metal_kernel_diff2_coarse_2D";
        id<MTLComputePipelineState> pipeline = getPipeline(kernelName);

        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];

        // Set buffers
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_eulers offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)trans_x offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)trans_y offset:0 atIndex:2];

        if (DATA3D) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)trans_z offset:0 atIndex:3];
            [encoder setBuffer:(__bridge id<MTLBuffer>)g_real offset:0 atIndex:4];
            [encoder setBuffer:(__bridge id<MTLBuffer>)g_imag offset:0 atIndex:5];
            [encoder setBuffer:(__bridge id<MTLBuffer>)mdlReal offset:0 atIndex:6];
            [encoder setBuffer:(__bridge id<MTLBuffer>)mdlImag offset:0 atIndex:7];

            id<MTLBuffer> projParams = createProjectorParamsBuffer(projector);
            [encoder setBuffer:projParams offset:0 atIndex:8];

            [encoder setBuffer:(__bridge id<MTLBuffer>)g_corr offset:0 atIndex:9];
            [encoder setBuffer:(__bridge id<MTLBuffer>)g_diff2s offset:0 atIndex:10];

            int trans_num = (int)translation_num;
            int img_size = (int)image_size;
            [encoder setBytes:&trans_num length:sizeof(int) atIndex:11];
            [encoder setBytes:&img_size length:sizeof(int) atIndex:12];
            [encoder setBytes:&eulers_per_block length:sizeof(int) atIndex:13];
            [encoder setBytes:&block_size length:sizeof(int) atIndex:14];
            [encoder setBytes:&prefetch_fraction length:sizeof(int) atIndex:15];
        } else {
            [encoder setBuffer:(__bridge id<MTLBuffer>)g_real offset:0 atIndex:3];
            [encoder setBuffer:(__bridge id<MTLBuffer>)g_imag offset:0 atIndex:4];
            [encoder setBuffer:(__bridge id<MTLBuffer>)mdlReal offset:0 atIndex:5];
            [encoder setBuffer:(__bridge id<MTLBuffer>)mdlImag offset:0 atIndex:6];

            id<MTLBuffer> projParams = createProjectorParamsBuffer(projector);
            [encoder setBuffer:projParams offset:0 atIndex:7];

            [encoder setBuffer:(__bridge id<MTLBuffer>)g_corr offset:0 atIndex:8];
            [encoder setBuffer:(__bridge id<MTLBuffer>)g_diff2s offset:0 atIndex:9];

            int trans_num = (int)translation_num;
            int img_size = (int)image_size;
            [encoder setBytes:&trans_num length:sizeof(int) atIndex:10];
            [encoder setBytes:&img_size length:sizeof(int) atIndex:11];
            [encoder setBytes:&eulers_per_block length:sizeof(int) atIndex:12];
            [encoder setBytes:&block_size length:sizeof(int) atIndex:13];
            [encoder setBytes:&prefetch_fraction length:sizeof(int) atIndex:14];
        }

        // Calculate shared memory size
        size_t sharedMemSize = (eulers_per_block * 9 +                    // s_eulers
                                block_size/prefetch_fraction * eulers_per_block * 2 + // s_ref_real/imag
                                block_size * 3) * sizeof(XFLOAT);         // s_real/imag/corr

        [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

        // Dispatch
        MTLSize gridSize = MTLSizeMake(grid_size * block_size, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(block_size, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        [encoder endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
    }
}

// Explicit template instantiations
template void diff2_coarse<false, false>(unsigned long, int, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, AccProjectorKernel&, XFLOAT*, XFLOAT*, unsigned long, unsigned long, int, int, deviceStream_t);
template void diff2_coarse<true, false>(unsigned long, int, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, AccProjectorKernel&, XFLOAT*, XFLOAT*, unsigned long, unsigned long, int, int, deviceStream_t);
template void diff2_coarse<true, true>(unsigned long, int, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, AccProjectorKernel&, XFLOAT*, XFLOAT*, unsigned long, unsigned long, int, int, deviceStream_t);

// ============================================================================
// diff2_fine Implementation
// ============================================================================

template<bool REF3D, bool DATA3D>
void diff2_fine(
    unsigned long grid_size,
    int block_size,
    XFLOAT *g_eulers,
    XFLOAT *g_imgs_real,
    XFLOAT *g_imgs_imag,
    XFLOAT *trans_x,
    XFLOAT *trans_y,
    XFLOAT *trans_z,
    XFLOAT *mdlReal,
    XFLOAT *mdlImag,
    AccProjectorKernel &projector,
    XFLOAT *g_corr_img,
    XFLOAT *g_diff2s,
    unsigned long image_size,
    XFLOAT sum_init,
    unsigned long orientation_num,
    unsigned long translation_num,
    unsigned long todo_blocks,
    unsigned long *d_rot_idx,
    unsigned long *d_trans_idx,
    unsigned long *d_job_idx,
    unsigned long *d_job_num,
    int chunk_sz,
    deviceStream_t stream)
{
    @autoreleasepool {
        initMetalIfNeeded();

        const char* kernelName = DATA3D ? "metal_kernel_diff2_fine_3D"
                                         : "metal_kernel_diff2_fine_2D";
        id<MTLComputePipelineState> pipeline = getPipeline(kernelName);

        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];

        // Set buffers based on dimensionality
        int bufIdx = 0;
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_eulers offset:0 atIndex:bufIdx++];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_imgs_real offset:0 atIndex:bufIdx++];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_imgs_imag offset:0 atIndex:bufIdx++];
        [encoder setBuffer:(__bridge id<MTLBuffer>)trans_x offset:0 atIndex:bufIdx++];
        [encoder setBuffer:(__bridge id<MTLBuffer>)trans_y offset:0 atIndex:bufIdx++];

        if (DATA3D) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)trans_z offset:0 atIndex:bufIdx++];
        }

        [encoder setBuffer:(__bridge id<MTLBuffer>)mdlReal offset:0 atIndex:bufIdx++];
        [encoder setBuffer:(__bridge id<MTLBuffer>)mdlImag offset:0 atIndex:bufIdx++];

        id<MTLBuffer> projParams = createProjectorParamsBuffer(projector);
        [encoder setBuffer:projParams offset:0 atIndex:bufIdx++];

        [encoder setBuffer:(__bridge id<MTLBuffer>)g_corr_img offset:0 atIndex:bufIdx++];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_diff2s offset:0 atIndex:bufIdx++];

        uint img_sz = (uint)image_size;
        [encoder setBytes:&img_sz length:sizeof(uint) atIndex:bufIdx++];
        [encoder setBytes:&sum_init length:sizeof(XFLOAT) atIndex:bufIdx++];

        uint orient_num = (uint)orientation_num;
        uint trans_num = (uint)translation_num;
        [encoder setBytes:&orient_num length:sizeof(uint) atIndex:bufIdx++];
        [encoder setBytes:&trans_num length:sizeof(uint) atIndex:bufIdx++];

        [encoder setBuffer:(__bridge id<MTLBuffer>)d_rot_idx offset:0 atIndex:bufIdx++];
        [encoder setBuffer:(__bridge id<MTLBuffer>)d_trans_idx offset:0 atIndex:bufIdx++];
        [encoder setBuffer:(__bridge id<MTLBuffer>)d_job_idx offset:0 atIndex:bufIdx++];
        [encoder setBuffer:(__bridge id<MTLBuffer>)d_job_num offset:0 atIndex:bufIdx++];

        [encoder setBytes:&block_size length:sizeof(int) atIndex:bufIdx++];
        [encoder setBytes:&chunk_sz length:sizeof(int) atIndex:bufIdx++];

        // Shared memory for accumulation
        size_t sharedMemSize = chunk_sz * block_size * sizeof(XFLOAT);
        [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];

        MTLSize gridSize = MTLSizeMake(todo_blocks * block_size, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(block_size, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        [encoder endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
    }
}

// Explicit template instantiations
template void diff2_fine<false, false>(unsigned long, int, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, AccProjectorKernel&, XFLOAT*, XFLOAT*, unsigned long, XFLOAT, unsigned long, unsigned long, unsigned long, unsigned long*, unsigned long*, unsigned long*, unsigned long*, int, deviceStream_t);
template void diff2_fine<true, false>(unsigned long, int, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, AccProjectorKernel&, XFLOAT*, XFLOAT*, unsigned long, XFLOAT, unsigned long, unsigned long, unsigned long, unsigned long*, unsigned long*, unsigned long*, unsigned long*, int, deviceStream_t);
template void diff2_fine<true, true>(unsigned long, int, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, XFLOAT*, AccProjectorKernel&, XFLOAT*, XFLOAT*, unsigned long, XFLOAT, unsigned long, unsigned long, unsigned long, unsigned long*, unsigned long*, unsigned long*, unsigned long*, int, deviceStream_t);

// ============================================================================
// Backprojection Implementations
// ============================================================================

void backproject2D(
    XFLOAT *g_img_real,
    XFLOAT *g_img_imag,
    XFLOAT *trans_x,
    XFLOAT *trans_y,
    XFLOAT *g_weights,
    XFLOAT *g_eulers,
    XFLOAT *g_mdl_real,
    XFLOAT *g_mdl_imag,
    XFLOAT *g_mdl_weight,
    AccProjectorKernel &projector,
    unsigned long image_size,
    unsigned long translation_idx,
    XFLOAT weight,
    deviceStream_t stream)
{
    @autoreleasepool {
        initMetalIfNeeded();

        id<MTLComputePipelineState> pipeline = getPipeline("metal_kernel_backproject2D");
        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];

        [encoder setBuffer:(__bridge id<MTLBuffer>)g_img_real offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_img_imag offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)trans_x offset:0 atIndex:2];
        [encoder setBuffer:(__bridge id<MTLBuffer>)trans_y offset:0 atIndex:3];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_weights offset:0 atIndex:4];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_eulers offset:0 atIndex:5];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_mdl_real offset:0 atIndex:6];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_mdl_imag offset:0 atIndex:7];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_mdl_weight offset:0 atIndex:8];

        id<MTLBuffer> projParams = createProjectorParamsBuffer(projector);
        [encoder setBuffer:projParams offset:0 atIndex:9];

        int img_size = (int)image_size;
        int trans_idx = (int)translation_idx;
        float wt = (float)weight;
        [encoder setBytes:&img_size length:sizeof(int) atIndex:10];
        [encoder setBytes:&trans_idx length:sizeof(int) atIndex:11];
        [encoder setBytes:&wt length:sizeof(float) atIndex:12];

        MTLSize gridSize = MTLSizeMake(image_size, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(MIN(image_size, 256ul), 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        [encoder endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
    }
}

void backproject3D(
    XFLOAT *g_img_real,
    XFLOAT *g_img_imag,
    XFLOAT *trans_x,
    XFLOAT *trans_y,
    XFLOAT *trans_z,
    XFLOAT *g_weights,
    XFLOAT *g_eulers,
    XFLOAT *g_mdl_real,
    XFLOAT *g_mdl_imag,
    XFLOAT *g_mdl_weight,
    AccProjectorKernel &projector,
    unsigned long image_size,
    unsigned long translation_idx,
    XFLOAT weight,
    deviceStream_t stream)
{
    @autoreleasepool {
        initMetalIfNeeded();

        id<MTLComputePipelineState> pipeline = getPipeline("metal_kernel_backproject3D");
        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];

        [encoder setBuffer:(__bridge id<MTLBuffer>)g_img_real offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_img_imag offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)trans_x offset:0 atIndex:2];
        [encoder setBuffer:(__bridge id<MTLBuffer>)trans_y offset:0 atIndex:3];
        [encoder setBuffer:(__bridge id<MTLBuffer>)trans_z offset:0 atIndex:4];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_weights offset:0 atIndex:5];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_eulers offset:0 atIndex:6];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_mdl_real offset:0 atIndex:7];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_mdl_imag offset:0 atIndex:8];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_mdl_weight offset:0 atIndex:9];

        id<MTLBuffer> projParams = createProjectorParamsBuffer(projector);
        [encoder setBuffer:projParams offset:0 atIndex:10];

        int img_size = (int)image_size;
        int trans_idx = (int)translation_idx;
        float wt = (float)weight;
        [encoder setBytes:&img_size length:sizeof(int) atIndex:11];
        [encoder setBytes:&trans_idx length:sizeof(int) atIndex:12];
        [encoder setBytes:&wt length:sizeof(float) atIndex:13];

        MTLSize gridSize = MTLSizeMake(image_size, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(MIN(image_size, 256ul), 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        [encoder endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
    }
}

void backproject2D_SGD(
    XFLOAT *g_img_real,
    XFLOAT *g_img_imag,
    XFLOAT *trans_x,
    XFLOAT *trans_y,
    XFLOAT *g_weights,
    XFLOAT *g_eulers,
    XFLOAT *g_mdl_real,
    XFLOAT *g_mdl_imag,
    XFLOAT *g_mdl_weight,
    AccProjectorKernel &projector,
    unsigned long image_size,
    unsigned long translation_idx,
    XFLOAT weight,
    XFLOAT significant_weight,
    XFLOAT sum_ref_weight,
    deviceStream_t stream)
{
    // For now, use standard backprojection (SGD variant needs additional kernel)
    backproject2D(g_img_real, g_img_imag, trans_x, trans_y, g_weights, g_eulers,
                  g_mdl_real, g_mdl_imag, g_mdl_weight, projector, image_size,
                  translation_idx, weight, stream);
}

void backproject3D_SGD(
    XFLOAT *g_img_real,
    XFLOAT *g_img_imag,
    XFLOAT *trans_x,
    XFLOAT *trans_y,
    XFLOAT *trans_z,
    XFLOAT *g_weights,
    XFLOAT *g_eulers,
    XFLOAT *g_mdl_real,
    XFLOAT *g_mdl_imag,
    XFLOAT *g_mdl_weight,
    AccProjectorKernel &projector,
    unsigned long image_size,
    unsigned long translation_idx,
    XFLOAT weight,
    XFLOAT significant_weight,
    XFLOAT sum_ref_weight,
    deviceStream_t stream)
{
    // For now, use standard backprojection (SGD variant needs additional kernel)
    backproject3D(g_img_real, g_img_imag, trans_x, trans_y, trans_z, g_weights,
                  g_eulers, g_mdl_real, g_mdl_imag, g_mdl_weight, projector,
                  image_size, translation_idx, weight, stream);
}

// ============================================================================
// Weighted Averaging Implementation
// ============================================================================

void wavg(
    XFLOAT *g_img_real,
    XFLOAT *g_img_imag,
    XFLOAT *g_img_weight,
    XFLOAT *g_out_real,
    XFLOAT *g_out_imag,
    XFLOAT *g_out_weight,
    XFLOAT particle_weight,
    unsigned long size,
    deviceStream_t stream)
{
    @autoreleasepool {
        initMetalIfNeeded();

        id<MTLComputePipelineState> pipeline = getPipeline("metal_kernel_wavg");
        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];

        [encoder setBuffer:(__bridge id<MTLBuffer>)g_img_real offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_img_imag offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_img_weight offset:0 atIndex:2];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_out_real offset:0 atIndex:3];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_out_imag offset:0 atIndex:4];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_out_weight offset:0 atIndex:5];

        float pw = (float)particle_weight;
        int sz = (int)size;
        [encoder setBytes:&pw length:sizeof(float) atIndex:6];
        [encoder setBytes:&sz length:sizeof(int) atIndex:7];

        MTLSize gridSize = MTLSizeMake(size, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(MIN(size, 256ul), 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        [encoder endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
    }
}

// ============================================================================
// Utility Kernel Implementations
// ============================================================================

void exponentiate(
    XFLOAT *g_array,
    XFLOAT add,
    unsigned long size,
    deviceStream_t stream)
{
    @autoreleasepool {
        initMetalIfNeeded();

        id<MTLComputePipelineState> pipeline = getPipeline("metal_kernel_exponentiate");
        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_array offset:0 atIndex:0];

        float addVal = (float)add;
        uint sz = (uint)size;
        [encoder setBytes:&addVal length:sizeof(float) atIndex:1];
        [encoder setBytes:&sz length:sizeof(uint) atIndex:2];

        MTLSize gridSize = MTLSizeMake(size, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(MIN(size, 256ul), 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        [encoder endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
    }
}

void softMaskOutsideMap(
    XFLOAT *vol,
    long int xdim,
    long int ydim,
    long int zdim,
    long int xinit,
    long int yinit,
    long int zinit,
    bool do_Mnoise,
    XFLOAT *Mnoise,
    XFLOAT radius,
    XFLOAT radius_p,
    XFLOAT cosine_width,
    deviceStream_t stream)
{
    @autoreleasepool {
        initMetalIfNeeded();

        id<MTLComputePipelineState> pipeline = getPipeline("metal_kernel_softMaskOutsideMap");
        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)vol offset:0 atIndex:0];

        uint xd = (uint)xdim, yd = (uint)ydim, zd = (uint)zdim;
        int xi = (int)xinit, yi = (int)yinit, zi = (int)zinit;
        float rad = (float)radius, cos_w = (float)cosine_width;

        [encoder setBytes:&xd length:sizeof(uint) atIndex:1];
        [encoder setBytes:&yd length:sizeof(uint) atIndex:2];
        [encoder setBytes:&zd length:sizeof(uint) atIndex:3];
        [encoder setBytes:&xi length:sizeof(int) atIndex:4];
        [encoder setBytes:&yi length:sizeof(int) atIndex:5];
        [encoder setBytes:&zi length:sizeof(int) atIndex:6];
        [encoder setBytes:&rad length:sizeof(float) atIndex:7];
        [encoder setBytes:&cos_w length:sizeof(float) atIndex:8];

        unsigned long totalSize = xdim * ydim * zdim;
        MTLSize gridSize = MTLSizeMake(totalSize, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(MIN(totalSize, 256ul), 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        [encoder endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
    }
}

void multiply(
    XFLOAT *A,
    XFLOAT *B,
    XFLOAT *OUT,
    unsigned long size,
    deviceStream_t stream)
{
    @autoreleasepool {
        initMetalIfNeeded();

        id<MTLComputePipelineState> pipeline = getPipeline("metal_kernel_multiply");
        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)A offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)B offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)OUT offset:0 atIndex:2];

        uint sz = (uint)size;
        [encoder setBytes:&sz length:sizeof(uint) atIndex:3];

        MTLSize gridSize = MTLSizeMake(size, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(MIN(size, 256ul), 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        [encoder endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
    }
}

void multiplyCTFs(
    XFLOAT *g_Fref_real,
    XFLOAT *g_Fref_imag,
    XFLOAT *g_ctf,
    bool do_scale_correction,
    XFLOAT *g_scale_correction,
    unsigned long image_size,
    deviceStream_t stream)
{
    @autoreleasepool {
        initMetalIfNeeded();

        id<MTLComputePipelineState> pipeline = getPipeline("metal_kernel_multiplyCTFs");
        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_Fref_real offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_Fref_imag offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_ctf offset:0 atIndex:2];

        uint doScale = do_scale_correction ? 1 : 0;
        [encoder setBytes:&doScale length:sizeof(uint) atIndex:3];

        if (do_scale_correction && g_scale_correction) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)g_scale_correction offset:0 atIndex:4];
        } else {
            // Create dummy buffer
            id<MTLBuffer> dummy = [g_device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
            [encoder setBuffer:dummy offset:0 atIndex:4];
        }

        uint sz = (uint)image_size;
        [encoder setBytes:&sz length:sizeof(uint) atIndex:5];

        MTLSize gridSize = MTLSizeMake(image_size, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(MIN(image_size, 256ul), 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        [encoder endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
    }
}

void applyWeights(
    XFLOAT *g_diff2s,
    XFLOAT *g_weights,
    XFLOAT weight,
    unsigned long size,
    deviceStream_t stream)
{
    @autoreleasepool {
        initMetalIfNeeded();

        id<MTLComputePipelineState> pipeline = getPipeline("metal_kernel_applyWeights");
        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_diff2s offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_weights offset:0 atIndex:1];

        float wt = (float)weight;
        uint sz = (uint)size;
        [encoder setBytes:&wt length:sizeof(float) atIndex:2];
        [encoder setBytes:&sz length:sizeof(uint) atIndex:3];

        MTLSize gridSize = MTLSizeMake(size, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(MIN(size, 256ul), 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        [encoder endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
    }
}

// ============================================================================
// Random Number Generation (Philox)
// ============================================================================

// Static seed offset counter for generating different sequences
static uint g_rng_seed_offset = 0;

void initRNG(
    void *rng_states,
    unsigned long long seed,
    unsigned long size,
    deviceStream_t stream)
{
    @autoreleasepool {
        initMetalIfNeeded();

        id<MTLComputePipelineState> pipeline = getPipeline("metal_kernel_initRNG");
        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)rng_states offset:0 atIndex:0];

        uint num_states = (uint)size;
        uint seed_lo = (uint)(seed & 0xFFFFFFFF);
        uint seed_hi = (uint)(seed >> 32);

        [encoder setBytes:&num_states length:sizeof(uint) atIndex:1];
        [encoder setBytes:&seed_lo length:sizeof(uint) atIndex:2];
        [encoder setBytes:&seed_hi length:sizeof(uint) atIndex:3];

        MTLSize gridSize = MTLSizeMake(size, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(MIN(size, 256ul), 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        [encoder endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        // Reset seed offset for new RNG state
        g_rng_seed_offset = 0;

        fprintf(stderr, "Metal INFO: Initialized Philox RNG with %lu states, seed=%llu\n", size, seed);
    }
}

void generateNormalDistribution2D(
    void *rng_states,
    XFLOAT *g_out_real,
    XFLOAT *g_out_imag,
    XFLOAT *g_spectra,
    unsigned long xdim,
    unsigned long ydim,
    deviceStream_t stream)
{
    @autoreleasepool {
        initMetalIfNeeded();

        id<MTLComputePipelineState> pipeline = getPipeline("metal_kernel_RNDnormalDistribution2D");
        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)rng_states offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_out_real offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_out_imag offset:0 atIndex:2];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_spectra offset:0 atIndex:3];

        uint xd = (uint)xdim;
        uint yd = (uint)ydim;
        uint seed_offset = g_rng_seed_offset++;

        [encoder setBytes:&xd length:sizeof(uint) atIndex:4];
        [encoder setBytes:&yd length:sizeof(uint) atIndex:5];
        [encoder setBytes:&seed_offset length:sizeof(uint) atIndex:6];

        MTLSize gridSize = MTLSizeMake(xdim, ydim, 1);
        MTLSize threadgroupSize = MTLSizeMake(MIN(xdim, 16ul), MIN(ydim, 16ul), 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        [encoder endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
    }
}

void generateNormalDistribution3D(
    void *rng_states,
    XFLOAT *g_out_real,
    XFLOAT *g_out_imag,
    XFLOAT *g_spectra,
    unsigned long xdim,
    unsigned long ydim,
    unsigned long zdim,
    deviceStream_t stream)
{
    @autoreleasepool {
        initMetalIfNeeded();

        id<MTLComputePipelineState> pipeline = getPipeline("metal_kernel_RNDnormalDistribution3D");
        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)rng_states offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_out_real offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_out_imag offset:0 atIndex:2];
        [encoder setBuffer:(__bridge id<MTLBuffer>)g_spectra offset:0 atIndex:3];

        uint xd = (uint)xdim;
        uint yd = (uint)ydim;
        uint zd = (uint)zdim;
        uint seed_offset = g_rng_seed_offset++;

        [encoder setBytes:&xd length:sizeof(uint) atIndex:4];
        [encoder setBytes:&yd length:sizeof(uint) atIndex:5];
        [encoder setBytes:&zd length:sizeof(uint) atIndex:6];
        [encoder setBytes:&seed_offset length:sizeof(uint) atIndex:7];

        MTLSize gridSize = MTLSizeMake(xdim, ydim, zdim);
        MTLSize threadgroupSize = MTLSizeMake(MIN(xdim, 8ul), MIN(ydim, 8ul), MIN(zdim, 4ul));
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        [encoder endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
    }
}

} // namespace MetalKernels

#endif // _METAL_ENABLED
