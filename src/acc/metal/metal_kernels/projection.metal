#include <metal_stdlib>
using namespace metal;

typedef float XFLOAT;

// Structure to match AccProjectorKernel
struct ProjectorParams {
    int mdlX, mdlXY, mdlZ;
    int imgX, imgY, imgZ;
    int mdlInitY, mdlInitZ;
    int maxR, maxR2, maxR2_padded;
    XFLOAT padding_factor;
};

// ============================================================================
// Projection helper functions
// ============================================================================

// Trilinear interpolation for 3D model (no texture memory version)
inline void project3Dmodel_notex(
    device const XFLOAT* mdlReal,
    device const XFLOAT* mdlImag,
    constant ProjectorParams& proj,
    int x, int y, int z,
    XFLOAT e0, XFLOAT e1, XFLOAT e2,
    XFLOAT e3, XFLOAT e4, XFLOAT e5,
    XFLOAT e6, XFLOAT e7, XFLOAT e8,
    thread XFLOAT& real, thread XFLOAT& imag)
{
    // Apply rotation matrix
    XFLOAT xp = (e0 * x + e1 * y + e2 * z) * proj.padding_factor;
    XFLOAT yp = (e3 * x + e4 * y + e5 * z) * proj.padding_factor;
    XFLOAT zp = (e6 * x + e7 * y + e8 * z) * proj.padding_factor;

    int r2 = xp*xp + yp*yp + zp*zp;

    if (r2 <= proj.maxR2_padded) {
        bool invers = (xp < 0);
        if (invers) {
            xp = -xp;
            yp = -yp;
            zp = -zp;
        }

        // Adjust coordinates
        yp -= proj.mdlInitY;
        zp -= proj.mdlInitZ;

        // Trilinear interpolation (simplified)
        int x0 = int(floor(xp));
        int y0 = int(floor(yp));
        int z0 = int(floor(zp));

        XFLOAT fx = xp - x0;
        XFLOAT fy = yp - y0;
        XFLOAT fz = zp - z0;

        // Clamp to model bounds
        x0 = clamp(x0, 0, proj.mdlX - 2);
        y0 = clamp(y0, 0, proj.mdlX - 2); // mdlX used as dimension
        z0 = clamp(z0, 0, proj.mdlZ - 2);

        int x1 = x0 + 1;
        int y1 = y0 + 1;
        int z1 = z0 + 1;

        // Sample 8 corners
        int idx000 = z0 * proj.mdlXY + y0 * proj.mdlX + x0;
        int idx001 = z1 * proj.mdlXY + y0 * proj.mdlX + x0;
        int idx010 = z0 * proj.mdlXY + y1 * proj.mdlX + x0;
        int idx011 = z1 * proj.mdlXY + y1 * proj.mdlX + x0;
        int idx100 = z0 * proj.mdlXY + y0 * proj.mdlX + x1;
        int idx101 = z1 * proj.mdlXY + y0 * proj.mdlX + x1;
        int idx110 = z0 * proj.mdlXY + y1 * proj.mdlX + x1;
        int idx111 = z1 * proj.mdlXY + y1 * proj.mdlX + x1;

        // Trilinear interpolation for real part
        XFLOAT c00_r = mdlReal[idx000] * (1-fx) + mdlReal[idx100] * fx;
        XFLOAT c01_r = mdlReal[idx001] * (1-fx) + mdlReal[idx101] * fx;
        XFLOAT c10_r = mdlReal[idx010] * (1-fx) + mdlReal[idx110] * fx;
        XFLOAT c11_r = mdlReal[idx011] * (1-fx) + mdlReal[idx111] * fx;

        XFLOAT c0_r = c00_r * (1-fy) + c10_r * fy;
        XFLOAT c1_r = c01_r * (1-fy) + c11_r * fy;

        real = c0_r * (1-fz) + c1_r * fz;

        // Trilinear interpolation for imaginary part
        XFLOAT c00_i = mdlImag[idx000] * (1-fx) + mdlImag[idx100] * fx;
        XFLOAT c01_i = mdlImag[idx001] * (1-fx) + mdlImag[idx101] * fx;
        XFLOAT c10_i = mdlImag[idx010] * (1-fx) + mdlImag[idx110] * fx;
        XFLOAT c11_i = mdlImag[idx011] * (1-fx) + mdlImag[idx111] * fx;

        XFLOAT c0_i = c00_i * (1-fy) + c10_i * fy;
        XFLOAT c1_i = c01_i * (1-fy) + c11_i * fy;

        imag = -(c0_i * (1-fz) + c1_i * fz);

        if (invers) {
            imag = -imag;
        }
    } else {
        real = 0;
        imag = 0;
    }
}

// Translation (phase shift)
inline void translatePixel(
    int x, int y,
    XFLOAT tx, XFLOAT ty,
    XFLOAT inReal, XFLOAT inImag,
    thread XFLOAT& outReal, thread XFLOAT& outImag)
{
    XFLOAT dotp = tx * x + ty * y;
    XFLOAT cos_phase = cos(dotp);
    XFLOAT sin_phase = sin(dotp);

    outReal = inReal * cos_phase - inImag * sin_phase;
    outImag = inReal * sin_phase + inImag * cos_phase;
}

inline void translatePixel3D(
    int x, int y, int z,
    XFLOAT tx, XFLOAT ty, XFLOAT tz,
    XFLOAT inReal, XFLOAT inImag,
    thread XFLOAT& outReal, thread XFLOAT& outImag)
{
    XFLOAT dotp = tx * x + ty * y + tz * z;
    XFLOAT cos_phase = cos(dotp);
    XFLOAT sin_phase = sin(dotp);

    outReal = inReal * cos_phase - inImag * sin_phase;
    outImag = inReal * sin_phase + inImag * cos_phase;
}

// ============================================================================
// Simplified diff2 coarse kernel (2D case)
// ============================================================================

kernel void metal_kernel_diff2_coarse_2D(
    device const XFLOAT* g_eulers [[buffer(0)]],
    device const XFLOAT* trans_x [[buffer(1)]],
    device const XFLOAT* trans_y [[buffer(2)]],
    device const XFLOAT* g_real [[buffer(3)]],
    device const XFLOAT* g_imag [[buffer(4)]],
    device const XFLOAT* mdlReal [[buffer(5)]],
    device const XFLOAT* mdlImag [[buffer(6)]],
    constant ProjectorParams& projector [[buffer(7)]],
    device const XFLOAT* g_corr [[buffer(8)]],
    device atomic<float>* g_diff2s [[buffer(9)]],
    constant int& translation_num [[buffer(10)]],
    constant int& image_size [[buffer(11)]],
    constant int& eulers_per_block [[buffer(12)]],
    threadgroup XFLOAT* shared_mem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint block_size [[threads_per_threadgroup]])
{
    // Simplified version - full implementation would match CUDA kernel structure
    // This is a stub to demonstrate the pattern

    // Prefetch Euler angles into shared memory
    threadgroup XFLOAT* s_eulers = shared_mem;
    int euler_idx = bid * eulers_per_block * 9 + tid;
    if (tid < eulers_per_block * 9) {
        s_eulers[tid] = g_eulers[euler_idx];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // TODO: Implement full diff2_coarse logic
    // This would include:
    // 1. Prefetching reference projections
    // 2. Prefetching image data
    // 3. Computing differences
    // 4. Accumulating diff2 scores
}

// ============================================================================
// Simplified backprojection kernel
// ============================================================================

kernel void metal_kernel_backproject2D(
    device const XFLOAT* g_img_real [[buffer(0)]],
    device const XFLOAT* g_img_imag [[buffer(1)]],
    device const XFLOAT* trans_x [[buffer(2)]],
    device const XFLOAT* trans_y [[buffer(3)]],
    device const XFLOAT* g_weights [[buffer(4)]],
    device const XFLOAT* g_eulers [[buffer(5)]],
    device atomic<float>* g_mdl_real [[buffer(6)]],
    device atomic<float>* g_mdl_imag [[buffer(7)]],
    device atomic<float>* g_mdl_weight [[buffer(8)]],
    constant ProjectorParams& projector [[buffer(9)]],
    constant int& translation_num [[buffer(10)]],
    constant int& image_size [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    // Simplified backprojection stub
    if (gid < image_size) {
        // Extract pixel coordinates
        int x = gid % projector.imgX;
        int y = gid / projector.imgX;

        if (y > projector.maxR)
            y -= projector.imgY;

        // TODO: Implement full backprojection logic
        // This would include:
        // 1. Apply translation
        // 2. Apply inverse rotation
        // 3. Accumulate into 3D model using atomic operations
    }
}

// ============================================================================
// Placeholder kernels for other operations
// ============================================================================

// Wavg kernel stub
kernel void metal_kernel_wavg(
    device const XFLOAT* g_img_real [[buffer(0)]],
    device const XFLOAT* g_img_imag [[buffer(1)]],
    device const XFLOAT* g_weight [[buffer(2)]],
    device XFLOAT* g_out_real [[buffer(3)]],
    device XFLOAT* g_out_imag [[buffer(4)]],
    constant int& size [[buffer(5)]],
    threadgroup XFLOAT* shared_data [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[thread_position_in_grid]])
{
    // TODO: Implement weighted averaging
}
