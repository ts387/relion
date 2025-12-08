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

// Helper macros
#define ceilfracf(a, b) (((a) + (b) - 1) / (b))
#define floorfracf(a, b) ((a) / (b))

// ============================================================================
// Projection helper functions
// ============================================================================

// Project 2D model (slice extraction)
inline void project2Dmodel_notex(
    device const XFLOAT* mdlReal,
    device const XFLOAT* mdlImag,
    constant ProjectorParams& proj,
    int x, int y,
    XFLOAT e0, XFLOAT e1,
    XFLOAT e3, XFLOAT e4,
    thread XFLOAT& real, thread XFLOAT& imag)
{
    XFLOAT xp = (e0 * x + e1 * y) * proj.padding_factor;
    XFLOAT yp = (e3 * x + e4 * y) * proj.padding_factor;

    int r2 = xp*xp + yp*yp;

    if (r2 <= proj.maxR2_padded) {
        bool invers = (xp < 0);
        if (invers) {
            xp = -xp;
            yp = -yp;
        }

        yp -= proj.mdlInitY;

        // Bilinear interpolation
        int x0 = int(floor(xp));
        int y0 = int(floor(yp));

        XFLOAT fx = xp - x0;
        XFLOAT fy = yp - y0;

        // mdlXY = mdlX * mdlY, so mdlY = mdlXY / mdlX
        int mdlY = proj.mdlXY / proj.mdlX;
        x0 = clamp(x0, 0, proj.mdlX - 2);
        y0 = clamp(y0, 0, mdlY - 2);

        int x1 = x0 + 1;
        int y1 = y0 + 1;

        int idx00 = y0 * proj.mdlX + x0;
        int idx01 = y1 * proj.mdlX + x0;
        int idx10 = y0 * proj.mdlX + x1;
        int idx11 = y1 * proj.mdlX + x1;

        XFLOAT c0_r = mdlReal[idx00] * (1-fx) + mdlReal[idx10] * fx;
        XFLOAT c1_r = mdlReal[idx01] * (1-fx) + mdlReal[idx11] * fx;
        real = c0_r * (1-fy) + c1_r * fy;

        XFLOAT c0_i = mdlImag[idx00] * (1-fx) + mdlImag[idx10] * fx;
        XFLOAT c1_i = mdlImag[idx01] * (1-fx) + mdlImag[idx11] * fx;
        imag = -(c0_i * (1-fy) + c1_i * fy);

        if (invers) {
            imag = -imag;
        }
    } else {
        real = 0;
        imag = 0;
    }
}

// Project 3D model (full 3D projection)
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

        yp -= proj.mdlInitY;
        zp -= proj.mdlInitZ;

        // Trilinear interpolation
        int x0 = int(floor(xp));
        int y0 = int(floor(yp));
        int z0 = int(floor(zp));

        XFLOAT fx = xp - x0;
        XFLOAT fy = yp - y0;
        XFLOAT fz = zp - z0;

        // mdlXY = mdlX * mdlY, so mdlY = mdlXY / mdlX
        int mdlY = proj.mdlXY / proj.mdlX;
        x0 = clamp(x0, 0, proj.mdlX - 2);
        y0 = clamp(y0, 0, mdlY - 2);
        z0 = clamp(z0, 0, proj.mdlZ - 2);

        int x1 = x0 + 1;
        int y1 = y0 + 1;
        int z1 = z0 + 1;

        int idx000 = z0 * proj.mdlXY + y0 * proj.mdlX + x0;
        int idx001 = z1 * proj.mdlXY + y0 * proj.mdlX + x0;
        int idx010 = z0 * proj.mdlXY + y1 * proj.mdlX + x0;
        int idx011 = z1 * proj.mdlXY + y1 * proj.mdlX + x0;
        int idx100 = z0 * proj.mdlXY + y0 * proj.mdlX + x1;
        int idx101 = z1 * proj.mdlXY + y0 * proj.mdlX + x1;
        int idx110 = z0 * proj.mdlXY + y1 * proj.mdlX + x1;
        int idx111 = z1 * proj.mdlXY + y1 * proj.mdlX + x1;

        XFLOAT c00_r = mdlReal[idx000] * (1-fx) + mdlReal[idx100] * fx;
        XFLOAT c01_r = mdlReal[idx001] * (1-fx) + mdlReal[idx101] * fx;
        XFLOAT c10_r = mdlReal[idx010] * (1-fx) + mdlReal[idx110] * fx;
        XFLOAT c11_r = mdlReal[idx011] * (1-fx) + mdlReal[idx111] * fx;
        XFLOAT c0_r = c00_r * (1-fy) + c10_r * fy;
        XFLOAT c1_r = c01_r * (1-fy) + c11_r * fy;
        real = c0_r * (1-fz) + c1_r * fz;

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
// Full diff2_coarse kernel implementation (2D version)
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
    constant int& block_sz [[buffer(13)]],
    constant int& prefetch_fraction [[buffer(14)]],
    threadgroup XFLOAT* shared_mem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint threadgroup_size [[threads_per_threadgroup]])
{
    // Shared memory layout
    threadgroup XFLOAT* s_eulers = shared_mem;
    threadgroup XFLOAT* s_ref_real = &shared_mem[eulers_per_block * 9];
    threadgroup XFLOAT* s_ref_imag = &s_ref_real[block_sz/prefetch_fraction * eulers_per_block];
    threadgroup XFLOAT* s_real = &s_ref_imag[block_sz/prefetch_fraction * eulers_per_block];
    threadgroup XFLOAT* s_imag = &s_real[block_sz];
    threadgroup XFLOAT* s_corr = &s_imag[block_sz];

    // Prefetch Euler matrices
    int max_block_pass_euler = ceilfracf(eulers_per_block * 9, block_sz) * block_sz;
    for (int i = tid; i < max_block_pass_euler; i += block_sz) {
        if (i < eulers_per_block * 9) {
            s_eulers[i] = g_eulers[bid * eulers_per_block * 9 + i];
        }
    }

    // Initialize diff2 accumulators
    XFLOAT diff2s[16]; // Max eulers_per_block = 16
    for (int i = 0; i < eulers_per_block; i++) {
        diff2s[i] = 0.0f;
    }

    // Get translation for this thread
    XFLOAT tx = trans_x[tid % translation_num];
    XFLOAT ty = trans_y[tid % translation_num];

    int max_block_pass_pixel = ceilfracf(image_size, block_sz) * block_sz;

    for (int init_pixel = 0; init_pixel < max_block_pass_pixel; init_pixel += block_sz/prefetch_fraction)
    {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Prefetch reference projections
        if (init_pixel + tid/prefetch_fraction < image_size)
        {
            int pixel_idx = init_pixel + tid/prefetch_fraction;
            int x = pixel_idx % projector.imgX;
            int y = floorfracf(pixel_idx, projector.imgX);

            if (y > projector.maxR)
                y -= projector.imgY;

            for (int i = tid % prefetch_fraction; i < eulers_per_block; i += prefetch_fraction)
            {
                XFLOAT ref_r, ref_i;
                project2Dmodel_notex(
                    mdlReal, mdlImag, projector,
                    x, y,
                    s_eulers[i*9], s_eulers[i*9+1],
                    s_eulers[i*9+3], s_eulers[i*9+4],
                    ref_r, ref_i);

                s_ref_real[eulers_per_block * (tid/prefetch_fraction) + i] = ref_r;
                s_ref_imag[eulers_per_block * (tid/prefetch_fraction) + i] = ref_i;
            }
        }

        // Prefetch image data
        if (init_pixel % block_sz == 0 && init_pixel + tid < image_size)
        {
            s_real[tid] = g_real[init_pixel + tid];
            s_imag[tid] = g_imag[init_pixel + tid];
            s_corr[tid] = g_corr[init_pixel + tid] / 2.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute differences
        if (tid/translation_num < block_sz/translation_num)
        {
            for (int i = tid/translation_num; i < block_sz/prefetch_fraction; i += block_sz/translation_num)
            {
                if ((init_pixel + i) >= image_size) break;

                int pixel_idx = init_pixel + i;
                int x = pixel_idx % projector.imgX;
                int y = floorfracf(pixel_idx, projector.imgX);

                if (y > projector.maxR)
                    y -= projector.imgY;

                XFLOAT real, imag;
                translatePixel(x, y, tx, ty,
                              s_real[i + init_pixel % block_sz],
                              s_imag[i + init_pixel % block_sz],
                              real, imag);

                for (int j = 0; j < eulers_per_block; j++)
                {
                    XFLOAT diff_real = s_ref_real[eulers_per_block * i + j] - real;
                    XFLOAT diff_imag = s_ref_imag[eulers_per_block * i + j] - imag;
                    diff2s[j] += (diff_real * diff_real + diff_imag * diff_imag) *
                                 s_corr[i + init_pixel % block_sz];
                }
            }
        }
    }

    // Atomic add to global memory
    for (int i = 0; i < eulers_per_block; i++)
    {
        atomic_fetch_add_explicit(&g_diff2s[(bid * eulers_per_block + i) * translation_num + tid % translation_num],
                                  diff2s[i],
                                  memory_order_relaxed);
    }
}

// ============================================================================
// Backprojection kernel (2D version)
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
    constant int& image_size [[buffer(10)]],
    constant int& translation_idx [[buffer(11)]],
    constant float& weight [[buffer(12)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < image_size)
    {
        // Extract pixel coordinates
        int x = gid % projector.imgX;
        int y = floorfracf(gid, projector.imgX);

        if (y > projector.maxR)
            y -= projector.imgY;

        // Get translation
        XFLOAT tx = trans_x[translation_idx];
        XFLOAT ty = trans_y[translation_idx];

        // Apply inverse translation
        XFLOAT img_real, img_imag;
        translatePixel(x, y, -tx, -ty, g_img_real[gid], g_img_imag[gid], img_real, img_imag);

        // Apply inverse rotation (transpose of rotation matrix)
        XFLOAT xp = g_eulers[0] * x + g_eulers[3] * y;
        XFLOAT yp = g_eulers[1] * x + g_eulers[4] * y;

        xp *= projector.padding_factor;
        yp *= projector.padding_factor;

        // Check if within bounds
        int r2 = xp*xp + yp*yp;
        if (r2 <= projector.maxR2_padded)
        {
            // Handle Hermitian symmetry
            bool invers = (xp < 0);
            if (invers) {
                xp = -xp;
                yp = -yp;
                img_imag = -img_imag;
            }

            yp -= projector.mdlInitY;

            // Compute model indices for bi-linear splat
            int x0 = int(floor(xp));
            int y0 = int(floor(yp));
            XFLOAT fx = xp - x0;
            XFLOAT fy = yp - y0;

            // mdlXY = mdlX * mdlY, so mdlY = mdlXY / mdlX
            int mdlY = projector.mdlXY / projector.mdlX;
            if (x0 >= 0 && x0 < projector.mdlX - 1 &&
                y0 >= 0 && y0 < mdlY - 1)
            {
                // Weighted bilinear splat into 4 neighboring voxels
                XFLOAT w00 = weight * (1-fx) * (1-fy);
                XFLOAT w01 = weight * (1-fx) * fy;
                XFLOAT w10 = weight * fx * (1-fy);
                XFLOAT w11 = weight * fx * fy;

                int idx00 = y0 * projector.mdlX + x0;
                int idx01 = (y0+1) * projector.mdlX + x0;
                int idx10 = y0 * projector.mdlX + (x0+1);
                int idx11 = (y0+1) * projector.mdlX + (x0+1);

                atomic_fetch_add_explicit(&g_mdl_real[idx00], img_real * w00, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_mdl_imag[idx00], img_imag * w00, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_mdl_weight[idx00], w00, memory_order_relaxed);

                atomic_fetch_add_explicit(&g_mdl_real[idx01], img_real * w01, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_mdl_imag[idx01], img_imag * w01, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_mdl_weight[idx01], w01, memory_order_relaxed);

                atomic_fetch_add_explicit(&g_mdl_real[idx10], img_real * w10, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_mdl_imag[idx10], img_imag * w10, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_mdl_weight[idx10], w10, memory_order_relaxed);

                atomic_fetch_add_explicit(&g_mdl_real[idx11], img_real * w11, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_mdl_imag[idx11], img_imag * w11, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_mdl_weight[idx11], w11, memory_order_relaxed);
            }
        }
    }
}

// ============================================================================
// Weighted averaging kernel
// ============================================================================

kernel void metal_kernel_wavg(
    device const XFLOAT* g_img_real [[buffer(0)]],
    device const XFLOAT* g_img_imag [[buffer(1)]],
    device const XFLOAT* g_img_weight [[buffer(2)]],
    device atomic<float>* g_out_real [[buffer(3)]],
    device atomic<float>* g_out_imag [[buffer(4)]],
    device atomic<float>* g_out_weight [[buffer(5)]],
    constant float& particle_weight [[buffer(6)]],
    constant int& size [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size)
    {
        XFLOAT img_weight = g_img_weight[gid];
        if (img_weight > 0.0f)
        {
            XFLOAT contrib_real = g_img_real[gid] * particle_weight;
            XFLOAT contrib_imag = g_img_imag[gid] * particle_weight;
            XFLOAT contrib_weight = img_weight * particle_weight;

            atomic_fetch_add_explicit(&g_out_real[gid], contrib_real, memory_order_relaxed);
            atomic_fetch_add_explicit(&g_out_imag[gid], contrib_imag, memory_order_relaxed);
            atomic_fetch_add_explicit(&g_out_weight[gid], contrib_weight, memory_order_relaxed);
        }
    }
}

// ============================================================================
// diff2_fine kernel (2D version) - Single orientation per block with dynamic scheduling
// ============================================================================

kernel void metal_kernel_diff2_fine_2D(
    device const XFLOAT* g_eulers [[buffer(0)]],
    device const XFLOAT* g_imgs_real [[buffer(1)]],
    device const XFLOAT* g_imgs_imag [[buffer(2)]],
    device const XFLOAT* trans_x [[buffer(3)]],
    device const XFLOAT* trans_y [[buffer(4)]],
    device const XFLOAT* mdlReal [[buffer(5)]],
    device const XFLOAT* mdlImag [[buffer(6)]],
    constant ProjectorParams& projector [[buffer(7)]],
    device const XFLOAT* g_corr_img [[buffer(8)]],
    device XFLOAT* g_diff2s [[buffer(9)]],
    constant uint& image_size [[buffer(10)]],
    constant XFLOAT& sum_init [[buffer(11)]],
    constant uint& orientation_num [[buffer(12)]],
    constant uint& translation_num [[buffer(13)]],
    device const uint* d_rot_idx [[buffer(14)]],
    device const uint* d_trans_idx [[buffer(15)]],
    device const uint* d_job_idx [[buffer(16)]],
    device const uint* d_job_num [[buffer(17)]],
    constant int& block_sz [[buffer(18)]],
    constant int& chunk_sz [[buffer(19)]],
    threadgroup XFLOAT* s [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]])
{
    // Local storage for outputs
    threadgroup XFLOAT s_outs[32]; // Max chunk_sz = 32

    // How many translations for this job
    uint trans_num = d_job_num[bid];

    // Initialize shared memory accumulators for each translation
    for (uint itrans = 0; itrans < trans_num; itrans++)
    {
        s[itrans * block_sz + tid] = 0.0f;
    }

    // Get the orientation index for this block
    uint ix = d_rot_idx[d_job_idx[bid]];
    uint iy; // Translation index

    // Number of passes to cover all pixels
    uint pass_num = ceilfracf(image_size, block_sz);

    // Process all pixels for this single orientation
    for (uint pass = 0; pass < pass_num; pass++)
    {
        uint pixel = pass * block_sz + tid;

        if (pixel < image_size)
        {
            // Extract pixel coordinates
            int x = pixel % projector.imgX;
            int y = floorfracf(pixel, projector.imgX);

            if (y > projector.maxR)
            {
                if (y >= projector.imgY - projector.maxR)
                    y = y - projector.imgY;
                else
                    x = projector.maxR; // Invalid pixel, will be outside bounds
            }

            // Project reference model for this single orientation
            XFLOAT ref_real, ref_imag;
            project2Dmodel_notex(
                mdlReal, mdlImag, projector,
                x, y,
                g_eulers[ix*9], g_eulers[ix*9+1],
                g_eulers[ix*9+3], g_eulers[ix*9+4],
                ref_real, ref_imag);

            // Compute diff2 for each translation
            for (uint itrans = 0; itrans < trans_num; itrans++)
            {
                iy = d_trans_idx[d_job_idx[bid]] + itrans;

                // Apply translation to image pixel
                XFLOAT shifted_real, shifted_imag;
                translatePixel(x, y, trans_x[iy], trans_y[iy],
                              g_imgs_real[pixel], g_imgs_imag[pixel],
                              shifted_real, shifted_imag);

                // Compute squared difference
                XFLOAT diff_real = ref_real - shifted_real;
                XFLOAT diff_imag = ref_imag - shifted_imag;

                s[itrans * block_sz + tid] += (diff_real * diff_real + diff_imag * diff_imag) *
                                               0.5f * g_corr_img[pixel];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Block-wide parallel reduction for each translation
    for (int j = block_sz / 2; j > 0; j /= 2)
    {
        if (tid < j)
        {
            for (uint itrans = 0; itrans < trans_num; itrans++)
            {
                s[itrans * block_sz + tid] += s[itrans * block_sz + tid + j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write final results
    if (tid < trans_num)
    {
        s_outs[tid] = s[tid * block_sz] + sum_init;
        iy = d_job_idx[bid] + tid;
        g_diff2s[iy] += s_outs[tid];
    }
}

// ============================================================================
// diff2_fine kernel (3D version) - For 3D data
// ============================================================================

kernel void metal_kernel_diff2_fine_3D(
    device const XFLOAT* g_eulers [[buffer(0)]],
    device const XFLOAT* g_imgs_real [[buffer(1)]],
    device const XFLOAT* g_imgs_imag [[buffer(2)]],
    device const XFLOAT* trans_x [[buffer(3)]],
    device const XFLOAT* trans_y [[buffer(4)]],
    device const XFLOAT* trans_z [[buffer(5)]],
    device const XFLOAT* mdlReal [[buffer(6)]],
    device const XFLOAT* mdlImag [[buffer(7)]],
    constant ProjectorParams& projector [[buffer(8)]],
    device const XFLOAT* g_corr_img [[buffer(9)]],
    device XFLOAT* g_diff2s [[buffer(10)]],
    constant uint& image_size [[buffer(11)]],
    constant XFLOAT& sum_init [[buffer(12)]],
    constant uint& orientation_num [[buffer(13)]],
    constant uint& translation_num [[buffer(14)]],
    device const uint* d_rot_idx [[buffer(15)]],
    device const uint* d_trans_idx [[buffer(16)]],
    device const uint* d_job_idx [[buffer(17)]],
    device const uint* d_job_num [[buffer(18)]],
    constant int& block_sz [[buffer(19)]],
    constant int& chunk_sz [[buffer(20)]],
    threadgroup XFLOAT* s [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]])
{
    threadgroup XFLOAT s_outs[32];

    uint trans_num = d_job_num[bid];

    // Initialize accumulators
    for (uint itrans = 0; itrans < trans_num; itrans++)
    {
        s[itrans * block_sz + tid] = 0.0f;
    }

    uint ix = d_rot_idx[d_job_idx[bid]];
    uint iy;
    uint pass_num = ceilfracf(image_size, block_sz);

    for (uint pass = 0; pass < pass_num; pass++)
    {
        uint pixel = pass * block_sz + tid;

        if (pixel < image_size)
        {
            // 3D coordinate extraction
            int z = floorfracf(pixel, projector.imgX * projector.imgY);
            int xy = pixel % (projector.imgX * projector.imgY);
            int x = xy % projector.imgX;
            int y = floorfracf(xy, projector.imgX);

            if (z > projector.maxR)
            {
                if (z >= projector.imgZ - projector.maxR)
                    z = z - projector.imgZ;
                else
                    x = projector.maxR;
            }
            if (y > projector.maxR)
            {
                if (y >= projector.imgY - projector.maxR)
                    y = y - projector.imgY;
                else
                    x = projector.maxR;
            }

            // Project 3D model
            XFLOAT ref_real, ref_imag;
            project3Dmodel_notex(
                mdlReal, mdlImag, projector,
                x, y, z,
                g_eulers[ix*9], g_eulers[ix*9+1], g_eulers[ix*9+2],
                g_eulers[ix*9+3], g_eulers[ix*9+4], g_eulers[ix*9+5],
                g_eulers[ix*9+6], g_eulers[ix*9+7], g_eulers[ix*9+8],
                ref_real, ref_imag);

            for (uint itrans = 0; itrans < trans_num; itrans++)
            {
                iy = d_trans_idx[d_job_idx[bid]] + itrans;

                XFLOAT shifted_real, shifted_imag;
                translatePixel3D(x, y, z, trans_x[iy], trans_y[iy], trans_z[iy],
                                g_imgs_real[pixel], g_imgs_imag[pixel],
                                shifted_real, shifted_imag);

                XFLOAT diff_real = ref_real - shifted_real;
                XFLOAT diff_imag = ref_imag - shifted_imag;

                s[itrans * block_sz + tid] += (diff_real * diff_real + diff_imag * diff_imag) *
                                               0.5f * g_corr_img[pixel];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Parallel reduction
    for (int j = block_sz / 2; j > 0; j /= 2)
    {
        if (tid < j)
        {
            for (uint itrans = 0; itrans < trans_num; itrans++)
            {
                s[itrans * block_sz + tid] += s[itrans * block_sz + tid + j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < trans_num)
    {
        s_outs[tid] = s[tid * block_sz] + sum_init;
        iy = d_job_idx[bid] + tid;
        g_diff2s[iy] += s_outs[tid];
    }
}

// ============================================================================
// diff2_coarse 3D kernel
// ============================================================================

kernel void metal_kernel_diff2_coarse_3D(
    device const XFLOAT* g_eulers [[buffer(0)]],
    device const XFLOAT* trans_x [[buffer(1)]],
    device const XFLOAT* trans_y [[buffer(2)]],
    device const XFLOAT* trans_z [[buffer(3)]],
    device const XFLOAT* g_real [[buffer(4)]],
    device const XFLOAT* g_imag [[buffer(5)]],
    device const XFLOAT* mdlReal [[buffer(6)]],
    device const XFLOAT* mdlImag [[buffer(7)]],
    constant ProjectorParams& projector [[buffer(8)]],
    device const XFLOAT* g_corr [[buffer(9)]],
    device atomic<float>* g_diff2s [[buffer(10)]],
    constant int& translation_num [[buffer(11)]],
    constant int& image_size [[buffer(12)]],
    constant int& eulers_per_block [[buffer(13)]],
    constant int& block_sz [[buffer(14)]],
    constant int& prefetch_fraction [[buffer(15)]],
    threadgroup XFLOAT* shared_mem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint threadgroup_size [[threads_per_threadgroup]])
{
    // Shared memory layout
    threadgroup XFLOAT* s_eulers = shared_mem;
    threadgroup XFLOAT* s_ref_real = &shared_mem[eulers_per_block * 9];
    threadgroup XFLOAT* s_ref_imag = &s_ref_real[block_sz/prefetch_fraction * eulers_per_block];
    threadgroup XFLOAT* s_real = &s_ref_imag[block_sz/prefetch_fraction * eulers_per_block];
    threadgroup XFLOAT* s_imag = &s_real[block_sz];
    threadgroup XFLOAT* s_corr = &s_imag[block_sz];

    // Prefetch Euler matrices
    int max_block_pass_euler = ceilfracf(eulers_per_block * 9, block_sz) * block_sz;
    for (int i = tid; i < max_block_pass_euler; i += block_sz) {
        if (i < eulers_per_block * 9) {
            s_eulers[i] = g_eulers[bid * eulers_per_block * 9 + i];
        }
    }

    // Initialize diff2 accumulators
    XFLOAT diff2s[16];
    for (int i = 0; i < eulers_per_block; i++) {
        diff2s[i] = 0.0f;
    }

    // Get translation for this thread
    XFLOAT tx = trans_x[tid % translation_num];
    XFLOAT ty = trans_y[tid % translation_num];
    XFLOAT tz = trans_z[tid % translation_num];

    int max_block_pass_pixel = ceilfracf(image_size, block_sz) * block_sz;

    for (int init_pixel = 0; init_pixel < max_block_pass_pixel; init_pixel += block_sz/prefetch_fraction)
    {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Prefetch reference projections for 3D
        if (init_pixel + tid/prefetch_fraction < image_size)
        {
            int pixel_idx = init_pixel + tid/prefetch_fraction;
            int z = floorfracf(pixel_idx, projector.imgX * projector.imgY);
            int xy = pixel_idx % (projector.imgX * projector.imgY);
            int x = xy % projector.imgX;
            int y = floorfracf(xy, projector.imgX);

            if (z > projector.maxR)
                z -= projector.imgZ;
            if (y > projector.maxR)
                y -= projector.imgY;

            for (int i = tid % prefetch_fraction; i < eulers_per_block; i += prefetch_fraction)
            {
                XFLOAT ref_r, ref_i;
                project3Dmodel_notex(
                    mdlReal, mdlImag, projector,
                    x, y, z,
                    s_eulers[i*9], s_eulers[i*9+1], s_eulers[i*9+2],
                    s_eulers[i*9+3], s_eulers[i*9+4], s_eulers[i*9+5],
                    s_eulers[i*9+6], s_eulers[i*9+7], s_eulers[i*9+8],
                    ref_r, ref_i);

                s_ref_real[eulers_per_block * (tid/prefetch_fraction) + i] = ref_r;
                s_ref_imag[eulers_per_block * (tid/prefetch_fraction) + i] = ref_i;
            }
        }

        // Prefetch image data
        if (init_pixel % block_sz == 0 && init_pixel + tid < image_size)
        {
            s_real[tid] = g_real[init_pixel + tid];
            s_imag[tid] = g_imag[init_pixel + tid];
            s_corr[tid] = g_corr[init_pixel + tid] / 2.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute differences
        if (tid/translation_num < block_sz/translation_num)
        {
            for (int i = tid/translation_num; i < block_sz/prefetch_fraction; i += block_sz/translation_num)
            {
                if ((init_pixel + i) >= image_size) break;

                int pixel_idx = init_pixel + i;
                int z = floorfracf(pixel_idx, projector.imgX * projector.imgY);
                int xy = pixel_idx % (projector.imgX * projector.imgY);
                int x = xy % projector.imgX;
                int y = floorfracf(xy, projector.imgX);

                if (z > projector.maxR)
                    z -= projector.imgZ;
                if (y > projector.maxR)
                    y -= projector.imgY;

                XFLOAT real, imag;
                translatePixel3D(x, y, z, tx, ty, tz,
                                s_real[i + init_pixel % block_sz],
                                s_imag[i + init_pixel % block_sz],
                                real, imag);

                for (int j = 0; j < eulers_per_block; j++)
                {
                    XFLOAT diff_real = s_ref_real[eulers_per_block * i + j] - real;
                    XFLOAT diff_imag = s_ref_imag[eulers_per_block * i + j] - imag;
                    diff2s[j] += (diff_real * diff_real + diff_imag * diff_imag) *
                                 s_corr[i + init_pixel % block_sz];
                }
            }
        }
    }

    // Atomic add to global memory
    for (int i = 0; i < eulers_per_block; i++)
    {
        atomic_fetch_add_explicit(&g_diff2s[(bid * eulers_per_block + i) * translation_num + tid % translation_num],
                                  diff2s[i],
                                  memory_order_relaxed);
    }
}

// ============================================================================
// backproject3D kernel
// ============================================================================

kernel void metal_kernel_backproject3D(
    device const XFLOAT* g_img_real [[buffer(0)]],
    device const XFLOAT* g_img_imag [[buffer(1)]],
    device const XFLOAT* trans_x [[buffer(2)]],
    device const XFLOAT* trans_y [[buffer(3)]],
    device const XFLOAT* trans_z [[buffer(4)]],
    device const XFLOAT* g_weights [[buffer(5)]],
    device const XFLOAT* g_eulers [[buffer(6)]],
    device atomic<float>* g_mdl_real [[buffer(7)]],
    device atomic<float>* g_mdl_imag [[buffer(8)]],
    device atomic<float>* g_mdl_weight [[buffer(9)]],
    constant ProjectorParams& projector [[buffer(10)]],
    constant int& image_size [[buffer(11)]],
    constant int& translation_idx [[buffer(12)]],
    constant float& weight [[buffer(13)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < image_size)
    {
        // Extract 3D coordinates
        int z = floorfracf(gid, projector.imgX * projector.imgY);
        int xy = gid % (projector.imgX * projector.imgY);
        int x = xy % projector.imgX;
        int y = floorfracf(xy, projector.imgX);

        if (z > projector.maxR)
            z -= projector.imgZ;
        if (y > projector.maxR)
            y -= projector.imgY;

        // Get translation
        XFLOAT tx = trans_x[translation_idx];
        XFLOAT ty = trans_y[translation_idx];
        XFLOAT tz = trans_z[translation_idx];

        // Apply inverse translation
        XFLOAT img_real, img_imag;
        translatePixel3D(x, y, z, -tx, -ty, -tz, g_img_real[gid], g_img_imag[gid], img_real, img_imag);

        // Apply inverse rotation (transpose of rotation matrix)
        XFLOAT xp = g_eulers[0] * x + g_eulers[3] * y + g_eulers[6] * z;
        XFLOAT yp = g_eulers[1] * x + g_eulers[4] * y + g_eulers[7] * z;
        XFLOAT zp = g_eulers[2] * x + g_eulers[5] * y + g_eulers[8] * z;

        xp *= projector.padding_factor;
        yp *= projector.padding_factor;
        zp *= projector.padding_factor;

        // Check bounds
        int r2 = xp*xp + yp*yp + zp*zp;
        if (r2 <= projector.maxR2_padded)
        {
            // Handle Hermitian symmetry
            bool invers = (xp < 0);
            if (invers) {
                xp = -xp;
                yp = -yp;
                zp = -zp;
                img_imag = -img_imag;
            }

            yp -= projector.mdlInitY;
            zp -= projector.mdlInitZ;

            // Trilinear splat
            int x0 = int(floor(xp));
            int y0 = int(floor(yp));
            int z0 = int(floor(zp));
            XFLOAT fx = xp - x0;
            XFLOAT fy = yp - y0;
            XFLOAT fz = zp - z0;

            // mdlXY = mdlX * mdlY, so mdlY = mdlXY / mdlX
            int mdlY = projector.mdlXY / projector.mdlX;
            if (x0 >= 0 && x0 < projector.mdlX - 1 &&
                y0 >= 0 && y0 < mdlY - 1 &&
                z0 >= 0 && z0 < projector.mdlZ - 1)
            {
                // 8 neighboring voxels for trilinear splat
                XFLOAT w000 = weight * (1-fx) * (1-fy) * (1-fz);
                XFLOAT w001 = weight * (1-fx) * (1-fy) * fz;
                XFLOAT w010 = weight * (1-fx) * fy * (1-fz);
                XFLOAT w011 = weight * (1-fx) * fy * fz;
                XFLOAT w100 = weight * fx * (1-fy) * (1-fz);
                XFLOAT w101 = weight * fx * (1-fy) * fz;
                XFLOAT w110 = weight * fx * fy * (1-fz);
                XFLOAT w111 = weight * fx * fy * fz;

                int idx000 = z0 * projector.mdlXY + y0 * projector.mdlX + x0;
                int idx001 = (z0+1) * projector.mdlXY + y0 * projector.mdlX + x0;
                int idx010 = z0 * projector.mdlXY + (y0+1) * projector.mdlX + x0;
                int idx011 = (z0+1) * projector.mdlXY + (y0+1) * projector.mdlX + x0;
                int idx100 = z0 * projector.mdlXY + y0 * projector.mdlX + (x0+1);
                int idx101 = (z0+1) * projector.mdlXY + y0 * projector.mdlX + (x0+1);
                int idx110 = z0 * projector.mdlXY + (y0+1) * projector.mdlX + (x0+1);
                int idx111 = (z0+1) * projector.mdlXY + (y0+1) * projector.mdlX + (x0+1);

                atomic_fetch_add_explicit(&g_mdl_real[idx000], img_real * w000, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_mdl_imag[idx000], img_imag * w000, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_mdl_weight[idx000], w000, memory_order_relaxed);

                atomic_fetch_add_explicit(&g_mdl_real[idx001], img_real * w001, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_mdl_imag[idx001], img_imag * w001, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_mdl_weight[idx001], w001, memory_order_relaxed);

                atomic_fetch_add_explicit(&g_mdl_real[idx010], img_real * w010, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_mdl_imag[idx010], img_imag * w010, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_mdl_weight[idx010], w010, memory_order_relaxed);

                atomic_fetch_add_explicit(&g_mdl_real[idx011], img_real * w011, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_mdl_imag[idx011], img_imag * w011, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_mdl_weight[idx011], w011, memory_order_relaxed);

                atomic_fetch_add_explicit(&g_mdl_real[idx100], img_real * w100, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_mdl_imag[idx100], img_imag * w100, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_mdl_weight[idx100], w100, memory_order_relaxed);

                atomic_fetch_add_explicit(&g_mdl_real[idx101], img_real * w101, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_mdl_imag[idx101], img_imag * w101, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_mdl_weight[idx101], w101, memory_order_relaxed);

                atomic_fetch_add_explicit(&g_mdl_real[idx110], img_real * w110, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_mdl_imag[idx110], img_imag * w110, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_mdl_weight[idx110], w110, memory_order_relaxed);

                atomic_fetch_add_explicit(&g_mdl_real[idx111], img_real * w111, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_mdl_imag[idx111], img_imag * w111, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_mdl_weight[idx111], w111, memory_order_relaxed);
            }
        }
    }
}
