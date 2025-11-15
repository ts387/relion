#include <metal_stdlib>
using namespace metal;

// Include common definitions from helper
typedef float XFLOAT;

// ============================================================================
// Basic utility kernels
// ============================================================================

// Multiply array by scalar
kernel void metal_kernel_multiply(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& multiplier [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        output[gid] = input[gid] * multiplier;
    }
}

// Exponentiate with offset
kernel void metal_kernel_exponentiate(
    device float* g_array [[buffer(0)]],
    constant float& add [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx < size) {
        float a = g_array[idx] + add;

#ifdef ACC_DOUBLE_PRECISION
        if (a < -700.0) {
            g_array[idx] = 0.0f;
        } else {
            g_array[idx] = exp(a);
        }
#else
        if (a < -88.0f) {
            g_array[idx] = 0.0f;
        } else {
            g_array[idx] = exp(a);
        }
#endif
    }
}

// Soft mask outside map
kernel void metal_kernel_softMaskOutsideMap(
    device float* vol [[buffer(0)]],
    constant uint& vol_size [[buffer(1)]],
    constant uint& xdim [[buffer(2)]],
    constant uint& ydim [[buffer(3)]],
    constant uint& zdim [[buffer(4)]],
    constant uint& xinit [[buffer(5)]],
    constant uint& yinit [[buffer(6)]],
    constant uint& zinit [[buffer(7)]],
    constant float& radius [[buffer(8)]],
    constant float& cosine_width [[buffer(9)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx < vol_size) {
        uint z = idx / (xdim * ydim);
        uint xy = idx % (xdim * ydim);
        uint y = xy / xdim;
        uint x = xy % xdim;

        int xp = int(x) - int(xinit);
        int yp = int(y) - int(yinit);
        int zp = int(z) - int(zinit);

        float r = sqrt(float(xp * xp + yp * yp + zp * zp));

        if (r > radius) {
            vol[idx] = 0.0f;
        } else if (r > radius - cosine_width) {
            float diff = r - (radius - cosine_width);
            float mask_value = 0.5f + 0.5f * cos(M_PI_F * diff / cosine_width);
            vol[idx] *= mask_value;
        }
    }
}

// Weights exponent (coarse)
kernel void metal_kernel_weights_exponent_coarse(
    device const float* g_pdf_orientation [[buffer(0)]],
    device const bool* g_pdf_orientation_zeros [[buffer(1)]],
    device const float* g_pdf_offset [[buffer(2)]],
    device const bool* g_pdf_offset_zeros [[buffer(3)]],
    device float* g_weights [[buffer(4)]],
    constant float& g_min_diff2 [[buffer(5)]],
    constant uint& nr_coarse_orient [[buffer(6)]],
    constant uint& nr_coarse_trans [[buffer(7)]],
    constant uint& max_idx [[buffer(8)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx < max_idx) {
        uint itrans = idx % nr_coarse_trans;
        uint iorient = (idx - itrans) / nr_coarse_trans;

        float diff2 = g_weights[idx];

        if (diff2 < g_min_diff2 ||
            g_pdf_orientation_zeros[iorient] ||
            g_pdf_offset_zeros[itrans]) {
            g_weights[idx] = -99e99f; // large negative number
        } else {
            g_weights[idx] = g_pdf_orientation[iorient] +
                           g_pdf_offset[itrans] +
                           g_min_diff2 - diff2;
        }
    }
}

// Center FFT (shift zero-frequency component to center)
kernel void metal_kernel_centerFFT_2D(
    device float* img_real [[buffer(0)]],
    device float* img_imag [[buffer(1)]],
    constant uint& xdim [[buffer(2)]],
    constant uint& ydim [[buffer(3)]],
    constant uint& xshift [[buffer(4)]],
    constant uint& yshift [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint x = gid.x;
    uint y = gid.y;

    if (x < xdim && y < ydim) {
        // Phase shift for centering
        float phase = M_PI_F * (float(x * xshift) / float(xdim) +
                                 float(y * yshift) / float(ydim));

        uint idx = y * xdim + x;
        float real = img_real[idx];
        float imag = img_imag[idx];

        float cos_phase = cos(phase);
        float sin_phase = sin(phase);

        img_real[idx] = real * cos_phase - imag * sin_phase;
        img_imag[idx] = real * sin_phase + imag * cos_phase;
    }
}

// Probability ratio kernel
kernel void metal_kernel_probRatio(
    device const float* g_Mccf [[buffer(0)]],
    device const float* g_Mmean [[buffer(1)]],
    device const float* g_Mstddev [[buffer(2)]],
    device float* g_Mweight [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    constant float& normfft [[buffer(5)]],
    constant float& sum_ref_under_circ_mask [[buffer(6)]],
    constant float& expected_Pratio [[buffer(7)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx < size) {
        float ccf = g_Mccf[idx];
        float mean = g_Mmean[idx];
        float stddev = g_Mstddev[idx];

        // Compute diff2
        float diff2 = ccf - sum_ref_under_circ_mask * mean;
        diff2 /= (sqrt(sum_ref_under_circ_mask) * stddev);
        diff2 = diff2 * normfft * normfft;

        // Convert to probability ratio
        g_Mweight[idx] = expected_Pratio - diff2;
    }
}

// Finalize Mstddev kernel
kernel void metal_kernel_finalizeMstddev(
    device const float* g_Mstddev2 [[buffer(0)]],
    device const float* g_Mmean [[buffer(1)]],
    device float* g_Mstddev [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    constant float& normfft [[buffer(4)]],
    constant uint& num_particles [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx < size) {
        float mean = g_Mmean[idx];
        float stddev2 = g_Mstddev2[idx];

        stddev2 = stddev2 / float(num_particles);
        stddev2 -= mean * mean;

        if (stddev2 > 0.0f) {
            g_Mstddev[idx] = sqrt(stddev2);
        } else {
            g_Mstddev[idx] = 1.0f; // Avoid division by zero
        }
    }
}

// Convolution helper (multiply in Fourier space)
kernel void metal_kernel_convol_A(
    device const float* d_A_real [[buffer(0)]],
    device const float* d_A_imag [[buffer(1)]],
    device const float* d_B_real [[buffer(2)]],
    device const float* d_B_imag [[buffer(3)]],
    device float* d_C_real [[buffer(4)]],
    device float* d_C_imag [[buffer(5)]],
    constant uint& size [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx < size) {
        float ar = d_A_real[idx];
        float ai = d_A_imag[idx];
        float br = d_B_real[idx];
        float bi = d_B_imag[idx];

        // Complex multiplication: (ar + i*ai) * (br + i*bi)
        d_C_real[idx] = ar * br - ai * bi;
        d_C_imag[idx] = ar * bi + ai * br;
    }
}

// Window FFT
kernel void metal_kernel_window_FT(
    device float* g_in_real [[buffer(0)]],
    device float* g_in_imag [[buffer(1)]],
    constant uint& iX [[buffer(2)]],
    constant uint& iY [[buffer(3)]],
    constant uint& iZ [[buffer(4)]],
    constant uint& iYX [[buffer(5)]],
    constant uint& oX [[buffer(6)]],
    constant uint& oY [[buffer(7)]],
    constant uint& oZ [[buffer(8)]],
    constant uint& oYX [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint x = gid.x;
    uint y = gid.y;
    uint z = gid.z;

    if (x < oX && y < oY && z < oZ) {
        // Check if within input bounds
        if (x < iX && y < iY && z < iZ) {
            uint i_idx = z * iYX + y * iX + x;
            uint o_idx = z * oYX + y * oX + x;

            // Just copy if within bounds
            if (i_idx != o_idx) {
                g_in_real[o_idx] = g_in_real[i_idx];
                g_in_imag[o_idx] = g_in_imag[i_idx];
            }
        } else {
            // Zero padding
            uint o_idx = z * oYX + y * oX + x;
            g_in_real[o_idx] = 0.0f;
            g_in_imag[o_idx] = 0.0f;
        }
    }
}
