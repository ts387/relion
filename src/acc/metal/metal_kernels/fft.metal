#include <metal_stdlib>
using namespace metal;

// ============================================================================
// GPU-Accelerated FFT Implementation for Metal
// Cooley-Tukey radix-2 FFT algorithm
// ============================================================================

typedef float XFLOAT;

// Complex number operations
struct Complex {
    XFLOAT real;
    XFLOAT imag;
};

inline Complex complex_add(Complex a, Complex b) {
    Complex c;
    c.real = a.real + b.real;
    c.imag = a.imag + b.imag;
    return c;
}

inline Complex complex_sub(Complex a, Complex b) {
    Complex c;
    c.real = a.real - b.real;
    c.imag = a.imag - b.imag;
    return c;
}

inline Complex complex_mul(Complex a, Complex b) {
    Complex c;
    c.real = a.real * b.real - a.imag * b.imag;
    c.imag = a.real * b.imag + a.imag * b.real;
    return c;
}

inline Complex complex_conj(Complex a) {
    Complex c;
    c.real = a.real;
    c.imag = -a.imag;
    return c;
}

// ============================================================================
// Bit-reversal permutation kernel
// ============================================================================

inline uint reverseBits(uint x, uint numBits) {
    uint result = 0;
    for (uint i = 0; i < numBits; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

kernel void metal_kernel_fft_bit_reversal(
    device const XFLOAT* input_real [[buffer(0)]],
    device const XFLOAT* input_imag [[buffer(1)]],
    device XFLOAT* output_real [[buffer(2)]],
    device XFLOAT* output_imag [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& log2n [[buffer(5)]],
    constant uint& batch_idx [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        uint rev_idx = reverseBits(gid, log2n);
        uint base_offset = batch_idx * n;

        output_real[base_offset + rev_idx] = input_real[base_offset + gid];
        output_imag[base_offset + rev_idx] = input_imag[base_offset + gid];
    }
}

// ============================================================================
// Butterfly operation for Cooley-Tukey FFT
// Each thread computes one butterfly
// ============================================================================

kernel void metal_kernel_fft_butterfly(
    device XFLOAT* data_real [[buffer(0)]],
    device XFLOAT* data_imag [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& stage [[buffer(3)]],          // Current stage (0 to log2n-1)
    constant int& direction [[buffer(4)]],       // 1 for forward, -1 for inverse
    constant uint& batch_idx [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint half_size = 1u << stage;           // 2^stage
    uint full_size = half_size << 1;        // 2^(stage+1)

    // Number of butterflies per group
    uint num_butterflies = n / 2;

    if (gid < num_butterflies) {
        // Determine which butterfly and which group
        uint group_idx = gid / half_size;
        uint pair_idx = gid % half_size;

        // Calculate indices for this butterfly
        uint base_offset = batch_idx * n;
        uint idx_a = base_offset + group_idx * full_size + pair_idx;
        uint idx_b = idx_a + half_size;

        // Calculate twiddle factor W_n^k = exp(-2*pi*i*k/n) for forward
        // For inverse: exp(+2*pi*i*k/n)
        XFLOAT angle = XFLOAT(direction) * (-2.0f) * M_PI_F * XFLOAT(pair_idx) / XFLOAT(full_size);

        Complex twiddle;
        twiddle.real = cos(angle);
        twiddle.imag = sin(angle);

        // Load data
        Complex a, b;
        a.real = data_real[idx_a];
        a.imag = data_imag[idx_a];
        b.real = data_real[idx_b];
        b.imag = data_imag[idx_b];

        // Apply twiddle factor to b
        Complex tb = complex_mul(twiddle, b);

        // Butterfly operation
        Complex out_a = complex_add(a, tb);
        Complex out_b = complex_sub(a, tb);

        // Store results
        data_real[idx_a] = out_a.real;
        data_imag[idx_a] = out_a.imag;
        data_real[idx_b] = out_b.real;
        data_imag[idx_b] = out_b.imag;
    }
}

// ============================================================================
// Scaling kernel for inverse FFT
// ============================================================================

kernel void metal_kernel_fft_scale(
    device XFLOAT* data_real [[buffer(0)]],
    device XFLOAT* data_imag [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant XFLOAT& scale [[buffer(3)]],
    constant uint& batch_idx [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        uint idx = batch_idx * n + gid;
        data_real[idx] *= scale;
        data_imag[idx] *= scale;
    }
}

// ============================================================================
// Real-to-Complex FFT post-processing
// Converts N-point complex FFT of packed real data to (N/2+1)-point complex output
// ============================================================================

kernel void metal_kernel_fft_r2c_postprocess(
    device const XFLOAT* fft_real [[buffer(0)]],
    device const XFLOAT* fft_imag [[buffer(1)]],
    device XFLOAT* output_real [[buffer(2)]],
    device XFLOAT* output_imag [[buffer(3)]],
    constant uint& n [[buffer(4)]],           // Original real size
    constant uint& batch_idx [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint half_n = n / 2;
    uint output_size = half_n + 1;

    if (gid <= half_n) {
        uint base_offset_in = batch_idx * n;
        uint base_offset_out = batch_idx * output_size;

        if (gid == 0) {
            // DC component
            output_real[base_offset_out] = fft_real[base_offset_in];
            output_imag[base_offset_out] = 0.0f;
        } else if (gid == half_n) {
            // Nyquist component (only for even n)
            if (n % 2 == 0) {
                output_real[base_offset_out + gid] = fft_real[base_offset_in];
                output_imag[base_offset_out + gid] = 0.0f;
            }
        } else {
            // General case: unpack from complex FFT of even/odd interleaved
            Complex F_k, F_nk;
            F_k.real = fft_real[base_offset_in + gid];
            F_k.imag = fft_imag[base_offset_in + gid];
            F_nk.real = fft_real[base_offset_in + n - gid];
            F_nk.imag = fft_imag[base_offset_in + n - gid];

            // Twiddle factor for R2C conversion
            XFLOAT angle = -M_PI_F * XFLOAT(gid) / XFLOAT(half_n);
            Complex twiddle;
            twiddle.real = cos(angle);
            twiddle.imag = sin(angle);

            // Extract real FFT components
            Complex F_nk_conj = complex_conj(F_nk);
            Complex sum = complex_add(F_k, F_nk_conj);
            Complex diff = complex_sub(F_k, F_nk_conj);

            // Apply twiddle
            diff.real *= -0.5f;
            diff.imag *= -0.5f;
            Complex tw_diff = complex_mul(twiddle, diff);

            output_real[base_offset_out + gid] = 0.5f * sum.real + tw_diff.imag;
            output_imag[base_offset_out + gid] = 0.5f * sum.imag - tw_diff.real;
        }
    }
}

// ============================================================================
// Complex-to-Real FFT pre-processing
// Prepares (N/2+1)-point complex input for N-point complex IFFT
// ============================================================================

kernel void metal_kernel_fft_c2r_preprocess(
    device const XFLOAT* input_real [[buffer(0)]],
    device const XFLOAT* input_imag [[buffer(1)]],
    device XFLOAT* ifft_real [[buffer(2)]],
    device XFLOAT* ifft_imag [[buffer(3)]],
    constant uint& n [[buffer(4)]],           // Target real size
    constant uint& batch_idx [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint half_n = n / 2;
    uint input_size = half_n + 1;

    if (gid < n) {
        uint base_offset_in = batch_idx * input_size;
        uint base_offset_out = batch_idx * n;

        if (gid <= half_n) {
            // Hermitian symmetry: output[k] = input[k]
            ifft_real[base_offset_out + gid] = input_real[base_offset_in + gid];
            ifft_imag[base_offset_out + gid] = input_imag[base_offset_in + gid];
        } else {
            // Hermitian symmetry: output[n-k] = conj(input[k])
            uint k = n - gid;
            ifft_real[base_offset_out + gid] = input_real[base_offset_in + k];
            ifft_imag[base_offset_out + gid] = -input_imag[base_offset_in + k];
        }
    }
}

// ============================================================================
// 2D FFT row-wise transform kernel
// Performs 1D FFT on each row
// ============================================================================

kernel void metal_kernel_fft2d_rows_butterfly(
    device XFLOAT* data_real [[buffer(0)]],
    device XFLOAT* data_imag [[buffer(1)]],
    constant uint& nx [[buffer(2)]],
    constant uint& ny [[buffer(3)]],
    constant uint& stage [[buffer(4)]],
    constant int& direction [[buffer(5)]],
    constant uint& batch_idx [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])  // gid.x = butterfly idx, gid.y = row idx
{
    uint row = gid.y;
    uint butterfly_idx = gid.x;

    if (row >= ny) return;

    uint half_size = 1u << stage;
    uint full_size = half_size << 1;
    uint num_butterflies = nx / 2;

    if (butterfly_idx < num_butterflies) {
        uint group_idx = butterfly_idx / half_size;
        uint pair_idx = butterfly_idx % half_size;

        // Calculate row base offset
        uint base_offset = (batch_idx * ny + row) * nx;
        uint idx_a = base_offset + group_idx * full_size + pair_idx;
        uint idx_b = idx_a + half_size;

        XFLOAT angle = XFLOAT(direction) * (-2.0f) * M_PI_F * XFLOAT(pair_idx) / XFLOAT(full_size);

        Complex twiddle;
        twiddle.real = cos(angle);
        twiddle.imag = sin(angle);

        Complex a, b;
        a.real = data_real[idx_a];
        a.imag = data_imag[idx_a];
        b.real = data_real[idx_b];
        b.imag = data_imag[idx_b];

        Complex tb = complex_mul(twiddle, b);
        Complex out_a = complex_add(a, tb);
        Complex out_b = complex_sub(a, tb);

        data_real[idx_a] = out_a.real;
        data_imag[idx_a] = out_a.imag;
        data_real[idx_b] = out_b.real;
        data_imag[idx_b] = out_b.imag;
    }
}

// ============================================================================
// 2D FFT column-wise transform kernel
// Performs 1D FFT on each column
// ============================================================================

kernel void metal_kernel_fft2d_cols_butterfly(
    device XFLOAT* data_real [[buffer(0)]],
    device XFLOAT* data_imag [[buffer(1)]],
    constant uint& nx [[buffer(2)]],
    constant uint& ny [[buffer(3)]],
    constant uint& stage [[buffer(4)]],
    constant int& direction [[buffer(5)]],
    constant uint& batch_idx [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])  // gid.x = col idx, gid.y = butterfly idx
{
    uint col = gid.x;
    uint butterfly_idx = gid.y;

    if (col >= nx) return;

    uint half_size = 1u << stage;
    uint full_size = half_size << 1;
    uint num_butterflies = ny / 2;

    if (butterfly_idx < num_butterflies) {
        uint group_idx = butterfly_idx / half_size;
        uint pair_idx = butterfly_idx % half_size;

        // Calculate column indices (strided access)
        uint base_offset = batch_idx * ny * nx;
        uint row_a = group_idx * full_size + pair_idx;
        uint row_b = row_a + half_size;
        uint idx_a = base_offset + row_a * nx + col;
        uint idx_b = base_offset + row_b * nx + col;

        XFLOAT angle = XFLOAT(direction) * (-2.0f) * M_PI_F * XFLOAT(pair_idx) / XFLOAT(full_size);

        Complex twiddle;
        twiddle.real = cos(angle);
        twiddle.imag = sin(angle);

        Complex a, b;
        a.real = data_real[idx_a];
        a.imag = data_imag[idx_a];
        b.real = data_real[idx_b];
        b.imag = data_imag[idx_b];

        Complex tb = complex_mul(twiddle, b);
        Complex out_a = complex_add(a, tb);
        Complex out_b = complex_sub(a, tb);

        data_real[idx_a] = out_a.real;
        data_imag[idx_a] = out_a.imag;
        data_real[idx_b] = out_b.real;
        data_imag[idx_b] = out_b.imag;
    }
}

// ============================================================================
// 2D bit-reversal for rows
// ============================================================================

kernel void metal_kernel_fft2d_bit_reversal_rows(
    device const XFLOAT* input_real [[buffer(0)]],
    device const XFLOAT* input_imag [[buffer(1)]],
    device XFLOAT* output_real [[buffer(2)]],
    device XFLOAT* output_imag [[buffer(3)]],
    constant uint& nx [[buffer(4)]],
    constant uint& ny [[buffer(5)]],
    constant uint& log2nx [[buffer(6)]],
    constant uint& batch_idx [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])  // gid.x = col, gid.y = row
{
    uint col = gid.x;
    uint row = gid.y;

    if (col < nx && row < ny) {
        uint rev_col = reverseBits(col, log2nx);
        uint base_offset = (batch_idx * ny + row) * nx;

        output_real[base_offset + rev_col] = input_real[base_offset + col];
        output_imag[base_offset + rev_col] = input_imag[base_offset + col];
    }
}

// ============================================================================
// 2D bit-reversal for columns
// ============================================================================

kernel void metal_kernel_fft2d_bit_reversal_cols(
    device const XFLOAT* input_real [[buffer(0)]],
    device const XFLOAT* input_imag [[buffer(1)]],
    device XFLOAT* output_real [[buffer(2)]],
    device XFLOAT* output_imag [[buffer(3)]],
    constant uint& nx [[buffer(4)]],
    constant uint& ny [[buffer(5)]],
    constant uint& log2ny [[buffer(6)]],
    constant uint& batch_idx [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])  // gid.x = col, gid.y = row
{
    uint col = gid.x;
    uint row = gid.y;

    if (col < nx && row < ny) {
        uint rev_row = reverseBits(row, log2ny);
        uint base = batch_idx * ny * nx;

        output_real[base + rev_row * nx + col] = input_real[base + row * nx + col];
        output_imag[base + rev_row * nx + col] = input_imag[base + row * nx + col];
    }
}

// ============================================================================
// 3D FFT butterfly kernel for z-dimension
// ============================================================================

kernel void metal_kernel_fft3d_z_butterfly(
    device XFLOAT* data_real [[buffer(0)]],
    device XFLOAT* data_imag [[buffer(1)]],
    constant uint& nx [[buffer(2)]],
    constant uint& ny [[buffer(3)]],
    constant uint& nz [[buffer(4)]],
    constant uint& stage [[buffer(5)]],
    constant int& direction [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])  // gid.x = col, gid.y = row, gid.z = butterfly
{
    uint col = gid.x;
    uint row = gid.y;
    uint butterfly_idx = gid.z;

    if (col >= nx || row >= ny) return;

    uint half_size = 1u << stage;
    uint full_size = half_size << 1;
    uint num_butterflies = nz / 2;

    if (butterfly_idx < num_butterflies) {
        uint group_idx = butterfly_idx / half_size;
        uint pair_idx = butterfly_idx % half_size;

        uint z_a = group_idx * full_size + pair_idx;
        uint z_b = z_a + half_size;
        uint idx_a = z_a * ny * nx + row * nx + col;
        uint idx_b = z_b * ny * nx + row * nx + col;

        XFLOAT angle = XFLOAT(direction) * (-2.0f) * M_PI_F * XFLOAT(pair_idx) / XFLOAT(full_size);

        Complex twiddle;
        twiddle.real = cos(angle);
        twiddle.imag = sin(angle);

        Complex a, b;
        a.real = data_real[idx_a];
        a.imag = data_imag[idx_a];
        b.real = data_real[idx_b];
        b.imag = data_imag[idx_b];

        Complex tb = complex_mul(twiddle, b);
        Complex out_a = complex_add(a, tb);
        Complex out_b = complex_sub(a, tb);

        data_real[idx_a] = out_a.real;
        data_imag[idx_a] = out_a.imag;
        data_real[idx_b] = out_b.real;
        data_imag[idx_b] = out_b.imag;
    }
}

// ============================================================================
// Copy kernel (for buffer management)
// ============================================================================

kernel void metal_kernel_fft_copy(
    device const XFLOAT* src_real [[buffer(0)]],
    device const XFLOAT* src_imag [[buffer(1)]],
    device XFLOAT* dst_real [[buffer(2)]],
    device XFLOAT* dst_imag [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        dst_real[gid] = src_real[gid];
        dst_imag[gid] = src_imag[gid];
    }
}

// ============================================================================
// Pack real data into complex format (even indices = real, odd indices = 0)
// ============================================================================

kernel void metal_kernel_fft_pack_real_to_complex(
    device const XFLOAT* real_data [[buffer(0)]],
    device XFLOAT* complex_real [[buffer(1)]],
    device XFLOAT* complex_imag [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& batch_idx [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        uint idx = batch_idx * n + gid;
        complex_real[idx] = real_data[idx];
        complex_imag[idx] = 0.0f;
    }
}

// ============================================================================
// Extract real part from complex data (for C2R output)
// ============================================================================

kernel void metal_kernel_fft_extract_real(
    device const XFLOAT* complex_real [[buffer(0)]],
    device XFLOAT* real_data [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& batch_idx [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        uint idx = batch_idx * n + gid;
        real_data[idx] = complex_real[idx];
    }
}
