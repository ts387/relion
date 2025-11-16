#include <metal_stdlib>
using namespace metal;

typedef float XFLOAT;

// ============================================================================
// Philox 4x32-10 Counter-Based Random Number Generator
// ============================================================================

// Philox constants (from original paper)
constant uint PHILOX_M4x32_0 = 0xD2511F53;
constant uint PHILOX_M4x32_1 = 0xCD9E8D57;
constant uint PHILOX_W32_0 = 0x9E3779B9;
constant uint PHILOX_W32_1 = 0xBB67AE85;

// Philox round function (single round)
inline void philox4x32_round(thread uint4& counter, thread uint2& key) {
    uint lo0 = counter.x * PHILOX_M4x32_0;
    uint hi0 = mulhi(counter.x, PHILOX_M4x32_0);
    uint lo1 = counter.z * PHILOX_M4x32_1;
    uint hi1 = mulhi(counter.z, PHILOX_M4x32_1);

    counter = uint4(hi1 ^ counter.y ^ key.x, lo1, hi0 ^ counter.w ^ key.y, lo0);
}

// Full Philox 4x32-10 (10 rounds)
inline uint4 philox4x32_10(uint4 counter, uint2 key) {
    for (int i = 0; i < 10; i++) {
        philox4x32_round(counter, key);
        key.x += PHILOX_W32_0;
        key.y += PHILOX_W32_1;
    }
    return counter;
}

// Convert uint to float in [0, 1)
inline float uint_to_float01(uint x) {
    // Use only upper 23 bits for mantissa (better quality)
    return float(x >> 9) * (1.0f / 8388608.0f); // 2^23
}

// Convert two uniforms to two Gaussians (Box-Muller transform)
inline void box_muller(float u1, float u2, thread float& g1, thread float& g2) {
    float r = sqrt(-2.0f * log(u1));
    float theta = 2.0f * M_PI_F * u2;
    g1 = r * cos(theta);
    g2 = r * sin(theta);
}

// ============================================================================
// RNG State Structure (for stateful generation)
// ============================================================================

struct PhiloxState {
    uint4 counter;
    uint2 key;
};

// ============================================================================
// Initialize RNG States
// ============================================================================

kernel void metal_kernel_initRNG(
    device PhiloxState* states [[buffer(0)]],
    constant uint& num_states [[buffer(1)]],
    constant uint& seed_lo [[buffer(2)]],
    constant uint& seed_hi [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < num_states) {
        // Each thread gets unique counter based on thread ID
        states[gid].counter = uint4(gid, 0, 0, 0);
        // Seed forms the key
        states[gid].key = uint2(seed_lo, seed_hi);
    }
}

// ============================================================================
// Generate Uniform Random Numbers
// ============================================================================

kernel void metal_kernel_generateUniform(
    device PhiloxState* states [[buffer(0)]],
    device XFLOAT* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        PhiloxState state = states[gid % 2048]; // Wrap around state pool

        // Increment counter for this sample
        state.counter.y = gid;

        uint4 result = philox4x32_10(state.counter, state.key);
        output[gid] = uint_to_float01(result.x);
    }
}

// ============================================================================
// Generate Gaussian Random Numbers (Box-Muller)
// ============================================================================

kernel void metal_kernel_generateGaussian(
    device PhiloxState* states [[buffer(0)]],
    device XFLOAT* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        PhiloxState state = states[gid % 2048];
        state.counter.y = gid / 2; // Pairs of samples

        uint4 result = philox4x32_10(state.counter, state.key);

        float u1 = uint_to_float01(result.x);
        float u2 = uint_to_float01(result.y);

        // Avoid log(0)
        u1 = max(u1, 1e-10f);

        float g1, g2;
        box_muller(u1, u2, g1, g2);

        // Output either first or second Gaussian
        output[gid] = (gid % 2 == 0) ? g1 : g2;
    }
}

// ============================================================================
// 2D Complex Gaussian Noise with Power Modulation
// ============================================================================

kernel void metal_kernel_RNDnormalDistribution2D(
    device PhiloxState* states [[buffer(0)]],
    device XFLOAT* g_out_real [[buffer(1)]],
    device XFLOAT* g_out_imag [[buffer(2)]],
    device const XFLOAT* g_spectra [[buffer(3)]],
    constant uint& xdim [[buffer(4)]],
    constant uint& ydim [[buffer(5)]],
    constant uint& seed_offset [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint x = gid.x;
    uint y = gid.y;

    if (x < xdim && y < ydim) {
        uint idx = y * xdim + x;

        // Get spectrum value for this frequency
        uint ires = min(x, xdim - 1);
        XFLOAT spectrum_val = g_spectra[ires];

        // Generate Philox counter for this pixel
        PhiloxState state = states[idx % 2048];
        state.counter.y = seed_offset;
        state.counter.z = idx;

        uint4 result = philox4x32_10(state.counter, state.key);

        float u1 = max(uint_to_float01(result.x), 1e-10f);
        float u2 = uint_to_float01(result.y);

        float g1, g2;
        box_muller(u1, u2, g1, g2);

        // Apply power spectrum modulation
        g_out_real[idx] = g1 * spectrum_val;
        g_out_imag[idx] = g2 * spectrum_val;
    }
}

// ============================================================================
// 3D Complex Gaussian Noise with Power Modulation
// ============================================================================

kernel void metal_kernel_RNDnormalDistribution3D(
    device PhiloxState* states [[buffer(0)]],
    device XFLOAT* g_out_real [[buffer(1)]],
    device XFLOAT* g_out_imag [[buffer(2)]],
    device const XFLOAT* g_spectra [[buffer(3)]],
    constant uint& xdim [[buffer(4)]],
    constant uint& ydim [[buffer(5)]],
    constant uint& zdim [[buffer(6)]],
    constant uint& seed_offset [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint x = gid.x;
    uint y = gid.y;
    uint z = gid.z;

    if (x < xdim && y < ydim && z < zdim) {
        uint idx = z * ydim * xdim + y * xdim + x;

        // Get spectrum value for this frequency
        uint ires = min(x, xdim - 1);
        XFLOAT spectrum_val = g_spectra[ires];

        // Generate Philox counter for this voxel
        PhiloxState state = states[idx % 2048];
        state.counter.y = seed_offset;
        state.counter.z = idx;

        uint4 result = philox4x32_10(state.counter, state.key);

        float u1 = max(uint_to_float01(result.x), 1e-10f);
        float u2 = uint_to_float01(result.y);

        float g1, g2;
        box_muller(u1, u2, g1, g2);

        // Apply power spectrum modulation
        g_out_real[idx] = g1 * spectrum_val;
        g_out_imag[idx] = g2 * spectrum_val;
    }
}

// ============================================================================
// Generate Random Weights for SGD
// ============================================================================

kernel void metal_kernel_generateRandomWeights(
    device PhiloxState* states [[buffer(0)]],
    device XFLOAT* weights [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant XFLOAT& min_val [[buffer(3)]],
    constant XFLOAT& max_val [[buffer(4)]],
    constant uint& seed_offset [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        PhiloxState state = states[gid % 2048];
        state.counter.y = seed_offset;
        state.counter.z = gid;

        uint4 result = philox4x32_10(state.counter, state.key);
        float u = uint_to_float01(result.x);

        // Scale to [min_val, max_val]
        weights[gid] = min_val + u * (max_val - min_val);
    }
}

// ============================================================================
// Random Particle Selection for Mini-Batch SGD
// ============================================================================

kernel void metal_kernel_selectRandomParticles(
    device PhiloxState* states [[buffer(0)]],
    device uint* selected_indices [[buffer(1)]],
    constant uint& num_particles [[buffer(2)]],
    constant uint& num_to_select [[buffer(3)]],
    constant uint& seed_offset [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < num_to_select) {
        PhiloxState state = states[gid % 2048];
        state.counter.y = seed_offset;
        state.counter.z = gid;

        uint4 result = philox4x32_10(state.counter, state.key);

        // Map to particle index
        uint particle_idx = result.x % num_particles;
        selected_indices[gid] = particle_idx;
    }
}

// ============================================================================
// Add Gaussian Noise to Data
// ============================================================================

kernel void metal_kernel_addGaussianNoise(
    device PhiloxState* states [[buffer(0)]],
    device XFLOAT* data [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    constant XFLOAT& sigma [[buffer(3)]],
    constant uint& seed_offset [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        PhiloxState state = states[gid % 2048];
        state.counter.y = seed_offset;
        state.counter.z = gid;

        uint4 result = philox4x32_10(state.counter, state.key);

        float u1 = max(uint_to_float01(result.x), 1e-10f);
        float u2 = uint_to_float01(result.y);

        float g1, g2;
        box_muller(u1, u2, g1, g2);

        // Add noise
        data[gid] += g1 * sigma;
    }
}

// ============================================================================
// Stateless Random Generation (for single-shot operations)
// ============================================================================

inline uint4 philox_stateless(uint4 counter, uint seed_lo, uint seed_hi) {
    uint2 key = uint2(seed_lo, seed_hi);
    return philox4x32_10(counter, key);
}

kernel void metal_kernel_generateUniform_stateless(
    device XFLOAT* output [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    constant uint& seed_lo [[buffer(2)]],
    constant uint& seed_hi [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        uint4 counter = uint4(gid, 0, 0, 0);
        uint4 result = philox_stateless(counter, seed_lo, seed_hi);
        output[gid] = uint_to_float01(result.x);
    }
}

kernel void metal_kernel_generateGaussian_stateless(
    device XFLOAT* output [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    constant uint& seed_lo [[buffer(2)]],
    constant uint& seed_hi [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        uint4 counter = uint4(gid / 2, gid, 0, 0);
        uint4 result = philox_stateless(counter, seed_lo, seed_hi);

        float u1 = max(uint_to_float01(result.x), 1e-10f);
        float u2 = uint_to_float01(result.y);

        float g1, g2;
        box_muller(u1, u2, g1, g2);

        output[gid] = (gid % 2 == 0) ? g1 : g2;
    }
}
