#include <metal_stdlib>
using namespace metal;

// Common type definitions
typedef float XFLOAT;  // Change to double if ACC_DOUBLE_PRECISION is set
typedef float2 XFLOAT2;
typedef float3 XFLOAT3;
typedef float4 XFLOAT4;

// Complex number structure
struct ComplexXFLOAT {
    XFLOAT real;
    XFLOAT imag;
};

// ============================================================================
// Atomic operations
// ============================================================================

// Atomic add for float (Metal 3.0+ supports atomic operations on device memory)
inline void atomicAdd(device atomic<float>* addr, float value) {
    atomic_fetch_add_explicit(addr, value, memory_order_relaxed);
}

// Atomic add for threadgroup memory
inline void atomicAdd(threadgroup atomic<float>* addr, float value) {
    atomic_fetch_add_explicit(addr, value, memory_order_relaxed);
}

// ============================================================================
// Thread group (SIMD) operations
// ============================================================================

// SIMD shuffle down (equivalent to CUDA's __shfl_down_sync)
template<typename T>
inline T simdShuffleDown(T value, uint16_t delta) {
    return simd_shuffle_down(value, delta);
}

// SIMD reduction (sum across SIMD group)
inline float simdSum(float value) {
    return simd_sum(value);
}

// SIMD reduction (max across SIMD group)
inline float simdMax(float value) {
    return simd_max(value);
}

// ============================================================================
// Threadgroup memory reduction patterns
// ============================================================================

// Reduce values within a threadgroup using shared memory
template<typename T>
inline T threadgroupReduceAdd(threadgroup T* shared_data,
                                uint thread_index,
                                uint threadgroup_size,
                                T value) {
    // Store value in shared memory
    shared_data[thread_index] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce using tree-based reduction
    for (uint stride = threadgroup_size / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            shared_data[thread_index] += shared_data[thread_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    return shared_data[0];
}

// ============================================================================
// Mathematical helper functions
// ============================================================================

// Complex multiplication
inline ComplexXFLOAT complexMul(ComplexXFLOAT a, ComplexXFLOAT b) {
    ComplexXFLOAT result;
    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;
    return result;
}

// Complex conjugate
inline ComplexXFLOAT complexConj(ComplexXFLOAT a) {
    ComplexXFLOAT result;
    result.real = a.real;
    result.imag = -a.imag;
    return result;
}

// Complex magnitude squared
inline XFLOAT complexNorm2(ComplexXFLOAT a) {
    return a.real * a.real + a.imag * a.imag;
}

// Fast inverse square root (optional optimization)
inline float fastInvSqrt(float x) {
    return rsqrt(x);  // Metal built-in reciprocal square root
}

// ============================================================================
// Coordinate transformations
// ============================================================================

// Convert 1D index to 3D coordinates
inline uint3 index1Dto3D(uint idx, uint width, uint height, uint depth) {
    uint z = idx / (width * height);
    uint remainder = idx % (width * height);
    uint y = remainder / width;
    uint x = remainder % width;
    return uint3(x, y, z);
}

// Convert 3D coordinates to 1D index
inline uint index3Dto1D(uint3 coords, uint width, uint height, uint depth) {
    return coords.z * width * height + coords.y * width + coords.x;
}

// ============================================================================
// Interpolation functions (for texture replacement)
// ============================================================================

// Linear interpolation
inline XFLOAT lerp(XFLOAT a, XFLOAT b, XFLOAT t) {
    return a + t * (b - a);
}

// Trilinear interpolation from a 3D buffer
inline XFLOAT trilinearInterpolate(device const XFLOAT* data,
                                   float3 coords,
                                   uint3 dimensions) {
    // Get integer coordinates
    uint3 c0 = uint3(floor(coords));
    uint3 c1 = min(c0 + 1, dimensions - 1);

    // Get fractional parts
    float3 frac = coords - float3(c0);

    // Sample the 8 corners
    uint width = dimensions.x;
    uint height = dimensions.y;
    uint depth = dimensions.z;

    XFLOAT v000 = data[index3Dto1D(uint3(c0.x, c0.y, c0.z), width, height, depth)];
    XFLOAT v001 = data[index3Dto1D(uint3(c0.x, c0.y, c1.z), width, height, depth)];
    XFLOAT v010 = data[index3Dto1D(uint3(c0.x, c1.y, c0.z), width, height, depth)];
    XFLOAT v011 = data[index3Dto1D(uint3(c0.x, c1.y, c1.z), width, height, depth)];
    XFLOAT v100 = data[index3Dto1D(uint3(c1.x, c0.y, c0.z), width, height, depth)];
    XFLOAT v101 = data[index3Dto1D(uint3(c1.x, c0.y, c1.z), width, height, depth)];
    XFLOAT v110 = data[index3Dto1D(uint3(c1.x, c1.y, c0.z), width, height, depth)];
    XFLOAT v111 = data[index3Dto1D(uint3(c1.x, c1.y, c1.z), width, height, depth)];

    // Trilinear interpolation
    XFLOAT v00 = lerp(v000, v100, frac.x);
    XFLOAT v01 = lerp(v001, v101, frac.x);
    XFLOAT v10 = lerp(v010, v110, frac.x);
    XFLOAT v11 = lerp(v011, v111, frac.x);

    XFLOAT v0 = lerp(v00, v10, frac.y);
    XFLOAT v1 = lerp(v01, v11, frac.y);

    return lerp(v0, v1, frac.z);
}

// ============================================================================
// Simple test kernel
// ============================================================================

kernel void metal_test_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        output[gid] = input[gid] * 2.0f;
    }
}
