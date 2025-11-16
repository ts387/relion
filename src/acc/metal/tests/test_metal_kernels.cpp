/*
 * Metal Kernel Validation Test Suite
 *
 * This file provides validation tests for Metal GPU kernels to ensure
 * correctness against reference CPU implementations and consistency
 * with CUDA backend behavior.
 */

#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>

#ifdef _METAL_ENABLED
#include "src/acc/metal/metal_kernels.h"
#include "src/acc/metal/metal_device.h"
#include "src/acc/metal/metal_mem_utils.h"
#include "src/acc/acc_projectorkernel_impl.h"
#endif

// Test configuration
const double TOLERANCE_FP32 = 1e-5;
const double TOLERANCE_RELATIVE = 1e-4;
const int NUM_TRIALS = 3;

// ============================================================================
// Test Result Tracking
// ============================================================================

struct TestResult {
    std::string name;
    bool passed;
    double max_error;
    double avg_error;
    double elapsed_ms;
};

std::vector<TestResult> g_test_results;

void reportResult(const std::string& name, bool passed, double max_err = 0.0, double avg_err = 0.0, double time_ms = 0.0) {
    TestResult result{name, passed, max_err, avg_err, time_ms};
    g_test_results.push_back(result);

    std::cout << (passed ? "[PASS]" : "[FAIL]") << " " << name;
    if (!passed || max_err > 0) {
        std::cout << " (max_err=" << std::scientific << std::setprecision(2) << max_err
                  << ", avg_err=" << avg_err << ")";
    }
    if (time_ms > 0) {
        std::cout << " [" << std::fixed << std::setprecision(2) << time_ms << " ms]";
    }
    std::cout << std::endl;
}

// ============================================================================
// Helper Functions
// ============================================================================

template<typename T>
double computeMaxError(const T* ref, const T* test, size_t size) {
    double max_err = 0.0;
    for (size_t i = 0; i < size; i++) {
        double err = std::abs((double)ref[i] - (double)test[i]);
        if (std::abs((double)ref[i]) > 1e-10) {
            err /= std::abs((double)ref[i]); // relative error
        }
        max_err = std::max(max_err, err);
    }
    return max_err;
}

template<typename T>
double computeAvgError(const T* ref, const T* test, size_t size) {
    double sum_err = 0.0;
    for (size_t i = 0; i < size; i++) {
        double err = std::abs((double)ref[i] - (double)test[i]);
        if (std::abs((double)ref[i]) > 1e-10) {
            err /= std::abs((double)ref[i]);
        }
        sum_err += err;
    }
    return sum_err / size;
}

// Box-Muller CPU reference
void cpuBoxMuller(float u1, float u2, float& g1, float& g2) {
    float r = sqrt(-2.0f * log(u1));
    float theta = 2.0f * M_PI * u2;
    g1 = r * cos(theta);
    g2 = r * sin(theta);
}

// ============================================================================
// Test Cases
// ============================================================================

#ifdef _METAL_ENABLED

// Test 1: Exponentiation Kernel
bool testExponentiate() {
    const size_t SIZE = 10000;
    std::vector<float> h_input(SIZE);
    std::vector<float> h_expected(SIZE);
    std::vector<float> h_result(SIZE);

    // Initialize with values that won't overflow
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    float add_val = -5.0f;

    for (size_t i = 0; i < SIZE; i++) {
        h_input[i] = dist(rng);
        float a = h_input[i] + add_val;
        if (a < -88.0f) {
            h_expected[i] = 0.0f;
        } else {
            h_expected[i] = exp(a);
        }
    }

    // Allocate GPU memory
    float* d_array = (float*)metalAllocateDevice(nullptr, SIZE * sizeof(float), METAL_MEM_SHARED);
    metalCopyHostToDevice(d_array, h_input.data(), SIZE * sizeof(float), 0);

    // Execute kernel
    auto start = std::chrono::high_resolution_clock::now();
    MetalKernels::exponentiate(d_array, add_val, SIZE, 0);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    // Copy back
    metalCopyDeviceToHost(h_result.data(), d_array, SIZE * sizeof(float), 0);

    // Validate
    double max_err = computeMaxError(h_expected.data(), h_result.data(), SIZE);
    double avg_err = computeAvgError(h_expected.data(), h_result.data(), SIZE);
    bool passed = max_err < TOLERANCE_RELATIVE;

    metalFreeDevice(d_array);

    reportResult("Exponentiate", passed, max_err, avg_err, elapsed);
    return passed;
}

// Test 2: Element-wise Multiplication
bool testMultiply() {
    const size_t SIZE = 10000;
    std::vector<float> h_A(SIZE), h_B(SIZE), h_expected(SIZE), h_result(SIZE);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (size_t i = 0; i < SIZE; i++) {
        h_A[i] = dist(rng);
        h_B[i] = dist(rng);
        h_expected[i] = h_A[i] * h_B[i];
    }

    float* d_A = (float*)metalAllocateDevice(nullptr, SIZE * sizeof(float), METAL_MEM_SHARED);
    float* d_B = (float*)metalAllocateDevice(nullptr, SIZE * sizeof(float), METAL_MEM_SHARED);
    float* d_OUT = (float*)metalAllocateDevice(nullptr, SIZE * sizeof(float), METAL_MEM_SHARED);

    metalCopyHostToDevice(d_A, h_A.data(), SIZE * sizeof(float), 0);
    metalCopyHostToDevice(d_B, h_B.data(), SIZE * sizeof(float), 0);

    auto start = std::chrono::high_resolution_clock::now();
    MetalKernels::multiply(d_A, d_B, d_OUT, SIZE, 0);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    metalCopyDeviceToHost(h_result.data(), d_OUT, SIZE * sizeof(float), 0);

    double max_err = computeMaxError(h_expected.data(), h_result.data(), SIZE);
    double avg_err = computeAvgError(h_expected.data(), h_result.data(), SIZE);
    bool passed = max_err < TOLERANCE_FP32;

    metalFreeDevice(d_A);
    metalFreeDevice(d_B);
    metalFreeDevice(d_OUT);

    reportResult("Multiply", passed, max_err, avg_err, elapsed);
    return passed;
}

// Test 3: RNG Initialization
bool testRNGInit() {
    const size_t NUM_STATES = 2048;
    const unsigned long long SEED = 0x12345678ABCDEF00ULL;

    // Allocate RNG states (PhiloxState is 24 bytes: uint4 + uint2 = 16 + 8)
    void* d_states = metalAllocateDevice(nullptr, NUM_STATES * 24, METAL_MEM_SHARED);

    auto start = std::chrono::high_resolution_clock::now();
    MetalKernels::initRNG(d_states, SEED, NUM_STATES, 0);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    // Verify states were initialized (check seed values)
    uint32_t* states_ptr = (uint32_t*)d_states;

    bool passed = true;
    uint32_t seed_lo = (uint32_t)(SEED & 0xFFFFFFFF);
    uint32_t seed_hi = (uint32_t)(SEED >> 32);

    // Check first few states
    for (size_t i = 0; i < 10; i++) {
        // PhiloxState: counter.x, counter.y, counter.z, counter.w, key.x, key.y
        uint32_t* state = states_ptr + i * 6;
        if (state[0] != i) { // counter.x should be thread ID
            passed = false;
            break;
        }
        if (state[4] != seed_lo || state[5] != seed_hi) { // key should be seed
            passed = false;
            break;
        }
    }

    metalFreeDevice(d_states);

    reportResult("RNG Initialization", passed, 0.0, 0.0, elapsed);
    return passed;
}

// Test 4: Gaussian Distribution Statistics
bool testGaussianDistribution() {
    const size_t SIZE = 100000;
    const unsigned long long SEED = 0xDEADBEEFCAFE0000ULL;

    void* d_states = metalAllocateDevice(nullptr, 2048 * 24, METAL_MEM_SHARED);
    float* d_output = (float*)metalAllocateDevice(nullptr, SIZE * sizeof(float), METAL_MEM_SHARED);

    MetalKernels::initRNG(d_states, SEED, 2048, 0);

    // Generate through 2D distribution with unit spectrum
    const size_t XDIM = 500;
    const size_t YDIM = 200; // 500 * 200 = 100000

    float* d_spectra = (float*)metalAllocateDevice(nullptr, XDIM * sizeof(float), METAL_MEM_SHARED);
    std::vector<float> h_spectra(XDIM, 1.0f); // unit spectrum
    metalCopyHostToDevice(d_spectra, h_spectra.data(), XDIM * sizeof(float), 0);

    float* d_out_real = (float*)metalAllocateDevice(nullptr, SIZE * sizeof(float), METAL_MEM_SHARED);
    float* d_out_imag = (float*)metalAllocateDevice(nullptr, SIZE * sizeof(float), METAL_MEM_SHARED);

    auto start = std::chrono::high_resolution_clock::now();
    MetalKernels::generateNormalDistribution2D(d_states, d_out_real, d_out_imag, d_spectra, XDIM, YDIM, 0);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    // Copy back and compute statistics
    std::vector<float> h_real(SIZE), h_imag(SIZE);
    metalCopyDeviceToHost(h_real.data(), d_out_real, SIZE * sizeof(float), 0);
    metalCopyDeviceToHost(h_imag.data(), d_out_imag, SIZE * sizeof(float), 0);

    // Compute mean and variance
    double mean_real = 0.0, mean_imag = 0.0;
    for (size_t i = 0; i < SIZE; i++) {
        mean_real += h_real[i];
        mean_imag += h_imag[i];
    }
    mean_real /= SIZE;
    mean_imag /= SIZE;

    double var_real = 0.0, var_imag = 0.0;
    for (size_t i = 0; i < SIZE; i++) {
        var_real += (h_real[i] - mean_real) * (h_real[i] - mean_real);
        var_imag += (h_imag[i] - mean_imag) * (h_imag[i] - mean_imag);
    }
    var_real /= SIZE;
    var_imag /= SIZE;

    // Gaussian should have mean ~0 and variance ~1
    bool passed = std::abs(mean_real) < 0.02 && std::abs(mean_imag) < 0.02 &&
                  std::abs(var_real - 1.0) < 0.05 && std::abs(var_imag - 1.0) < 0.05;

    double max_err = std::max(std::abs(mean_real), std::max(std::abs(mean_imag),
                     std::max(std::abs(var_real - 1.0), std::abs(var_imag - 1.0))));

    metalFreeDevice(d_states);
    metalFreeDevice(d_out_real);
    metalFreeDevice(d_out_imag);
    metalFreeDevice(d_spectra);

    reportResult("Gaussian Distribution", passed, max_err, 0.0, elapsed);

    if (!passed) {
        std::cout << "  Details: mean_real=" << mean_real << ", mean_imag=" << mean_imag
                  << ", var_real=" << var_real << ", var_imag=" << var_imag << std::endl;
    }

    return passed;
}

// Test 5: Soft Mask Outside Map
bool testSoftMaskOutsideMap() {
    const long int XDIM = 64, YDIM = 64, ZDIM = 64;
    const size_t SIZE = XDIM * YDIM * ZDIM;
    std::vector<float> h_vol(SIZE, 1.0f);
    std::vector<float> h_expected(SIZE);
    std::vector<float> h_result(SIZE);

    float radius = 20.0f;
    float cosine_width = 5.0f;
    long int xinit = XDIM / 2, yinit = YDIM / 2, zinit = ZDIM / 2;

    // CPU reference
    for (long int z = 0; z < ZDIM; z++) {
        for (long int y = 0; y < YDIM; y++) {
            for (long int x = 0; x < XDIM; x++) {
                int xp = x - xinit;
                int yp = y - yinit;
                int zp = z - zinit;
                float r = sqrt((float)(xp * xp + yp * yp + zp * zp));

                size_t idx = z * YDIM * XDIM + y * XDIM + x;
                if (r > radius) {
                    h_expected[idx] = 0.0f;
                } else if (r > radius - cosine_width) {
                    float diff = r - (radius - cosine_width);
                    float mask_value = 0.5f + 0.5f * cos(M_PI * diff / cosine_width);
                    h_expected[idx] = 1.0f * mask_value;
                } else {
                    h_expected[idx] = 1.0f;
                }
            }
        }
    }

    float* d_vol = (float*)metalAllocateDevice(nullptr, SIZE * sizeof(float), METAL_MEM_SHARED);
    metalCopyHostToDevice(d_vol, h_vol.data(), SIZE * sizeof(float), 0);

    auto start = std::chrono::high_resolution_clock::now();
    MetalKernels::softMaskOutsideMap(d_vol, XDIM, YDIM, ZDIM, xinit, yinit, zinit,
                                      false, nullptr, radius, radius, cosine_width, 0);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    metalCopyDeviceToHost(h_result.data(), d_vol, SIZE * sizeof(float), 0);

    double max_err = computeMaxError(h_expected.data(), h_result.data(), SIZE);
    double avg_err = computeAvgError(h_expected.data(), h_result.data(), SIZE);
    bool passed = max_err < TOLERANCE_FP32;

    metalFreeDevice(d_vol);

    reportResult("SoftMaskOutsideMap", passed, max_err, avg_err, elapsed);
    return passed;
}

// Test 6: CTF Multiplication
bool testMultiplyCTFs() {
    const size_t SIZE = 10000;
    std::vector<float> h_real(SIZE), h_imag(SIZE), h_ctf(SIZE);
    std::vector<float> h_expected_real(SIZE), h_expected_imag(SIZE);
    std::vector<float> h_result_real(SIZE), h_result_imag(SIZE);

    std::mt19937 rng(789);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < SIZE; i++) {
        h_real[i] = dist(rng);
        h_imag[i] = dist(rng);
        h_ctf[i] = dist(rng);
        h_expected_real[i] = h_real[i] * h_ctf[i];
        h_expected_imag[i] = h_imag[i] * h_ctf[i];
    }

    float* d_real = (float*)metalAllocateDevice(nullptr, SIZE * sizeof(float), METAL_MEM_SHARED);
    float* d_imag = (float*)metalAllocateDevice(nullptr, SIZE * sizeof(float), METAL_MEM_SHARED);
    float* d_ctf = (float*)metalAllocateDevice(nullptr, SIZE * sizeof(float), METAL_MEM_SHARED);

    metalCopyHostToDevice(d_real, h_real.data(), SIZE * sizeof(float), 0);
    metalCopyHostToDevice(d_imag, h_imag.data(), SIZE * sizeof(float), 0);
    metalCopyHostToDevice(d_ctf, h_ctf.data(), SIZE * sizeof(float), 0);

    auto start = std::chrono::high_resolution_clock::now();
    MetalKernels::multiplyCTFs(d_real, d_imag, d_ctf, false, nullptr, SIZE, 0);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    metalCopyDeviceToHost(h_result_real.data(), d_real, SIZE * sizeof(float), 0);
    metalCopyDeviceToHost(h_result_imag.data(), d_imag, SIZE * sizeof(float), 0);

    double max_err_real = computeMaxError(h_expected_real.data(), h_result_real.data(), SIZE);
    double max_err_imag = computeMaxError(h_expected_imag.data(), h_result_imag.data(), SIZE);
    double max_err = std::max(max_err_real, max_err_imag);
    bool passed = max_err < TOLERANCE_FP32;

    metalFreeDevice(d_real);
    metalFreeDevice(d_imag);
    metalFreeDevice(d_ctf);

    reportResult("MultiplyCTFs", passed, max_err, 0.0, elapsed);
    return passed;
}

#endif // _METAL_ENABLED

// ============================================================================
// Main Test Runner
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "=====================================================" << std::endl;
    std::cout << "Metal Kernel Validation Test Suite" << std::endl;
    std::cout << "=====================================================" << std::endl;

#ifndef _METAL_ENABLED
    std::cout << "ERROR: Metal support not enabled. Rebuild with -DMETAL=ON" << std::endl;
    return 1;
#else
    // Initialize Metal device
    try {
        Metal::DeviceManager::initialize();
        std::cout << "Metal Device: " << Metal::DeviceManager::getDeviceInfo(0).name << std::endl;
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cout << "ERROR: Failed to initialize Metal: " << e.what() << std::endl;
        return 1;
    }

    // Run tests
    std::cout << "Running validation tests..." << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;

    int total_tests = 0;
    int passed_tests = 0;

    // Basic kernels
    if (testExponentiate()) passed_tests++;
    total_tests++;

    if (testMultiply()) passed_tests++;
    total_tests++;

    if (testSoftMaskOutsideMap()) passed_tests++;
    total_tests++;

    if (testMultiplyCTFs()) passed_tests++;
    total_tests++;

    // RNG tests
    if (testRNGInit()) passed_tests++;
    total_tests++;

    if (testGaussianDistribution()) passed_tests++;
    total_tests++;

    // Summary
    std::cout << "-----------------------------------------------------" << std::endl;
    std::cout << "SUMMARY: " << passed_tests << "/" << total_tests << " tests passed" << std::endl;

    if (passed_tests == total_tests) {
        std::cout << "All tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests FAILED!" << std::endl;
        return 1;
    }
#endif
}
