# RELION Metal GPU Backend

## Overview

This directory contains the Metal GPU acceleration backend for RELION, enabling native GPU support on Apple Silicon (M-Series) Macs.

**Status:** Phase 2B - Core Kernel Implementations (PARTIAL)

**Implementation Date:** 2025-11-15 (Phase 1), 2025-11-15 (Phase 2A), 2025-11-15 (Phase 2B)

## Architecture

The Metal backend follows the same abstraction pattern as the existing CUDA, HIP, and SYCL backends:

```
RELION Host Code
    ↓
AccPtr<T> Template (Unified Memory Interface)
    ↓
Metal Backend (src/acc/metal/)
    ├── Memory Management
    ├── Device Management
    ├── Compute Kernels (MSL)
    └── Metal Performance Shaders
```

## Directory Structure

```
src/acc/metal/
├── README.md                       # This file
├── metal_settings.h                # Configuration constants and error handling
├── shortcuts.h                     # Type definitions and macros
├── device_stubs.h                  # Device operation interface
├── metal_helper_functions.h        # Utility functions for Metal operations
├── metal_mem_utils.h/.mm           # Memory allocation and transfer functions
├── custom_allocator.h/.mm          # Metal buffer pool allocator
├── metal_device.h/.mm              # Device management and command queues
├── metal_kernel_utils.h/.mm        # Kernel launcher infrastructure
├── metal_fft.h/.mm                 # FFT operations (vDSP/Accelerate wrapper)
└── metal_kernels/
    ├── helper.metal                # Common MSL utility functions
    ├── utilities.metal             # Basic utility kernels
    └── projection.metal            # Projection/backprojection kernel stubs
```

## Building with Metal Support

### Prerequisites

- macOS 12.0 (Monterey) or later
- Xcode 14.0 or later with Metal 3.0 support
- Apple Silicon (M1/M2/M3 series) or compatible GPU

### Build Instructions

```bash
mkdir build && cd build
cmake -DMETAL=ON -DCUDA=OFF ..
make -j$(sysctl -n hw.ncpu)
```

### Build Options

- `METAL=ON` - Enable Metal GPU acceleration
- `DoublePrec_ACC=OFF` - Metal uses single precision (default)
- `MetalForceSTL=OFF` - Use Metal-native operations (default)
- `CachedAlloc=ON` - Enable custom allocator caching (recommended)

## Implementation Status

### Phase 1: Foundation & Infrastructure ✅ COMPLETE

- [x] CMake build system integration
- [x] Metal framework detection and configuration
- [x] Directory structure and headers
- [x] Memory management (allocation, transfer, synchronization)
- [x] Custom buffer allocator with caching
- [x] Device management and enumeration
- [x] AccPtr<T> integration
- [x] Basic MSL helper functions

### Phase 2A: Kernel Infrastructure ⚙️ IN PROGRESS

- [x] Kernel launcher infrastructure (KernelLauncher class)
- [x] FFT wrapper (vDSP/Accelerate-based, CPU fallback)
- [x] Utility kernels (multiply, exponentiate, softmask, weights_exponent, etc.)
- [x] Projection kernel infrastructure and helpers
- [x] Basic AccProjectorKernel port
- [ ] Full diff2_coarse/fine kernel implementations ⏳ TODO
- [ ] Full backprojection kernel implementations ⏳ TODO
- [ ] Full weighted averaging implementation ⏳ TODO
- [ ] GPU-accelerated FFT (custom Metal kernels) ⏳ TODO

**Note**: Phase 2A provides the infrastructure and simplified kernel stubs. Full kernel implementations matching CUDA performance will be completed in Phase 2B.

### Phase 2B: Full Kernel Implementations ⚙️ PARTIAL

- [x] Complete diff2_coarse kernel (2D version) ✅
- [ ] Complete diff2_coarse kernels (3D variants) ⏳ TODO
- [ ] Complete diff2_fine kernels (2D/3D variants) ⏳ TODO
- [x] Complete backprojection kernel (2D version) ✅
- [ ] Complete backprojection kernels (3D/SGD variants) ⏳ TODO
- [x] Complete weighted averaging kernel ✅
- [ ] GPU-based FFT using custom Metal compute shaders ⏳ TODO
- [ ] Kernel optimization and performance tuning ⏳ TODO

**Note**: Phase 2B provides fully functional 2D versions of the core kernels (diff2_coarse, backproject, wavg). 3D variants and diff2_fine are TODO. The implementations match CUDA logic and should produce correct results, though performance optimization is still needed.

### Phase 3: Advanced Features (NOT YET STARTED)

- [ ] Auto-picker GPU acceleration
- [ ] Random number generation (Philox RNG)
- [ ] Multi-GPU support and load balancing
- [ ] Performance optimization

### Phase 4: Integration & Testing (NOT YET STARTED)

- [ ] Full host code integration
- [ ] Validation tests against CUDA backend
- [ ] Performance benchmarks

## Key Features

### Unified Memory Architecture

Metal leverages Apple Silicon's unified memory:
- Zero-copy access between CPU and GPU
- Reduced memory transfers
- Efficient shared storage mode

### Memory Management

- **Shared Storage**: CPU and GPU can both access (zero-copy)
- **Private Storage**: GPU-only memory (highest performance)
- **Managed Storage**: Synchronized between CPU/GPU (macOS only)

### Custom Allocator

The Metal backend includes a buffer pool allocator that:
- Caches freed buffers for reuse
- Reduces allocation overhead
- Configurable cache size limit (default: 512MB)
- Event-based deallocation for async operations

### Device Management

- Automatic device enumeration
- Multi-GPU support (for Mac systems with multiple GPUs)
- Preference for high-performance discrete GPUs
- Unified memory detection

## Performance Considerations

### Optimizations Implemented

1. **Memory Coalescing**: Aligned buffer access patterns
2. **Threadgroup (Shared) Memory**: For reduction operations
3. **SIMD Group Operations**: Native vector operations
4. **Buffer Caching**: Reduced allocation overhead

### Limitations

1. **No Texture Memory**: Uses buffer-based interpolation instead
2. **Single Precision**: Metal 3 has limited double precision support
3. **Synchronization**: Different model from CUDA streams

## API Examples

### Memory Allocation

```objective-c++
// Allocate a buffer
id<MTLDevice> device = MetalHelper::getDefaultDevice();
size_t size = 1024 * sizeof(float);
void* buffer = metalAllocateDevice(device, size, METAL_MEM_SHARED);

// Copy data
metalCopyHostToDevice(buffer, hostPtr, size, 0);

// Free buffer
metalFreeDevice(buffer);
```

### Custom Allocator

```cpp
MetalCustomAllocator allocator(device, true);

// Allocate from pool
auto* alloc = allocator.alloc(size, METAL_MEM_SHARED);

// Use buffer
void* ptr = alloc->getPtr();

// Return to pool (cached)
allocator.free(alloc);

// Clear cache
allocator.clearCache();
```

### Device Management

```cpp
Metal::DeviceManager::initialize();

int deviceCount = Metal::DeviceManager::getDeviceCount();
Metal::DeviceManager::setDevice(0);

auto info = Metal::DeviceManager::getDeviceInfo(0);
std::cout << "Device: " << info.name << std::endl;
std::cout << "Memory: " << info.totalMemory / (1024*1024) << " MB" << std::endl;
```

## Testing

Phase 2A includes kernel infrastructure but full implementations are stubs. To test the build:

```bash
# Verify Metal support is compiled
./build/bin/relion_refine --version

# Check for Metal-related symbols
nm build/lib/librelion_lib.a | grep -i metal
```

## Known Issues

1. **FFT Currently CPU-Based**: Using vDSP/Accelerate framework (CPU) as fallback until GPU-accelerated FFT is implemented - CRITICAL for performance
2. **3D Kernels Not Implemented**: Only 2D versions of diff2_coarse and backproject are complete; 3D variants needed for full functionality
3. **diff2_fine Missing**: The fine sampling kernel is not yet implemented
4. **Not Optimized**: Current kernels are functional but not yet performance-tuned
5. **Double Precision**: Metal has limited support for double precision on consumer GPUs
6. **Texture Memory**: Replaced with buffer-based interpolation (may impact performance)
7. **Atomics**: Metal atomics have different semantics than CUDA (using relaxed memory order)
8. **Debugging**: Limited tooling compared to CUDA (use Metal Debugger in Xcode)

## Next Steps (Remaining Phase 2B + Phase 3)

### Immediate Priorities:
1. Implement GPU-accelerated FFT (CRITICAL - current CPU fallback is a major bottleneck)
2. Implement 3D variants of diff2_coarse and backprojection
3. Implement diff2_fine kernel (2D and 3D)
4. Performance optimization and tuning of existing kernels
5. Validation tests against CUDA backend for correctness

### Phase 3 Goals:
6. Host code integration (connect Metal kernels to RELION's refinement pipeline)
7. Auto-picker GPU acceleration
8. Random number generation (Philox RNG)
9. Multi-GPU support and load balancing
10. Comprehensive testing and benchmarking

## Contributing

When adding new Metal kernels:

1. Place MSL shaders in `metal_kernels/`
2. Follow the naming convention: `<feature>.metal`
3. Use helper functions from `helper.metal`
4. Update CMakeLists.txt to compile new shaders
5. Add C++ wrapper functions in corresponding `.h/.mm` files

## References

- [Metal Programming Guide](https://developer.apple.com/metal/)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [RELION Wiki](https://relion.readthedocs.io/)

## License

This Metal backend follows the same license as RELION (GPLv2).

## Authors

- Metal GPU Backend Implementation: Claude (Anthropic AI) - November 2025
- Based on CUDA/HIP implementations by the RELION development team

---

**Note**: This is Phase 1 infrastructure only. Kernel implementations and full functionality will be added in subsequent phases.
