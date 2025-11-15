#ifndef METAL_SETTINGS_H_
#define METAL_SETTINGS_H_

#include <signal.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "src/macros.h"
#include "src/error.h"

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>
#endif

// Required Metal version
#define METAL_VERSION_MAJOR 3
#define METAL_VERSION_MINOR 0

#define LAUNCH_CHECK
#define METAL_BENCHMARK_OLD true

// Error handling ----------------------

#ifdef LAUNCH_CHECK
#define LAUNCH_HANDLE_ERROR( err ) (LaunchHandleError( err, __FILE__, __LINE__ ))
#define LAUNCH_PRIVATE_ERROR(func, status) {  \
                       (status) = (func); \
                       LAUNCH_HANDLE_ERROR(status); \
                   }
#else
#define LAUNCH_HANDLE_ERROR( err ) (err) //Do nothing
#define LAUNCH_PRIVATE_ERROR( err ) (err) //Do nothing
#endif

#ifdef DEBUG_METAL
#define DEBUG_HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#define DEBUG_PRIVATE_ERROR(func, status) {  \
                       (status) = (func); \
                       DEBUG_HANDLE_ERROR(status); \
                   }
#else
#define DEBUG_HANDLE_ERROR( err ) (err) //Do nothing
#define DEBUG_PRIVATE_ERROR( err ) (err) //Do nothing
#endif

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#define PRIVATE_ERROR(func, status) {  \
                       (status) = (func); \
                       HANDLE_ERROR(status); \
                   }

#ifdef __cplusplus
extern "C" {
#endif

// Error handler for NSError pointers (Objective-C)
static inline void HandleNSError(void* err, const char *file, int line)
{
#ifdef __OBJC__
    NSError* error = (__bridge NSError*)err;
    if (error != nil)
    {
        fprintf(stderr, "METAL ERROR: %s in %s at line %d\n",
                        [error.localizedDescription UTF8String], file, line);
        fflush(stdout);
#ifdef DEBUG_METAL
        raise(SIGSEGV);
#else
        CRITICAL(ERRGPUKERN);
#endif
    }
#endif
}

// Generic error handler (for bool/int error codes)
static inline void HandleError(int err, const char *file, int line)
{
    if (err != 0)
    {
        fprintf(stderr, "METAL ERROR: Error code %d in %s at line %d\n",
                        err, file, line);
        fflush(stdout);
#ifdef DEBUG_METAL
        raise(SIGSEGV);
#else
        CRITICAL(ERRGPUKERN);
#endif
    }
}

#ifdef LAUNCH_CHECK
static inline void LaunchHandleError(int err, const char *file, int line)
{
    if (err != 0)
    {
        fprintf(stderr, "METAL KERNEL_ERROR: Error code %d in %s at line %d\n",
                        err, file, line);
        fflush(stdout);
        CRITICAL(ERRGPUKERN);
    }
}
#endif

#ifdef __cplusplus
}
#endif

// GENERAL -----------------------------
#define MAX_RESOL_SHARED_MEM        32
#define BLOCK_SIZE                  128
// -------------------------------------


// COARSE DIFF -------------------------
#define D2C_BLOCK_SIZE_2D           512
#define D2C_EULERS_PER_BLOCK_2D     4

#define D2C_BLOCK_SIZE_REF3D        128
#define D2C_EULERS_PER_BLOCK_REF3D  16

#define D2C_BLOCK_SIZE_DATA3D       64
#define D2C_EULERS_PER_BLOCK_DATA3D 32
// -------------------------------------


// FINE DIFF ---------------------------
#define D2F_BLOCK_SIZE_2D           256
#define D2F_CHUNK_2D                7

#define D2F_BLOCK_SIZE_REF3D        256
#define D2F_CHUNK_REF3D             7

#define D2F_BLOCK_SIZE_DATA3D       512
#define D2F_CHUNK_DATA3D            4
// -------------------------------------


// WAVG --------------------------------
#define WAVG_BLOCK_SIZE_DATA3D      512
#define WAVG_BLOCK_SIZE             256
// -------------------------------------


// MISC --------------------------------
#define SUMW_BLOCK_SIZE             32
#define SOFTMASK_BLOCK_SIZE         128
#define CFTT_BLOCK_SIZE             128
#define PROBRATIO_BLOCK_SIZE        128
#define POWERCLASS_BLOCK_SIZE       128
#define PROJDIFF_CHUNK_SIZE         14
// -------------------------------------

// RANDOMIZATION -----------------------
#define RND_BLOCK_NUM                   64
#define RND_BLOCK_SIZE                  32
// -------------------------------------


#define BACKPROJECTION4_BLOCK_SIZE 64
#define BACKPROJECTION4_GROUP_SIZE 16
#define BACKPROJECTION4_PREFETCH_COUNT 3
#define BP_2D_BLOCK_SIZE 128
#define BP_REF3D_BLOCK_SIZE 128
#define BP_DATA3D_BLOCK_SIZE 640


#define REF_GROUP_SIZE 3            // -- Number of references to be treated per block --
                                    // This applies to wavg and reduces global memory
                                    // accesses roughly proportionally, but scales shared
                                    // memory usage by allocating
                                    // ( 6*REF_GROUP_SIZE + 4 ) * BLOCK_SIZE XFLOATS. // DEPRECATED

#define NR_CLASS_MUTEXES 5

// Metal-specific settings
#define MAX_THREADGROUP_SIZE 1024
#define MAX_THREADS_PER_THREADGROUP 1024
#define MAX_BUFFER_SIZE (256 * 1024 * 1024)  // 256MB per buffer
#define MAX_TEXTURE_SIZE 16384

// The approximate minimum amount of memory each process occupies on a device (in MBs)
#define GPU_THREAD_MEMORY_OVERHEAD_MB 200

#endif /* METAL_SETTINGS_H_ */
