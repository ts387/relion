#ifndef METAL_KERNELS_H_
#define METAL_KERNELS_H_

#include "src/acc/metal/metal_settings.h"
#include "src/acc/acc_projectorkernel_impl.h"

#ifdef _METAL_ENABLED

// Forward declarations for Metal kernel execution
namespace MetalKernels {

// ============================================================================
// Projection Difference Kernels
// ============================================================================

// diff2_coarse - Multiple orientations per block
// Template parameters match SYCL/CUDA pattern for dispatcher compatibility
template<bool REF3D, bool DATA3D, int block_sz, int eulers_per_block, int prefetch_fraction>
inline void diff2_coarse(
    unsigned long grid_size,
    XFLOAT *g_eulers,
    XFLOAT *trans_x,
    XFLOAT *trans_y,
    XFLOAT *trans_z,
    XFLOAT *g_real,
    XFLOAT *g_imag,
    AccProjectorKernel projector,
    XFLOAT *g_corr,
    XFLOAT *g_diff2s,
    unsigned long translation_num,
    unsigned long image_size,
    deviceStream_t stream);

// diff2_fine - Single orientation per block with dynamic scheduling
template<bool REF3D, bool DATA3D, int block_sz, int chunk_sz>
inline void diff2_fine(
    unsigned long grid_size,
    XFLOAT *g_eulers,
    XFLOAT *g_imgs_real,
    XFLOAT *g_imgs_imag,
    XFLOAT *trans_x,
    XFLOAT *trans_y,
    XFLOAT *trans_z,
    AccProjectorKernel projector,
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
    deviceStream_t stream);

// ============================================================================
// Backprojection Kernels
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
    deviceStream_t stream);

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
    deviceStream_t stream);

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
    deviceStream_t stream);

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
    deviceStream_t stream);

// ============================================================================
// Weighted Averaging Kernels
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
    deviceStream_t stream);

// ============================================================================
// Utility Kernels
// ============================================================================

void exponentiate(
    XFLOAT *g_array,
    XFLOAT add,
    unsigned long size,
    deviceStream_t stream);

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
    deviceStream_t stream);

void multiply(
    XFLOAT *A,
    XFLOAT *B,
    XFLOAT *OUT,
    unsigned long size,
    deviceStream_t stream);

void multiplyCTFs(
    XFLOAT *g_Fref_real,
    XFLOAT *g_Fref_imag,
    XFLOAT *g_ctf,
    bool do_scale_correction,
    XFLOAT *g_scale_correction,
    unsigned long image_size,
    deviceStream_t stream);

void applyWeights(
    XFLOAT *g_diff2s,
    XFLOAT *g_weights,
    XFLOAT weight,
    unsigned long size,
    deviceStream_t stream);

// ============================================================================
// Random Number Generation (Philox)
// ============================================================================

void initRNG(
    void *rng_states,
    unsigned long long seed,
    unsigned long size,
    deviceStream_t stream);

void generateNormalDistribution2D(
    void *rng_states,
    XFLOAT *g_out_real,
    XFLOAT *g_out_imag,
    XFLOAT *g_spectra,
    unsigned long xdim,
    unsigned long ydim,
    deviceStream_t stream);

void generateNormalDistribution3D(
    void *rng_states,
    XFLOAT *g_out_real,
    XFLOAT *g_out_imag,
    XFLOAT *g_spectra,
    unsigned long xdim,
    unsigned long ydim,
    unsigned long zdim,
    deviceStream_t stream);

// ============================================================================
// Performance Profiling API
// ============================================================================

// Print profiling report (call at end of processing)
// Enable profiling by setting RELION_METAL_PROFILING=1 environment variable
void printProfilingReport();

} // namespace MetalKernels

#endif // _METAL_ENABLED

#endif // METAL_KERNELS_H_
