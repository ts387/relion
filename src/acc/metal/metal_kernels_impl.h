/*
 * Metal Kernel Template Implementations
 *
 * This file provides the inline template wrappers that dispatch to the
 * Metal kernel implementations. The actual Metal code is in metal_kernels.mm.
 */

#ifndef METAL_KERNELS_IMPL_H_
#define METAL_KERNELS_IMPL_H_

#include "src/acc/metal/metal_kernels.h"

#ifdef _METAL_ENABLED

namespace MetalKernels {

// Non-template implementations declared in metal_kernels.mm
// These are the actual Metal kernel launchers
void diff2_coarse_impl(
    unsigned long grid_size,
    int block_size,
    XFLOAT *g_eulers,
    XFLOAT *trans_x,
    XFLOAT *trans_y,
    XFLOAT *trans_z,
    XFLOAT *g_real,
    XFLOAT *g_imag,
    XFLOAT *mdlReal,
    XFLOAT *mdlImag,
    int mdlX, int mdlXY, int mdlZ,
    int imgX, int imgY, int imgZ,
    int mdlInitY, int mdlInitZ,
    int maxR, int maxR2,
    XFLOAT padding_factor,
    XFLOAT *g_corr,
    XFLOAT *g_diff2s,
    unsigned long translation_num,
    unsigned long image_size,
    int eulers_per_block,
    int prefetch_fraction,
    bool is_3D,
    deviceStream_t stream);

void diff2_fine_impl(
    unsigned long grid_size,
    int block_size,
    XFLOAT *g_eulers,
    XFLOAT *g_imgs_real,
    XFLOAT *g_imgs_imag,
    XFLOAT *trans_x,
    XFLOAT *trans_y,
    XFLOAT *trans_z,
    XFLOAT *mdlReal,
    XFLOAT *mdlImag,
    int mdlX, int mdlXY, int mdlZ,
    int imgX, int imgY, int imgZ,
    int mdlInitY, int mdlInitZ,
    int maxR, int maxR2,
    XFLOAT padding_factor,
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
    int chunk_sz,
    bool is_3D,
    deviceStream_t stream);

// ============================================================================
// Template wrappers that call the non-template implementations
// ============================================================================

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
    deviceStream_t stream)
{
    diff2_coarse_impl(
        grid_size,
        block_sz,
        g_eulers,
        trans_x,
        trans_y,
        trans_z,
        g_real,
        g_imag,
        projector.mdlReal,
        projector.mdlImag,
        projector.mdlX,
        projector.mdlXY,
        projector.mdlZ,
        projector.imgX,
        projector.imgY,
        projector.imgZ,
        projector.mdlInitY,
        projector.mdlInitZ,
        projector.maxR,
        projector.maxR2,
        projector.padding_factor,
        g_corr,
        g_diff2s,
        translation_num,
        image_size,
        eulers_per_block,
        prefetch_fraction,
        DATA3D,
        stream
    );
}

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
    deviceStream_t stream)
{
    diff2_fine_impl(
        grid_size,
        block_sz,
        g_eulers,
        g_imgs_real,
        g_imgs_imag,
        trans_x,
        trans_y,
        trans_z,
        projector.mdlReal,
        projector.mdlImag,
        projector.mdlX,
        projector.mdlXY,
        projector.mdlZ,
        projector.imgX,
        projector.imgY,
        projector.imgZ,
        projector.mdlInitY,
        projector.mdlInitZ,
        projector.maxR,
        projector.maxR2,
        projector.padding_factor,
        g_corr_img,
        g_diff2s,
        image_size,
        sum_init,
        orientation_num,
        translation_num,
        todo_blocks,
        d_rot_idx,
        d_trans_idx,
        d_job_idx,
        d_job_num,
        chunk_sz,
        DATA3D,
        stream
    );
}

} // namespace MetalKernels

#endif // _METAL_ENABLED

#endif // METAL_KERNELS_IMPL_H_
