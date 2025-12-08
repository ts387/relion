#ifndef ACC_PROJECTORKERNELIMPL_H_
#define ACC_PROJECTORKERNELIMPL_H_

#include <cmath>

#ifndef PROJECTOR_NO_TEXTURES
	#ifdef _CUDA_ENABLED
		#define PROJECTOR_PTR_TYPE cudaTextureObject_t
	#elif _HIP_ENABLED
		#define PROJECTOR_PTR_TYPE hipTextureObject_t
	#endif
#else
	#define PROJECTOR_PTR_TYPE XFLOAT *
#endif

// Metal-specific trilinear interpolation helpers for separate real/imag arrays
#ifdef _METAL_ENABLED
namespace MetalInterpolation {

// 3D trilinear interpolation for separate real/imag arrays
inline void trilinear3D(
    const XFLOAT* mdlReal, const XFLOAT* mdlImag,
    XFLOAT xp, XFLOAT yp, XFLOAT zp,
    int mdlX, int mdlXY, int mdlInitY, int mdlInitZ,
    XFLOAT &real, XFLOAT &imag)
{
    // Adjust for array origin
    yp -= mdlInitY;
    zp -= mdlInitZ;

    // Get integer and fractional parts
    int x0 = (int)floor(xp);
    int y0 = (int)floor(yp);
    int z0 = (int)floor(zp);

    XFLOAT fx = xp - x0;
    XFLOAT fy = yp - y0;
    XFLOAT fz = zp - z0;

    // Clamp to valid range
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    // Calculate linear indices for 8 corners
    // Using RELION's indexing: idx = z * mdlXY + y * mdlX + x
    auto idx = [&](int x, int y, int z) -> size_t {
        return (size_t)z * mdlXY + (size_t)y * mdlX + (size_t)x;
    };

    // Trilinear interpolation weights
    XFLOAT w000 = (1-fx) * (1-fy) * (1-fz);
    XFLOAT w100 = fx * (1-fy) * (1-fz);
    XFLOAT w010 = (1-fx) * fy * (1-fz);
    XFLOAT w110 = fx * fy * (1-fz);
    XFLOAT w001 = (1-fx) * (1-fy) * fz;
    XFLOAT w101 = fx * (1-fy) * fz;
    XFLOAT w011 = (1-fx) * fy * fz;
    XFLOAT w111 = fx * fy * fz;

    // Sample and interpolate
    real = w000 * mdlReal[idx(x0,y0,z0)] + w100 * mdlReal[idx(x1,y0,z0)] +
           w010 * mdlReal[idx(x0,y1,z0)] + w110 * mdlReal[idx(x1,y1,z0)] +
           w001 * mdlReal[idx(x0,y0,z1)] + w101 * mdlReal[idx(x1,y0,z1)] +
           w011 * mdlReal[idx(x0,y1,z1)] + w111 * mdlReal[idx(x1,y1,z1)];

    imag = w000 * mdlImag[idx(x0,y0,z0)] + w100 * mdlImag[idx(x1,y0,z0)] +
           w010 * mdlImag[idx(x0,y1,z0)] + w110 * mdlImag[idx(x1,y1,z0)] +
           w001 * mdlImag[idx(x0,y0,z1)] + w101 * mdlImag[idx(x1,y0,z1)] +
           w011 * mdlImag[idx(x0,y1,z1)] + w111 * mdlImag[idx(x1,y1,z1)];
}

// 2D bilinear interpolation for separate real/imag arrays
inline void bilinear2D(
    const XFLOAT* mdlReal, const XFLOAT* mdlImag,
    XFLOAT xp, XFLOAT yp,
    int mdlX, int mdlInitY,
    XFLOAT &real, XFLOAT &imag)
{
    // Adjust for array origin
    yp -= mdlInitY;

    // Get integer and fractional parts
    int x0 = (int)floor(xp);
    int y0 = (int)floor(yp);

    XFLOAT fx = xp - x0;
    XFLOAT fy = yp - y0;

    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // Calculate linear indices
    auto idx = [&](int x, int y) -> size_t {
        return (size_t)y * mdlX + (size_t)x;
    };

    // Bilinear interpolation weights
    XFLOAT w00 = (1-fx) * (1-fy);
    XFLOAT w10 = fx * (1-fy);
    XFLOAT w01 = (1-fx) * fy;
    XFLOAT w11 = fx * fy;

    // Sample and interpolate
    real = w00 * mdlReal[idx(x0,y0)] + w10 * mdlReal[idx(x1,y0)] +
           w01 * mdlReal[idx(x0,y1)] + w11 * mdlReal[idx(x1,y1)];

    imag = w00 * mdlImag[idx(x0,y0)] + w10 * mdlImag[idx(x1,y0)] +
           w01 * mdlImag[idx(x0,y1)] + w11 * mdlImag[idx(x1,y1)];
}

} // namespace MetalInterpolation
#endif // _METAL_ENABLED

class AccProjectorKernel
{

public:
	int mdlX, mdlXY, mdlZ,
		imgX, imgY, imgZ,
		mdlInitY, mdlInitZ,
		maxR, maxR2, maxR2_padded;
	XFLOAT 	padding_factor;

#if defined _CUDA_ENABLED || defined _HIP_ENABLED
	PROJECTOR_PTR_TYPE mdlReal;
	PROJECTOR_PTR_TYPE mdlImag;
#elif defined _METAL_ENABLED
	XFLOAT *mdlReal;
	XFLOAT *mdlImag;
#elif _SYCL_ENABLED
	PROJECTOR_PTR_TYPE mdlComplex;
#else
	std::complex<XFLOAT> *mdlComplex;
#endif

#if defined _CUDA_ENABLED || defined _HIP_ENABLED
	AccProjectorKernel(
			int mdlX, int mdlY, int mdlZ,
			int imgX, int imgY, int imgZ,
			int mdlInitY, int mdlInitZ,
			XFLOAT padding_factor,
			int maxR,
			PROJECTOR_PTR_TYPE mdlReal, PROJECTOR_PTR_TYPE mdlImag
			):
				mdlX(mdlX), mdlXY(mdlX*mdlY), mdlZ(mdlZ),
				imgX(imgX), imgY(imgY), imgZ(imgZ),
				mdlInitY(mdlInitY), mdlInitZ(mdlInitZ),
				padding_factor(padding_factor),
				maxR(maxR), maxR2(maxR*maxR), maxR2_padded(maxR*maxR*padding_factor*padding_factor),
				mdlReal(mdlReal), mdlImag(mdlImag)
		{};
#elif defined _METAL_ENABLED
	AccProjectorKernel(
			int mdlX, int mdlY, int mdlZ,
			int imgX, int imgY, int imgZ,
			int mdlInitY, int mdlInitZ,
			XFLOAT padding_factor,
			int maxR,
			XFLOAT *mdlReal, XFLOAT *mdlImag
			):
				mdlX(mdlX), mdlXY(mdlX*mdlY), mdlZ(mdlZ),
				imgX(imgX), imgY(imgY), imgZ(imgZ),
				mdlInitY(mdlInitY), mdlInitZ(mdlInitZ),
				padding_factor(padding_factor),
				maxR(maxR), maxR2(maxR*maxR), maxR2_padded(maxR*maxR*padding_factor*padding_factor),
				mdlReal(mdlReal), mdlImag(mdlImag)
		{};
#else
	AccProjectorKernel(
			int mdlX, int mdlY, int mdlZ,
			int imgX, int imgY, int imgZ,
			int mdlInitY, int mdlInitZ,
			XFLOAT padding_factor,
			int maxR,
#if _SYCL_ENABLED
			PROJECTOR_PTR_TYPE mdlComplex
#else
			std::complex<XFLOAT> *mdlComplex
#endif
			):
			mdlX(mdlX), mdlXY(mdlX*mdlY), mdlZ(mdlZ),
			imgX(imgX), imgY(imgY), imgZ(imgZ),
			mdlInitY(mdlInitY), mdlInitZ(mdlInitZ),
			padding_factor(padding_factor),
			maxR(maxR), maxR2(maxR*maxR), maxR2_padded(maxR*maxR*padding_factor*padding_factor),
			mdlComplex(mdlComplex)
		{};
#endif

#if defined _CUDA_ENABLED || defined _HIP_ENABLED
	__device__ __forceinline__
#else
	inline
#endif
	void project3Dmodel(
			int x,
			int y,
			int z,
			XFLOAT e0,
			XFLOAT e1,
			XFLOAT e2,
			XFLOAT e3,
			XFLOAT e4,
			XFLOAT e5,
			XFLOAT e6,
			XFLOAT e7,
			XFLOAT e8,
			XFLOAT &real,
			XFLOAT &imag)
	{
		XFLOAT xp = (e0 * x + e1 * y + e2 * z) * padding_factor;
		XFLOAT yp = (e3 * x + e4 * y + e5 * z) * padding_factor;
		XFLOAT zp = (e6 * x + e7 * y + e8 * z) * padding_factor;

		int r2 = xp*xp + yp*yp + zp*zp;

		if (r2 <= maxR2_padded)
		{

#ifdef PROJECTOR_NO_TEXTURES
			bool invers(xp < 0);
			if (invers)
			{
				xp = -xp;
				yp = -yp;
				zp = -zp;
			}

#if defined _CUDA_ENABLED || defined _HIP_ENABLED
			real =   no_tex3D(mdlReal, xp, yp, zp, mdlX, mdlXY, mdlInitY, mdlInitZ);
			imag = - no_tex3D(mdlImag, xp, yp, zp, mdlX, mdlXY, mdlInitY, mdlInitZ);
#elif defined _METAL_ENABLED
			// Metal: trilinear interpolation on separate real/imag arrays
			MetalInterpolation::trilinear3D(mdlReal, mdlImag, xp, yp, zp,
			                                mdlX, mdlXY, mdlInitY, mdlInitZ,
			                                real, imag);
			imag = -imag;
#elif _SYCL_ENABLED
			syclKernels::no_tex3D(mdlComplex, real, imag, xp, yp, zp, mdlX, mdlXY, mdlInitY, mdlInitZ);
			imag = -imag;
#else
			CpuKernels::complex3D(mdlComplex, real, imag, xp, yp, zp, mdlX, mdlXY, mdlInitY, mdlInitZ);
#endif

			if(invers)
			    imag = -imag;


#else
			if (xp < 0)
			{
				// Get complex conjugated hermitian symmetry pair
				xp = -xp;
				yp = -yp;
				zp = -zp;

				yp -= mdlInitY;
				zp -= mdlInitZ;

				real =    tex3D<XFLOAT>(mdlReal, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
				imag =  - tex3D<XFLOAT>(mdlImag, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
			}
			else
			{
				yp -= mdlInitY;
				zp -= mdlInitZ;

				real =   tex3D<XFLOAT>(mdlReal, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
				imag =   tex3D<XFLOAT>(mdlImag, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
			}
#endif
		}
		else
		{
			real = (XFLOAT)0;
			imag = (XFLOAT)0;
		}
	}

#if defined _CUDA_ENABLED || defined _HIP_ENABLED
	__device__ __forceinline__
#else
	inline
#endif
	void project3Dmodel(
			int x,
			int y,
			XFLOAT e0,
			XFLOAT e1,
			XFLOAT e3,
			XFLOAT e4,
			XFLOAT e6,
			XFLOAT e7,
			XFLOAT &real,
			XFLOAT &imag)
	{
		XFLOAT xp = (e0 * x + e1 * y ) * padding_factor;
		XFLOAT yp = (e3 * x + e4 * y ) * padding_factor;
		XFLOAT zp = (e6 * x + e7 * y ) * padding_factor;

		int r2 = xp*xp + yp*yp + zp*zp;

		if (r2 <= maxR2_padded)
		{

#ifdef PROJECTOR_NO_TEXTURES
			bool invers(xp < 0);
			if (invers)
			{
				xp = -xp;
				yp = -yp;
				zp = -zp;
			}

	#if defined _CUDA_ENABLED || defined _HIP_ENABLED
			real = no_tex3D(mdlReal, xp, yp, zp, mdlX, mdlXY, mdlInitY, mdlInitZ);
			imag = no_tex3D(mdlImag, xp, yp, zp, mdlX, mdlXY, mdlInitY, mdlInitZ);
	#elif defined _METAL_ENABLED
			// Metal: trilinear interpolation on separate real/imag arrays
			MetalInterpolation::trilinear3D(mdlReal, mdlImag, xp, yp, zp,
			                                mdlX, mdlXY, mdlInitY, mdlInitZ,
			                                real, imag);
	#elif _SYCL_ENABLED
			syclKernels::no_tex3D(mdlComplex, real, imag, xp, yp, zp, mdlX, mdlXY, mdlInitY, mdlInitZ);
	#else
			CpuKernels::complex3D(mdlComplex, real, imag, xp, yp, zp, mdlX, mdlXY, mdlInitY, mdlInitZ);
	#endif

			if(invers)
			    imag = -imag;
#else
			if (xp < 0)
			{
				// Get complex conjugated hermitian symmetry pair
				xp = -xp;
				yp = -yp;
				zp = -zp;

				yp -= mdlInitY;
				zp -= mdlInitZ;

				real =    tex3D<XFLOAT>(mdlReal, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
				imag =  - tex3D<XFLOAT>(mdlImag, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
			}
			else
			{
				yp -= mdlInitY;
				zp -= mdlInitZ;

				real =   tex3D<XFLOAT>(mdlReal, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
				imag =   tex3D<XFLOAT>(mdlImag, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
			}
#endif
		}
		else
		{
			real = (XFLOAT)0;
			imag = (XFLOAT)0;
		}
	}

#if defined _CUDA_ENABLED || defined _HIP_ENABLED
__device__ __forceinline__
#else
	inline
#endif
	void project2Dmodel(
				int x,
				int y,
				XFLOAT e0,
				XFLOAT e1,
				XFLOAT e3,
				XFLOAT e4,
				XFLOAT &real,
				XFLOAT &imag)
	{
		XFLOAT xp = (e0 * x + e1 * y ) * padding_factor;
		XFLOAT yp = (e3 * x + e4 * y ) * padding_factor;

		int r2 = xp*xp + yp*yp;

		if (r2 <= maxR2_padded)
		{
#ifdef PROJECTOR_NO_TEXTURES
			bool invers(xp < 0);
			if (invers)
			{
				xp = -xp;
				yp = -yp;
			}

	#if defined _CUDA_ENABLED || defined _HIP_ENABLED
			real = no_tex2D(mdlReal, xp, yp, mdlX, mdlInitY);
			imag = no_tex2D(mdlImag, xp, yp, mdlX, mdlInitY);
	#elif defined _METAL_ENABLED
			// Metal: bilinear interpolation on separate real/imag arrays
			MetalInterpolation::bilinear2D(mdlReal, mdlImag, xp, yp,
			                               mdlX, mdlInitY, real, imag);
	#elif _SYCL_ENABLED
			syclKernels::no_tex2D(mdlComplex, real, imag, xp, yp, mdlX, mdlInitY);
	#else
			CpuKernels::complex2D(mdlComplex, real, imag, xp, yp, mdlX, mdlInitY);
	#endif

			if(invers)
			    imag = -imag;

#else
			if (xp < 0)
			{
				// Get complex conjugated hermitian symmetry pair
				xp = -xp;
				yp = -yp;
				yp -= mdlInitY;

				real =   tex2D<XFLOAT>(mdlReal, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5);
				imag = - tex2D<XFLOAT>(mdlImag, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5);
			}
			else
			{
				yp -= mdlInitY;
				real =   tex2D<XFLOAT>(mdlReal, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5);
				imag =   tex2D<XFLOAT>(mdlImag, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5);
			}
#endif
		}
		else
		{
			real=(XFLOAT)0;
			imag=(XFLOAT)0;
		}
	}

	static AccProjectorKernel makeKernel(AccProjector &p, int imgX, int imgY, int imgZ, int imgMaxR)
	{
		int maxR = p.mdlMaxR >= imgMaxR ? imgMaxR : p.mdlMaxR;

		AccProjectorKernel k(
					p.mdlX, p.mdlY, p.mdlZ,
					imgX, imgY, imgZ,
					p.mdlInitY, p.mdlInitZ,
					p.padding_factor,
					maxR,
#if defined _METAL_ENABLED
					p.mdlReal,
					p.mdlImag
#elif !defined PROJECTOR_NO_TEXTURES
					*p.mdlReal,
					*p.mdlImag
#else
					p.mdlComplex
#endif
				);
		return k;
	}
};  // class AccProjectorKernel


#endif
