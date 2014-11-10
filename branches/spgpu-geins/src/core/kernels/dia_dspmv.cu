/*
 * spGPU - Sparse matrices on GPU library.
 * 
 * Copyright (C) 2010 - 2013
 *     Davide Barbieri - University of Rome Tor Vergata
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * version 3 as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */
 
#include "cudadebug.h"
#include "cudalang.h"

extern "C"
{
#include "core.h"
#include "dia.h"
}

#include "debug.h"

#ifdef ENABLE_CACHE
// Texture cache management
texture < int2, 1, cudaReadModeElementType > x_tex;

#define bind_tex_x(x) cudaBindTexture(NULL, x_tex, x)
#define unbind_tex_x(x) cudaUnbindTexture(x_tex)
#endif

/*
__device__ void
spgpuDdiaspmv_ (double *z, const double *y, double alpha, const double* dM, const int* offsets, int dMPitch, int rows, int cols, int diags, const double *x, double beta)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	double zProd = 0.0;
	double t_a = 0.0;

	double yVal = 0.0;
	if (idx < rows && beta != 0.0)
		yVal = y[idx];
	
	if(idx<rows)
	{
		for(int i=0; i<diags; i++)
		{
			t_a = dM[i*dMPitch+idx];
			int temp_off = offsets[i];
			int c = temp_off+idx;
			
			if(c >= 0 && c < cols)
				zProd += t_a*x[idx+temp_off];
      		}
      		
      		if (beta == 0.0)
			z[idx] = PREC_DMUL(alpha, zProd);
		else
			z[idx] = PREC_DADD(PREC_DMUL (beta, yVal), PREC_DMUL (alpha, zProd));
	}
}
*/

__device__ void
spgpuDdiaspmv_ (double *z, const double *y, double alpha, const double* dM, const int* offsets, int dMPitch, int rows, int cols, int diags, const double *x, double beta)
{
	int i = threadIdx.x + blockIdx.x * (blockDim.x);
	
	double yVal = 0.0;


	if (i < rows && beta != 0.0)
		yVal = y[i];

	double zProd = 0.0;

	dM += i;

	extern __shared__ int offsetsChunk[];

	int rounds = (diags + blockDim.x - 1)/blockDim.x;
	
	for (int r = 0; r < rounds; r++)
	{
		// in the last round diags will be <= blockDim.x
		if (threadIdx.x < diags)
			offsetsChunk[threadIdx.x] = offsets[threadIdx.x];
	
		__syncthreads();
	
		if (i < rows)
		{
			int count = min(diags, blockDim.x );
			
			int j;
			for (j=0; j<=count-3; j += 3)
			{
				// Prefetch 3 values
				int column1 = offsetsChunk[j] + i;
				int column2 = offsetsChunk[j+1] + i;
				int column3 = offsetsChunk[j+2] + i;	
				
				bool inside1 = column1 >= 0 && column1 < cols;
				bool inside2 = column2 >= 0 && column2 < cols;
				bool inside3 = column3 >= 0 && column3 < cols;			

				// Anticipate global memory read
				
				
#ifdef ENABLE_CACHE
				int2 xValue1, xValue2, xValue3;
#else				
				double xValue1, xValue2, xValue3;
#endif		
				double mValue1, mValue2, mValue3;
				
				if(inside1)
                		{
                			mValue1 = dM[0];
#ifdef ENABLE_CACHE
					xValue1 = tex1Dfetch (x_tex, column1);
#else
					xValue1 = x[column1];
#endif				
				}
				dM += dMPitch;
							
				if(inside2)
                		{
                			mValue2 = dM[0];
#ifdef ENABLE_CACHE
					xValue2 = tex1Dfetch (x_tex, column2);
#else
					xValue2 = x[column2];
#endif				
				}
				dM += dMPitch;
							
				if(inside3)
                		{
                			mValue3 = dM[0];
#ifdef ENABLE_CACHE
					xValue3 = tex1Dfetch (x_tex, column3);
#else
					xValue3 = x[column3];
#endif				
				}
				dM += dMPitch;
							
				if(inside1)
#ifdef ENABLE_CACHE
					zProd = PREC_DADD(zProd, PREC_DMUL (mValue1, __hiloint2double (xValue1.y, xValue1.x)));
#else
					zProd = PREC_DADD(zProd, PREC_DMUL (mValue1, xValue1));
#endif	


				if(inside2)
#ifdef ENABLE_CACHE
					zProd = PREC_DADD(zProd, PREC_DMUL (mValue2, __hiloint2double (xValue2.y, xValue2.x)));
#else
					zProd = PREC_DADD(zProd, PREC_DMUL (mValue2, xValue2));
#endif
					
				if(inside3)
#ifdef ENABLE_CACHE
					zProd = PREC_DADD(zProd, PREC_DMUL (mValue3, __hiloint2double (xValue3.y, xValue3.x)));
#else
					zProd = PREC_DADD(zProd, PREC_DMUL (mValue3, xValue3));
#endif
				
				
			}
	
			for (;j<count; ++j)
			{
				int column = offsetsChunk[j] + i;
				
				if(column >= 0 && column < cols)
                		{
#ifdef ENABLE_CACHE
					int2 xValue = tex1Dfetch (x_tex, column);
					zProd = PREC_DADD(zProd, PREC_DMUL (dM[0], __hiloint2double (xValue.y, xValue.x)));
#else
					double xValue = x[column];
					zProd = PREC_DADD(zProd, PREC_DMUL (dM[0], xValue));
#endif				
			
				}
				
				dM += dMPitch;
			}
		}
		
		diags -= blockDim.x;
		offsets += blockDim.x;
		__syncthreads();
	}


	// Since z and y are accessed with the same offset by the same thread,
	// and the write to z follows the y read, y and z can share the same base address (in-place computing).
	
	if (i >= rows)
		return;
	
	if (beta == 0.0)
		z[i] = PREC_DMUL(alpha, zProd);
	else
		z[i] = PREC_DADD(PREC_DMUL (beta, yVal), PREC_DMUL (alpha, zProd));
}

// Force to recompile and optimize with llvm
__global__ void
spgpuDdiaspmv_krn_b0 (double *z, const double *y, double alpha, const double* dM, const int* offsets, int dMPitch, int rows, int cols, int diags, const double *x)
{
	spgpuDdiaspmv_ (z, y, alpha, dM, offsets, dMPitch, rows, cols, diags, x, 0.0);
}

__global__ void
spgpuDdiaspmv_krn (double *z, const double *y, double alpha, const double* dM, const int* offsets, int dMPitch, int rows, int cols, int diags, const double *x, double beta)
{
	spgpuDdiaspmv_ (z, y, alpha, dM, offsets, dMPitch, rows, cols, diags, x, beta);
}

void
_spgpuDdiaspmv (spgpuHandle_t handle, int threadCount, double* z, const double *y, double alpha, 
	const double* dM, const int* offsets, int dMPitch, int rows, int cols, int diags,
	const double *x, double beta)
{
	dim3 block (threadCount);
	dim3 grid ((rows + threadCount - 1) / threadCount);

#ifdef ENABLE_CACHE
	bind_tex_x (x);
#endif

	if (beta != 0.0)
		spgpuDdiaspmv_krn <<< grid, block, block.x*sizeof(int), handle->currentStream >>> (z, y, alpha, dM, offsets, dMPitch, rows, cols, diags, x, beta);
	else
		spgpuDdiaspmv_krn_b0 <<< grid, block, block.x*sizeof(int), handle->currentStream >>> (z, y, alpha, dM, offsets, dMPitch, rows, cols, diags, x);

#ifdef ENABLE_CACHE
  	unbind_tex_x (x);
#endif

}

void 
spgpuDdiaspmv (spgpuHandle_t handle, 
	double* z, 
	const double *y, 
	double alpha, 
	const double* dM, 
	const int* offsets, 
	int dMPitch, 
	int rows,
	int cols, 
	int diags,
	const double *x, 
	double beta)
{
	cudaFuncSetCacheConfig(spgpuDdiaspmv_krn, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(spgpuDdiaspmv_krn_b0, cudaFuncCachePreferL1);
	
	cudaDeviceProp deviceProp;
    	cudaGetDeviceProperties(&deviceProp, 0);
    	
    	int threadCount = 128;
	
	int maxNForACall = max(handle->maxGridSizeX, threadCount*handle->maxGridSizeX);
	
	while (rows > maxNForACall) //managing large vectors
	{
		_spgpuDdiaspmv (handle, threadCount, z, y, alpha, dM, offsets, dMPitch, maxNForACall, cols, diags, x, beta);

		y = y + maxNForACall;
		z = z + maxNForACall;
		dM = dM + maxNForACall;
		
		rows -= maxNForACall;
	}
	
	_spgpuDdiaspmv (handle, threadCount, z, y, alpha, dM, offsets, dMPitch, rows, cols, diags, x, beta);
	
	cudaCheckError("CUDA error on dia_sspmv");
}

