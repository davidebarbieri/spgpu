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
texture < float, 1, cudaReadModeElementType > x_tex;

#define bind_tex_x(x) cudaBindTexture(NULL, x_tex, x)
#define unbind_tex_x(x) cudaUnbindTexture(x_tex)
#endif



__device__ void
spgpuSdiaspmv_ (float *z, const float *y, float alpha, const float* dM, const int* offsets, int dMPitch, int rows, int cols, int diags, const float *x, float beta)
{
	int i = threadIdx.x + blockIdx.x * (blockDim.x);
	
	float yVal = 0.0f;


	if (i < rows && beta != 0.0f)
		yVal = y[i];

	float zProd = 0.0f;

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
				// prefetch 3 values
				int column1 = offsetsChunk[j] + i;
				int column2 = offsetsChunk[j+1] + i;
				int column3 = offsetsChunk[j+2] + i;				
				
				float xValue1, xValue2, xValue3;
				float mValue1, mValue2, mValue3;
				
				bool inside1 = column1 >= 0 && column1 < cols;
				bool inside2 = column2 >= 0 && column2 < cols;
				bool inside3 = column3 >= 0 && column3 < cols;
				
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
					zProd = PREC_FADD(zProd, PREC_FMUL (xValue1, mValue1));
				
				if(inside2)
					zProd = PREC_FADD(zProd, PREC_FMUL (xValue2, mValue2));
					
				if(inside3)
					zProd = PREC_FADD(zProd, PREC_FMUL (xValue3, mValue3));
			}
	
			for (;j<count; ++j)
			{
				int column = offsetsChunk[j] + i;
				
				if(column >= 0 && column < cols)
                		{
#ifdef ENABLE_CACHE
					float xValue = tex1Dfetch (x_tex, column);
#else
					float xValue = x[column];
#endif				
			
					zProd = PREC_FADD(zProd, PREC_FMUL (xValue, dM[0]));
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
	
	if (beta == 0.0f)
		z[i] = PREC_FMUL(alpha, zProd);
	else
		z[i] = PREC_FADD(PREC_FMUL (beta, yVal), PREC_FMUL (alpha, zProd));
}

// Force to recompile and optimize with llvm
__global__ void
spgpuSdiaspmv_krn_b0 (float *z, const float *y, float alpha, const float* dM, const int* offsets, int dMPitch, int rows, int cols, int diags, const float *x)
{
	spgpuSdiaspmv_ (z, y, alpha, dM, offsets, dMPitch, rows, cols, diags, x, 0.0f);
}

__global__ void
spgpuSdiaspmv_krn (float *z, const float *y, float alpha, const float* dM, const int* offsets, int dMPitch, int rows, int cols, int diags, const float *x, float beta)
{
	spgpuSdiaspmv_ (z, y, alpha, dM, offsets, dMPitch, rows, cols, diags, x, beta);
}

void
_spgpuSdiaspmv (spgpuHandle_t handle, int threadCount, float* z, const float *y, float alpha, 
	const float* dM, const int* offsets, int dMPitch, int rows, int cols, int diags,
	const float *x, float beta)
{
	dim3 block (threadCount);
	dim3 grid ((rows + threadCount - 1) / threadCount);

#ifdef ENABLE_CACHE
	bind_tex_x (x);
#endif

	if (beta != 0.0f)
		spgpuSdiaspmv_krn <<< grid, block, block.x*sizeof(int), handle->currentStream >>> (z, y, alpha, dM, offsets, dMPitch, rows, cols, diags, x, beta);
	else
		spgpuSdiaspmv_krn_b0 <<< grid, block, block.x*sizeof(int), handle->currentStream >>> (z, y, alpha, dM, offsets, dMPitch, rows, cols, diags, x);

#ifdef ENABLE_CACHE
  	unbind_tex_x (x);
#endif

}

void 
spgpuSdiaspmv (spgpuHandle_t handle, 
	float* z, 
	const float *y, 
	float alpha, 
	const float* dM, 
	const int* offsets, 
	int dMPitch, 
	int rows,
	int cols, 
	int diags,
	const float *x, 
	float beta)
{
	cudaFuncSetCacheConfig(spgpuSdiaspmv_krn, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(spgpuSdiaspmv_krn_b0, cudaFuncCachePreferL1);
	
	cudaDeviceProp deviceProp;
    	cudaGetDeviceProperties(&deviceProp, 0);
    	
    	int threadCount;
    	
	if (deviceProp.major < 2)
    		threadCount = 64; 
    	else	
		threadCount = 512; 

	int maxThreadForACall = threadCount*65535;
	
	while (rows > maxThreadForACall) //managing large vectors
	{
		_spgpuSdiaspmv (handle, threadCount, z, y, alpha, dM, offsets, dMPitch, maxThreadForACall, cols, diags, x, beta);

		y = y + maxThreadForACall;
		dM = dM + maxThreadForACall;
		
		rows -= maxThreadForACall;
	}
	
	_spgpuSdiaspmv (handle, threadCount, z, y, alpha, dM, offsets, dMPitch, rows, cols, diags, x, beta);
	
	cudaCheckError("CUDA error on dia_sspmv");
}

