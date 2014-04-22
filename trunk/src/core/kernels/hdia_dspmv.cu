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
#include "hdia.h"
}

#include "debug.h"

#ifdef ENABLE_CACHE
// Texture cache management
texture < int2, 1, cudaReadModeElementType > x_tex;

#define bind_tex_x(x) cudaBindTexture(NULL, x_tex, x)
#define unbind_tex_x(x) cudaUnbindTexture(x_tex)
#endif

__device__ void
spgpuDhdiaspmv_ (double *z, const double *y, double alpha, const double* dM, const int* offsets, int hackSize, const int* hackOffsets, 
	int rows, int cols, const double *x, double beta, int hackCount)
{
	int i = threadIdx.x + blockIdx.x * (blockDim.x);
	
	double yVal = 0.0f;

	if (i < rows && beta != 0.0f)
		yVal = y[i];

	double zProd = 0.0f;
	
	int hackId = i / hackSize;
	int hackLaneId = i % hackSize;
	
	
	// shared between offsetsChunks and warpHackOffsetTemp
	extern __shared__ int dynShrMem[]; 

	int hackOffset = 0;
	int nextOffset = 0;
	
	unsigned int laneId = threadIdx.x % warpSize;
	unsigned int warpId = threadIdx.x / warpSize;
	
#if __CUDA_ARCH__ < 300	
	int* warpHackOffset = dynShrMem;


	if (laneId == 0 && i < rows)
	{
		warpHackOffset[warpId] = hackOffsets[hackId];
		warpHackOffset[warpId + (blockDim.x / warpSize)] = hackOffsets[hackId+1];
	}
	
	__syncthreads();
	hackOffset = warpHackOffset[warpId];
	nextOffset = warpHackOffset[warpId + blockDim.x / warpSize];
	__syncthreads();
#else
	if (laneId == 0 && i < rows)
	{
		hackOffset = hackOffsets[hackId];
		nextOffset = hackOffsets[hackId+1];
	}
	
	hackOffset = __shfl(hackOffset, 0);	
	nextOffset = __shfl(nextOffset, 0);
#endif
	
	if (hackId >= hackCount)
		return;

	dM += hackOffset*hackSize + hackLaneId;
	offsets += hackOffset;
	
	// diags for this hack is next hackOffset minus current hackOffset
	int diags = nextOffset - hackOffset;
	
	
	// Warp oriented
	int rounds = (diags + warpSize - 1)/warpSize;
	
	volatile int *offsetsChunk = dynShrMem + warpId*warpSize;
	
	for (int r = 0; r < rounds; r++)
	{
		// in the last round diags will be <= warpSize
		if (laneId < diags)
			offsetsChunk[laneId] = offsets[laneId];
	
		if (i < rows)
		{
			int count = min(diags, warpSize);
			
			int j;
			for (j=0; j<=count-2; j += 2)
			{
				// prefetch 3 values
				int column1 = offsetsChunk[j] + i;
				int column2 = offsetsChunk[j+1] + i;			
				
				int2 xValue1, xValue2;
				double mValue1, mValue2;
				
				bool inside1 = column1 >= 0 && column1 < cols;
				bool inside2 = column2 >= 0 && column2 < cols;
				
				if(inside1)
                		{
                			mValue1 = dM[0];
#ifdef ENABLE_CACHE
					xValue1 = tex1Dfetch (x_tex, column1);
#else
					xValue1 = x[column1];
#endif				
				}
				
				dM += hackSize;
							
				if(inside2)
                		{
                			mValue2 = dM[0];
#ifdef ENABLE_CACHE
					xValue2 = tex1Dfetch (x_tex, column2);
#else
					xValue2 = x[column2];
#endif					
				}

				dM += hackSize;					
											
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
				
				dM += hackSize;
			}
		}
		
		diags -= warpSize;
		offsets += warpSize;
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
spgpuDhdiaspmv_krn_b0 (double *z, const double *y, double alpha, const double* dM, const int* offsets, int hackSize, const int* hackOffsets, int rows, int cols, const double *x, int hackCount)
{
	spgpuDhdiaspmv_ (z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, 0.0, hackCount);
}

__global__ void
spgpuDhdiaspmv_krn (double *z, const double *y, double alpha, const double* dM, const int* offsets, int hackSize, const int* hackOffsets, int rows, int cols, const double *x, double beta, int hackCount)
{
	spgpuDhdiaspmv_ (z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, beta, hackCount);
}

void
_spgpuDhdiaspmv (spgpuHandle_t handle, int threadCount, double* z, const double *y, double alpha, 
	const double* dM, const int* offsets, int hackSize, const int* hackOffsets, int rows, int cols,
	const double *x, double beta)
{
	dim3 block (threadCount);
	dim3 grid ((rows + threadCount - 1) / threadCount);

	int hackCount = (rows + hackSize - 1)/hackSize;
	
#ifdef ENABLE_CACHE
	bind_tex_x (x);
#endif

	if (beta != 0.0)
		spgpuDhdiaspmv_krn <<< grid, block, block.x*sizeof(int), handle->currentStream >>> (z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, beta, hackCount);
	else
		spgpuDhdiaspmv_krn_b0 <<< grid, block, block.x*sizeof(int), handle->currentStream >>> (z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, hackCount);

#ifdef ENABLE_CACHE
  	unbind_tex_x (x);
#endif

}

void 
spgpuDhdiaspmv (spgpuHandle_t handle, 
	double* z, 
	const double *y, 
	double alpha, 
	const double* dM, 
	const int* offsets, 
	int hackSize, 
	const int* hackOffsets,
	int rows,
	int cols, 
	const double *x, 
	double beta)
{
	__assert(hackSize % 32 == 0, "Only hacks whose length is a multiple of 32 are supported...");
	
	cudaFuncSetCacheConfig(spgpuDhdiaspmv_krn, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(spgpuDhdiaspmv_krn_b0, cudaFuncCachePreferL1);
	
	cudaDeviceProp deviceProp;
    	cudaGetDeviceProperties(&deviceProp, 0);
    	
    	int threadCount = 128;

	int maxThreadForACall = threadCount*65535;
	
	while (rows > maxThreadForACall) //managing large vectors
	{
		_spgpuDhdiaspmv (handle, threadCount, z, y, alpha, dM, offsets, hackSize, hackOffsets, maxThreadForACall, cols, x, beta);

		y = y + maxThreadForACall;
		z = z + maxThreadForACall;
		hackOffsets += maxThreadForACall/hackSize;
		
		rows -= maxThreadForACall;
	}
	
	_spgpuDhdiaspmv (handle, threadCount, z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, beta);
	
	cudaCheckError("CUDA error on hdia_dspmv");
}
