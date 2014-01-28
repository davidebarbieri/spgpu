/*
 * spGPU - Sparse matrices on GPU library.
 * 
 * Copyright (C) 2010 - 2014
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
texture < float, 1, cudaReadModeElementType > x_tex;

#define bind_tex_x(x) cudaBindTexture(NULL, x_tex, x)
#define unbind_tex_x(x) cudaUnbindTexture(x_tex)
#endif

template<int blockRows,int blockCols>
__device__ void
spgpuSbhdiaspmv_ (float *z, const float *y, float alpha, const float* dM, const int* offsets, int hackSize, const int* hackOffsets, int rows, int cols, const float *x, float beta, int hackCount)
{
	int id = threadIdx.x + blockIdx.x * (blockDim.x);
	
	float yVal[blockRows];

#pragma unroll
	for (int j=0; j<blockRows; ++j)
		yVal[j] = 0.0f;
			
	if (beta != 0.0f)
	{
		#pragma unroll
		for (int j=0; j<blockRows; ++j)
			yVal[j] = y[id*blockRows + j];
	}

	float zProd[blockRows];
	
	#pragma unroll
	for (int j=0; j<blockRows; ++j)
	{
		zProd[j] = 0.0f;
	}
	
	int hackId = id / hackSize;
	int hackLaneId = id % hackSize;
	
	
	// shared between offsetsChunks and warpHackOffsetTemp
	extern __shared__ int dynShrMem[]; 

	int hackOffset = 0;
	int nextOffset = 0;
	
	unsigned int laneId = threadIdx.x % warpSize;
	unsigned int warpId = threadIdx.x / warpSize;
	
#if __CUDA_ARCH__ < 300	
	int* warpHackOffset = dynShrMem;


	if (laneId == 0 && id < rows)
	{
		warpHackOffset[warpId] = hackOffsets[hackId];
		warpHackOffset[warpId + (blockDim.x / warpSize)] = hackOffsets[hackId+1];
	}
	
	__syncthreads();
	hackOffset = warpHackOffset[warpId];
	nextOffset = warpHackOffset[warpId + blockDim.x / warpSize];
	__syncthreads();
#else
	if (laneId == 0 && id < rows)
	{
		hackOffset = hackOffsets[hackId];
		nextOffset = hackOffsets[hackId+1];
	}
	
	hackOffset = __shfl(hackOffset, 0);	
	nextOffset = __shfl(nextOffset, 0);
#endif
	
	if (hackId >= hackCount)
		return;

	int blockSize = (blockRows*blockCols);
	dM += blockSize*(hackOffset*hackSize + hackLaneId);
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
	
		if (id < rows)
		{
			int count = min(diags, warpSize);
			
			for (int j=0;j<count; ++j)
			{
				int blockColumn = offsetsChunk[j] + id;
				
				if(blockColumn >= 0 && blockColumn < cols)
                		{
                			float xBlock[blockCols];
#ifdef ENABLE_CACHE
					#pragma unroll
					for (int k=0; k < blockCols; ++k)
					{
						xBlock[k] = tex1Dfetch (x_tex, blockColumn*blockCols + k);
					}
					
					const float* innerDm = dM;
					#pragma unroll					
					for (int k=0; k < blockCols; ++k)
					#pragma unroll					
						for (int l=0; l < blockRows; ++l)
						{
							zProd[l] = PREC_FADD(zProd[l], PREC_FMUL (innerDm[0],xBlock[k]));
							innerDm = innerDm + 1;
						}					
#else
					
#endif				
				}
				
				dM += hackSize*blockSize;
			}
		}
		
		diags -= warpSize;
		offsets += warpSize;
	}


	// Since z and y are accessed with the same offset by the same thread,
	// and the write to z follows the y read, y and z can share the same base address (in-place computing).
	
	if (id >= rows)
		return;
	
	if (beta == 0.0f)
	{
#pragma unroll
		for (int j=0; j<blockRows; ++j)
			z[id*blockRows + j] = PREC_FMUL(alpha, zProd[j]);
	}
	else
	{
#pragma unroll
		for (int j=0; j<blockRows; ++j)
			z[id*blockRows + j] = PREC_FADD(PREC_FMUL (beta, yVal[j]), PREC_FMUL (alpha, zProd[j]));
	}
}

// Force to recompile and optimize with llvm
template<int blockRows,int blockCols>
__global__ void
spgpuSbhdiaspmv_krn_b0 (float *z, const float *y, float alpha, const float* dM, const int* offsets, int hackSize, const int* hackOffsets, int rows, int cols, const float *x, int hackCount)
{
	spgpuSbhdiaspmv_<blockRows,blockCols>(z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, 0.0f, hackCount);
}


template<int blockRows,int blockCols>
__global__ void
spgpuSbhdiaspmv_krn (float *z, const float *y, float alpha, const float* dM, const int* offsets, int hackSize, const int* hackOffsets, int rows, int cols, const float *x, float beta, int hackCount)
{
	spgpuSbhdiaspmv_<blockRows,blockCols>(z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, beta, hackCount);
}


template<int blockRows,int blockCols>
void
_spgpuSbhdiaspmv (spgpuHandle_t handle, int threadCount, float* z, const float *y, float alpha, 
	const float* dM, const int* offsets, int hackSize, const int* hackOffsets, int rows, int cols,
	const float *x, float beta)
{
	dim3 block (threadCount);
	dim3 grid ((rows + threadCount - 1) / threadCount);

	int hackCount = (rows + hackSize - 1)/hackSize;
	
#ifdef ENABLE_CACHE
	bind_tex_x (x);
#endif

	if (beta != 0.0f)
		spgpuSbhdiaspmv_krn<blockRows,blockCols> <<< grid, block, block.x*sizeof(int), handle->currentStream >>> (z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, beta, hackCount);
	else
		spgpuSbhdiaspmv_krn_b0<blockRows,blockCols> <<< grid, block, block.x*sizeof(int), handle->currentStream >>> (z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, hackCount);

#ifdef ENABLE_CACHE
  	unbind_tex_x (x);
#endif

}

template<int blockRows,int blockCols>
void 
spgpuSbhdiaspmv_ (spgpuHandle_t handle, 
	float* z, 
	const float *y, 
	float alpha, 
	const float* dM, 
	const int* offsets, 
	int hackSize, 
	const int* hackOffsets,
	int rows,
	int cols, 
	const float *x, 
	float beta)
{
	__assert(hackSize % 32 == 0, "Only hacks whose length is a multiple of 32 are supported...");
	
	cudaFuncSetCacheConfig(spgpuSbhdiaspmv_krn<blockRows,blockCols>, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(spgpuSbhdiaspmv_krn_b0<blockRows,blockCols>, cudaFuncCachePreferL1);
	
	cudaDeviceProp deviceProp;
    	cudaGetDeviceProperties(&deviceProp, 0);
    	
    	int threadCount = 128;

	int maxThreadForACall = threadCount*65535;
	
	while (rows > maxThreadForACall) //managing large vectors
	{
		_spgpuSbhdiaspmv<blockRows,blockCols> (handle, threadCount, z, y, alpha, dM, offsets, hackSize, hackOffsets, maxThreadForACall, cols, x, beta);

		y = y + blockRows*maxThreadForACall;
		z = z + blockRows*maxThreadForACall;
		
		hackOffsets += maxThreadForACall/hackSize;
		
		rows -= maxThreadForACall;
	}
	
	_spgpuSbhdiaspmv<blockRows,blockCols> (handle, threadCount, z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, beta);
	
	cudaCheckError("CUDA error on bhdia_dspmv");
}


void 
spgpuSbhdiaspmv (spgpuHandle_t handle, 
	float* z, 
	const float *y, 
	float alpha,
	int blockRows,
	int blockCols, 
	const float* dM, 
	const int* offsets, 
	int hackSize, 
	const int* hackOffsets,
	int rows,
	int cols, 
	const float *x, 
	float beta)
{
	__assert(blockRows == blockCols, "Only square blocks are supported.");
	
	if (blockRows == 1)
		spgpuSbhdiaspmv_<1,1>(handle,z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, beta);
	else if (blockRows == 2)
		spgpuSbhdiaspmv_<2,2>(handle,z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, beta);
	else if (blockRows == 3)
		spgpuSbhdiaspmv_<3,3>(handle,z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, beta);
	else if (blockRows == 4)
		spgpuSbhdiaspmv_<4,4>(handle,z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, beta);
	else
	{
		__assert(0, "Unsupported non zero block size.");
	}
}

