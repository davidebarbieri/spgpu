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
texture < int2, 1, cudaReadModeElementType > x_tex;

#define bind_tex_x(x) cudaBindTexture(NULL, x_tex, x)
#define unbind_tex_x(x) cudaUnbindTexture(x_tex)
#endif

template<int blockCols>
__device__ void
spgpuDbhdiaspmv_rows_2 (double2 *z, const double2 *y, double alpha, const double2* dM, const int* offsets, int hackSize, const int* hackOffsets, int rows, int cols, const double2 *x, double beta, int hackCount)
{
	int id = threadIdx.x + blockIdx.x * (blockDim.x);
	
	double2 yVal;

	yVal = make_double2(0.0, 0.0);
			
	if (beta != 0.0)
	{
		yVal = y[id];
	}

	double2 zProd = make_double2(0.0, 0.0);
	
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

	dM += (hackOffset*hackSize + hackLaneId)*blockCols;
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
                			double xBlock[blockCols];
#ifdef ENABLE_CACHE
					#pragma unroll
					for (int k=0; k < blockCols; ++k)
					{
						int2 xValue = tex1Dfetch (x_tex, blockColumn*blockCols + k);
						xBlock[k] = __hiloint2double (xValue.y, xValue.x);
					}					
#else
					#pragma unroll
					for (int k=0; k < blockCols; ++k)
						xBlock[k] = x[blockColumn*blockCols + k];
#endif				
					#pragma unroll					
					for (int k=0; k < blockCols; ++k)
					{
						zProd = PREC_DADD(zProd, PREC_DMUL (dM[k],xBlock[k]));
					}
				}
				
				dM += hackSize*blockCols;
			}
		}
		
		diags -= warpSize;
		offsets += warpSize;
	}


	// Since z and y are accessed with the same offset by the same thread,
	// and the write to z follows the y read, y and z can share the same base address (in-place computing).
	
	if (id >= rows)
		return;
	
	if (beta == 0.0)
	{
		z[id] = PREC_DMUL(alpha, zProd);
	}
	else
	{
		z[id] = PREC_DADD(PREC_DMUL (beta, yVal), PREC_DMUL (alpha, zProd));
	}
}



template<int blockRows,int blockCols>
__device__ void
spgpuDbhdiaspmv_ (double *z, const double *y, double alpha, const double* dM, const int* offsets, int hackSize, const int* hackOffsets, int rows, int cols, const double *x, double beta, int hackCount)
{
	int id = threadIdx.x + blockIdx.x * (blockDim.x);
	
	double yVal[blockRows];

#pragma unroll
	for (int j=0; j<blockRows; ++j)
		yVal[j] = 0.0;
			
	if (beta != 0.0)
	{
		#pragma unroll
		for (int j=0; j<blockRows; ++j)
			yVal[j] = y[id*blockRows + j];
	}

	double zProd[blockRows];
	
	#pragma unroll
	for (int j=0; j<blockRows; ++j)
	{
		zProd[j] = 0.0;
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
	dM += (hackOffset*hackSize + hackLaneId)*blockSize;
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
                			double xBlock[blockCols];
#ifdef ENABLE_CACHE
					#pragma unroll
					for (int k=0; k < blockCols; ++k)
					{
						int2 xValue = tex1Dfetch (x_tex, blockColumn*blockCols + k);
						xBlock[k] = __hiloint2double (xValue.y, xValue.x);
					}					
#else
					#pragma unroll
					for (int k=0; k < blockCols; ++k)
						xBlock[k] = x[blockColumn*blockCols + k];
#endif				
					const double* innerDm = dM;
					#pragma unroll					
					for (int k=0; k < blockCols; ++k)
					#pragma unroll					
						for (int l=0; l < blockRows; ++l)
						{
							zProd[l] = PREC_DADD(zProd[l], PREC_DMUL (innerDm[0],xBlock[k]));
							innerDm = innerDm + 1;
						}
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
	
	if (beta == 0.0)
	{
#pragma unroll
		for (int j=0; j<blockRows; ++j)
			z[id*blockRows + j] = PREC_DMUL(alpha, zProd[j]);
	}
	else
	{
#pragma unroll
		for (int j=0; j<blockRows; ++j)
			z[id*blockRows + j] = PREC_DADD(PREC_DMUL (beta, yVal[j]), PREC_DMUL (alpha, zProd[j]));
	}
}

// Force to recompile and optimize with llvm
template<int blockRows,int blockCols>
__global__ void
spgpuDbhdiaspmv_krn_b0 (double *z, const double *y, double alpha, const double* dM, const int* offsets, int hackSize, const int* hackOffsets, int rows, int cols, const double *x, int hackCount)
{
	spgpuDbhdiaspmv_<blockRows,blockCols>(z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, 0.0, hackCount);
}


template<int blockRows,int blockCols>
__global__ void
spgpuDbhdiaspmv_krn (double *z, const double *y, double alpha, const double* dM, const int* offsets, int hackSize, const int* hackOffsets, int rows, int cols, const double *x, double beta, int hackCount)
{
	spgpuDbhdiaspmv_<blockRows,blockCols>(z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, beta, hackCount);
}

template<int blockCols>
__global__ void
spgpuDbhdiaspmv_krn_b0_rows_2 (double2 *z, const double2 *y, double alpha, const double2* dM, const int* offsets, int hackSize, const int* hackOffsets, int rows, int cols, const double2 *x, int hackCount)
{
	spgpuDbhdiaspmv_rows_2<blockCols>(z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, 0.0, hackCount);
}


template<int blockCols>
__global__ void
spgpuDbhdiaspmv_krn_rows_2 (double2 *z, const double2 *y, double alpha, const double2* dM, const int* offsets, int hackSize, const int* hackOffsets, int rows, int cols, const double2 *x, double beta, int hackCount)
{
	spgpuDbhdiaspmv_rows_2<blockCols>(z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, beta, hackCount);
}

template<int blockRows,int blockCols>
void
_spgpuDbhdiaspmv (spgpuHandle_t handle, int threadCount, double* z, const double *y, double alpha, 
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
		spgpuDbhdiaspmv_krn<blockRows,blockCols> <<< grid, block, block.x*sizeof(int), handle->currentStream >>> (z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, beta, hackCount);
	else
		spgpuDbhdiaspmv_krn_b0<blockRows,blockCols> <<< grid, block, block.x*sizeof(int), handle->currentStream >>> (z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, hackCount);

#ifdef ENABLE_CACHE
  	unbind_tex_x (x);
#endif

}

template<int blockCols>
void
_spgpuDbhdiaspmv_rows_2 (spgpuHandle_t handle, int threadCount, double2 *z, const double2 *y, double alpha, 
	const double2 *dM, const int* offsets, int hackSize, const int* hackOffsets, int rows, int cols,
	const double2 *x, double beta)
{
	dim3 block (threadCount);
	dim3 grid ((rows + threadCount - 1) / threadCount);

	int hackCount = (rows + hackSize - 1)/hackSize;
	
#ifdef ENABLE_CACHE
	bind_tex_x (x);
#endif

	if (beta != 0.0)
		spgpuDbhdiaspmv_krn_rows_2<blockCols> <<< grid, block, block.x*sizeof(int), handle->currentStream >>> (z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, beta, hackCount);
	else
		spgpuDbhdiaspmv_krn_b0_rows_2<blockCols> <<< grid, block, block.x*sizeof(int), handle->currentStream >>> (z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, hackCount);

#ifdef ENABLE_CACHE
  	unbind_tex_x (x);
#endif

}

template<int blockCols>
void 
spgpuDbhdiaspmv_rows_2 (spgpuHandle_t handle, 
	double2 *z, 
	const double2 *y, 
	double alpha, 
	const double2 *dM, 
	const int* offsets, 
	int hackSize, 
	const int* hackOffsets,
	int rows,
	int cols, 
	const double2 *x, 
	double beta)
{
	__assert(hackSize % 32 == 0, "Only hacks whose length is a multiple of 32 are supported...");
	
	cudaFuncSetCacheConfig(spgpuDbhdiaspmv_krn_rows_2<blockCols>, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(spgpuDbhdiaspmv_krn_b0_rows_2<blockCols>, cudaFuncCachePreferL1);
	
	cudaDeviceProp deviceProp;
    	cudaGetDeviceProperties(&deviceProp, 0);
    	
    	int threadCount = 128;

	int maxNForACall = max(handle->maxGridSizeX, threadCount*handle->maxGridSizeX);
	
	while (rows > maxNForACall) //managing large vectors
	{
		_spgpuDbhdiaspmv_rows_2<blockCols> (handle, threadCount, z, y, alpha, dM, offsets, hackSize, hackOffsets, maxNForACall, cols, x, beta);

		y = y + maxNForACall;
		z = z + maxNForACall;
		
		hackOffsets += maxNForACall/hackSize;
		
		rows -= maxNForACall;
	}
	
	_spgpuDbhdiaspmv_rows_2<blockCols> (handle, threadCount, z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, beta);
	
	cudaCheckError("CUDA error on bhdia_dspmv");
}

template<int blockRows,int blockCols>
void 
spgpuDbhdiaspmv_ (spgpuHandle_t handle, 
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
	
	cudaFuncSetCacheConfig(spgpuDbhdiaspmv_krn<blockRows,blockCols>, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(spgpuDbhdiaspmv_krn_b0<blockRows,blockCols>, cudaFuncCachePreferL1);
	
	cudaDeviceProp deviceProp;
    	cudaGetDeviceProperties(&deviceProp, 0);
    	
    	int threadCount = 128;

	int maxNForACall = max(handle->maxGridSizeX, threadCount*handle->maxGridSizeX);
	
	while (rows > maxNForACall) //managing large vectors
	{
		_spgpuDbhdiaspmv<blockRows,blockCols> (handle, threadCount, z, y, alpha, dM, offsets, hackSize, hackOffsets, maxNForACall, cols, x, beta);

		y = y + blockRows*maxNForACall;
		z = z + blockRows*maxNForACall;
		
		hackOffsets += maxNForACall/hackSize;
		
		rows -= maxNForACall;
	}
	
	_spgpuDbhdiaspmv<blockRows,blockCols> (handle, threadCount, z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, beta);
	
	cudaCheckError("CUDA error on bhdia_dspmv");
}


void 
spgpuDbhdiaspmv (spgpuHandle_t handle, 
	double* z, 
	const double *y, 
	double alpha,
	int blockRows,
	int blockCols, 
	const double* dM, 
	const int* offsets, 
	int hackSize, 
	const int* hackOffsets,
	int rows,
	int cols, 
	const double *x, 
	double beta)
{
	if (blockRows == 1)
	{
		if (blockCols == 1)
			spgpuDbhdiaspmv_<1,1>(handle,z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, beta);
		else if (blockCols == 2)
			spgpuDbhdiaspmv_<1,2>(handle,z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, beta);
		else if (blockCols == 3)
			spgpuDbhdiaspmv_<1,3>(handle,z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, beta);
		else if (blockCols == 4)
			spgpuDbhdiaspmv_<1,4>(handle,z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, beta);
		else
			__assert(0, "Unsupported non zero block size.");
	}
	else if (blockRows == 2)
	{
		if (blockCols == 1)
			spgpuDbhdiaspmv_rows_2<1>(handle, (double2*)z, (double2*)y, alpha, (double2*)dM, offsets, hackSize, hackOffsets, rows, cols, (double2*)x, beta);
		else if (blockCols == 2)
			spgpuDbhdiaspmv_rows_2<2>(handle, (double2*)z, (double2*)y, alpha, (double2*)dM, offsets, hackSize, hackOffsets, rows, cols, (double2*)x, beta);
		else if (blockCols == 3)
			spgpuDbhdiaspmv_rows_2<3>(handle, (double2*)z, (double2*)y, alpha, (double2*)dM, offsets, hackSize, hackOffsets, rows, cols, (double2*)x, beta);
		else if (blockCols == 4)
			spgpuDbhdiaspmv_rows_2<4>(handle, (double2*)z, (double2*)y, alpha, (double2*)dM, offsets, hackSize, hackOffsets, rows, cols, (double2*)x, beta);
		else
			__assert(0, "Unsupported non zero block size.");
	}
	else if (blockRows == 3)
	{
		if (blockCols == 1)
			spgpuDbhdiaspmv_<3,1>(handle,z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, beta);
		else if (blockCols == 2)
			spgpuDbhdiaspmv_<3,2>(handle,z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, beta);
		else if (blockCols == 3)
			spgpuDbhdiaspmv_<3,3>(handle,z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, beta);
		else if (blockCols == 4)
			spgpuDbhdiaspmv_<3,4>(handle,z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, beta);
		else
			__assert(0, "Unsupported non zero block size.");
	}
	else if (blockRows == 4)
	{
		if (blockCols == 1)
			spgpuDbhdiaspmv_<4,1>(handle,z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, beta);
		else if (blockCols == 2)
			spgpuDbhdiaspmv_<4,2>(handle,z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, beta);
		else if (blockCols == 3)
			spgpuDbhdiaspmv_<4,3>(handle,z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, beta);
		else if (blockCols == 4)
			spgpuDbhdiaspmv_<4,4>(handle,z, y, alpha, dM, offsets, hackSize, hackOffsets, rows, cols, x, beta);
		else
			__assert(0, "Unsupported non zero block size.");
	} 
	else
	{
		__assert(0, "Unsupported non zero block size.");
	}
}

