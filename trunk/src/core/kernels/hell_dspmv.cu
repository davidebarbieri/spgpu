/*
 * spGPU - Sparse matrices on GPU library.
 * 
 * Copyright (C) 2010 - 2012 
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
#include "hell.h"
}


#include "debug.h"


#ifdef ENABLE_CACHE
// Texture cache management
texture < int2, 1, cudaReadModeElementType > x_tex;

#define bind_tex_x(x) cudaBindTexture(NULL, x_tex, x)
#define unbind_tex_x(x) cudaUnbindTexture(x_tex)
#endif

#define THREAD_BLOCK 128
#define MAX_N_FOR_A_CALL (THREAD_BLOCK*65535)

#if __CUDA_ARCH__ < 300
extern __shared__ int dynShrMem[]; 
#endif

__device__ void
spgpuDhellspmv_ridx (int i, double yVal, int outRow,
	double *z, const double *y, double alpha, const double* cM, const int* rP, int hackSize, const int* hackOffsets, const int* rS, const int* rIdx, int rows, const double *x, double beta, int baseIndex)
{
	

	double zProd = 0.0;

	rS += i; 
	
	int hackId = i / hackSize;
	int hackLaneId = i % hackSize;

	int hackOffset;
	unsigned int laneId = threadIdx.x % 32;
#if __CUDA_ARCH__ < 300
        // "volatile" used to avoid __syncthreads()
        volatile int* warpHackOffset = dynShrMem;

        unsigned int warpId = threadIdx.x / 32;

        if (laneId == 0)
                warpHackOffset[warpId] = hackOffsets[hackId];

        hackOffset = warpHackOffset[warpId] + hackLaneId;
#else
     	if (laneId == 0)
                hackOffset = hackOffsets[hackId];
        hackOffset = __shfl(hackOffset, 0) + hackLaneId;

#endif

	rP += hackOffset; 
	cM += hackOffset; 

	int rowSize = rS[0];

	for (int j = 0; j < rowSize / 2; j++)
	{
		int pointers1, pointers2;
		double values1, values2;
		
#ifdef ENABLE_CACHE
		int2 fetches1, fetches2;
#else
		double fetches1, fetches2;
#endif
		
		pointers1 = rP[0] - baseIndex;
		rP += hackSize;  
		pointers2 = rP[0] - baseIndex;
		rP += hackSize;

		values1 = cM[0];
		cM += hackSize;
		values2 = cM[0];
		cM += hackSize;

#ifdef ENABLE_CACHE

		fetches1 = tex1Dfetch (x_tex, pointers1);
		fetches2 = tex1Dfetch (x_tex, pointers2);

		// avoid MAD on pre-Fermi
		zProd = PREC_DADD(zProd, PREC_DMUL (values1, __hiloint2double (fetches1.y, fetches1.x)));
		zProd = PREC_DADD(zProd, PREC_DMUL (values2, __hiloint2double (fetches2.y, fetches2.x)));
#else
		fetches1 = x[pointers1];
		fetches2 = x[pointers2];

		// avoid MAD on pre-Fermi
		zProd = PREC_DADD(zProd, PREC_DMUL (values1, fetches1));
		zProd = PREC_DADD(zProd, PREC_DMUL (values2, fetches2));	
#endif
	}

	// odd row size
	if (rowSize % 2)
    {
      int pointer = rP[0] - baseIndex;
      double value = cM[0];

#ifdef ENABLE_CACHE
		int2 fetch;
		fetch = tex1Dfetch (x_tex, pointer);
		zProd = PREC_DADD(zProd, PREC_DMUL (value, __hiloint2double (fetch.y, fetch.x)));
#else
		double fetch;
		fetch = x[pointer];
		zProd = PREC_DADD(zProd, PREC_DMUL (value, fetch));
#endif
    }

	// Since z and y are accessed with the same offset by the same thread,
	// and the write to z follows the y read, y and z can share the same base address (in-place computing).
	if (beta == 0.0)
		z[outRow] = PREC_DMUL(alpha, zProd);
	else
		z[outRow] = PREC_DADD(PREC_DMUL (beta, yVal), PREC_DMUL (alpha, zProd));
}


__global__ void
spgpuDhellspmv_krn_ridx (double *z, const double *y, double alpha, const double* cM, const int* rP, int hackSize, const int* hackOffsets, const int* rS, const int* rIdx, int rows, const double *x, double beta, int baseIndex)
{
	int i = threadIdx.x + blockIdx.x * (THREAD_BLOCK);
	if (i >= rows)
		return;

	int outRow = rIdx[i];
	double yVal;
	if (beta != 0.0)
		yVal = y[outRow];

	spgpuDhellspmv_ridx (i, yVal, outRow,
		z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rIdx, rows, x, beta, baseIndex);
}

__device__ void
spgpuDhellspmv_ (double *z, const double *y, double alpha, const double* cM, const int* rP, int hackSize, const int* hackOffsets, const int* rS, int rows, const double *x, double beta, int baseIndex)
{
	int i = threadIdx.x + blockIdx.x * (THREAD_BLOCK);
	if (i >= rows)
		return;

	double yVal;

	if (beta != 0.0)
		yVal = y[i];

	spgpuDhellspmv_ridx (i, yVal, i,
		z, y, alpha, cM, rP, hackSize, hackOffsets, rS, NULL, rows, x, beta, baseIndex);
}

// Force to recompile and optimize with llvm
__global__ void
spgpuDhellspmv_krn_b0 (double *z, const double *y, double alpha, const double* cM, const int* rP, int hackSize, const int* hackOffsets, const int* rS, int rows, const double *x, int baseIndex)
{
	spgpuDhellspmv_ (z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rows, x, 0.0f, baseIndex);
}

__global__ void
spgpuDhellspmv_krn (double *z, const double *y, double alpha, const double* cM, const int* rP, int hackSize, const int* hackOffsets, const int* rS, int rows, const double *x, double beta, int baseIndex)
{
	spgpuDhellspmv_ (z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rows, x, beta, baseIndex);
}

void
_spgpuDhellspmv (spgpuHandle_t handle, double* z, const double *y, double alpha, const double* cM, const int* rP, int hackSize, const int* hackOffsets, const int* rS, const int* rIdx, int rows, const double *x, double beta, int baseIndex)
{
	dim3 block (THREAD_BLOCK);
	dim3 grid ((rows + THREAD_BLOCK - 1) / THREAD_BLOCK);

        int shrMemSize;
#if __CUDA_ARCH__ < 300
       	int warpsPerBlock = THREAD_BLOCK/handle->warpSize;
        shrMemSize = warpsPerBlock*sizeof(int);
#else
       	shrMemSize = 0;
#endif

#ifdef ENABLE_CACHE
	bind_tex_x ((const int2 *) x);
#endif

	if (rIdx)
		spgpuDhellspmv_krn_ridx <<< grid, block, shrMemSize, handle->currentStream >>> (z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rIdx, rows, x, beta, baseIndex);
	else
	{
		if (beta != 0.0)
			spgpuDhellspmv_krn <<< grid, block, shrMemSize, handle->currentStream >>> (z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rows, x, beta, baseIndex);
		else
			spgpuDhellspmv_krn_b0 <<< grid, block, shrMemSize, handle->currentStream >>> (z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rows, x, baseIndex);
	}

	
#ifdef ENABLE_CACHE
  	unbind_tex_x ((const int2 *) x);
#endif

}



void 
spgpuDhellspmv (spgpuHandle_t handle,
	__device double *z, 
	const __device double *y, 
	double alpha, 
	const __device double* cM, 
	const __device int* rP,
	int hackSize,
	const __device int* hackOffsets, 
	const __device int* rS,
	const __device int* rIdx, 
	int rows, 
	const __device double *x, 
	double beta,
	int baseIndex)
{
	__assert(hackSize % 32 == 0, "Only hacks whose length is a multiple of 32 are supported...");

	
	while (rows > MAX_N_FOR_A_CALL) //managing large vectors
	{
		_spgpuDhellspmv (handle, z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rIdx, MAX_N_FOR_A_CALL, x, beta, baseIndex);

		y = y + MAX_N_FOR_A_CALL;
		z = z + MAX_N_FOR_A_CALL;
		rS = rS + MAX_N_FOR_A_CALL;
		hackOffsets += MAX_N_FOR_A_CALL/hackSize;
		
		rows -= MAX_N_FOR_A_CALL;
	}
	
	_spgpuDhellspmv (handle, z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rIdx, rows, x, beta, baseIndex);
	
	cudaCheckError("CUDA error on hell_dspmv");
}
