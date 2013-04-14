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
texture < int4, 1, cudaReadModeElementType > x_tex;

#define bind_tex_x(x) cudaBindTexture(NULL, x_tex, x)
#define unbind_tex_x(x) cudaUnbindTexture(x_tex)
#endif

#define THREAD_BLOCK 128
#define MAX_N_FOR_A_CALL (THREAD_BLOCK*65535)

extern __shared__ int dynShrMem[]; 

__device__ void
spgpuZhellspmv_ridx (int i, cuDoubleComplex yVal, int outRow,
	cuDoubleComplex *z, const cuDoubleComplex *y, cuDoubleComplex alpha, const cuDoubleComplex* cM, const int* rP, int hackSize, const int* hackOffsets, const int* rS, const int* rIdx, int rows, const cuDoubleComplex *x, cuDoubleComplex beta, int baseIndex)
{
	cuDoubleComplex zProd = make_cuDoubleComplex(0.0, 0.0);

	rS += i; 
	
	int hackId = i / hackSize;
	int hackLaneId = i % hackSize;

	// "volatile" used to avoid __syncthreads()
	volatile int* warpHackOffset = dynShrMem;

	int hackOffset;

	unsigned int laneId = threadIdx.x % 32;
	unsigned int warpId = threadIdx.x / 32;

	if (laneId == 0)
		warpHackOffset[warpId] = hackOffsets[hackId];
	
	hackOffset = warpHackOffset[warpId] + hackLaneId;

	rP += hackOffset; 
	cM += hackOffset; 

	int rowSize = rS[0];

	for (int j = 0; j < rowSize / 2; j++)
	{
		int pointers1, pointers2;
		cuDoubleComplex values1, values2;
		cuDoubleComplex fetches1, fetches2;
		
		pointers1 = rP[0] - baseIndex;
		rP += hackSize;  
		pointers2 = rP[0] - baseIndex;
		rP += hackSize;

		values1 = cM[0];
		cM += hackSize;
		values2 = cM[0];
		cM += hackSize;

#ifdef ENABLE_CACHE
		int4 f1 = tex1Dfetch (x_tex, pointers1);
		int4 f2 = tex1Dfetch (x_tex, pointers2);
		fetches1.x = __hiloint2double (f1.y, f1.x);
		fetches1.y = __hiloint2double (f1.w, f1.z);
		
		fetches2.x = __hiloint2double (f2.y, f2.x);
		fetches2.y = __hiloint2double (f2.w, f2.z);
#else
		fetches1 = x[pointers1];
		fetches2 = x[pointers2];	
#endif

		zProd = cuCfma (values1, fetches1, zProd);
		zProd = cuCfma (values2, fetches2, zProd);	
	}

	// odd row size
	if (rowSize % 2)
   	{
		int pointer = rP[0] - baseIndex;
		cuDoubleComplex value = cM[0];
		
		cuDoubleComplex fetch;
#ifdef ENABLE_CACHE
		int4 f1 = tex1Dfetch (x_tex, pointer);
		fetch.x = __hiloint2double (f1.y, f1.x);
		fetch.y = __hiloint2double (f1.w, f1.z);
#else
		fetch = x[pointer];
#endif
		zProd = cuCfma (value, fetch, zProd);
	}

	// Since z and y are accessed with the same offset by the same thread,
	// and the write to z follows the y read, y and z can share the same base address (in-place computing).
	if (cuDoubleComplex_isNotZero(beta))
		z[outRow] = cuCadd(cuCmul (beta, yVal), cuCmul (alpha, zProd));
	else
		z[outRow] = cuCmul(alpha, zProd);
}


__global__ void
spgpuZhellspmv_krn_ridx (cuDoubleComplex *z, const cuDoubleComplex *y, cuDoubleComplex alpha, const cuDoubleComplex* cM, const int* rP, int hackSize, const int* hackOffsets, const int* rS, const int* rIdx, int rows, const cuDoubleComplex *x, cuDoubleComplex beta, int baseIndex)
{
	int i = threadIdx.x + blockIdx.x * (THREAD_BLOCK);
	if (i >= rows)
		return;

	int outRow = rIdx[i];
	cuDoubleComplex yVal;
	if (cuDoubleComplex_isNotZero(beta))
		yVal = y[outRow];

	spgpuZhellspmv_ridx (i, yVal, outRow,
		z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rIdx, rows, x, beta, baseIndex);
}

__device__ void
spgpuZhellspmv_ (cuDoubleComplex *z, const cuDoubleComplex *y, cuDoubleComplex alpha, const cuDoubleComplex* cM, const int* rP, int hackSize, const int* hackOffsets, const int* rS, int rows, const cuDoubleComplex *x, cuDoubleComplex beta, int baseIndex)
{
	int i = threadIdx.x + blockIdx.x * (THREAD_BLOCK);
	if (i >= rows)
		return;

	cuDoubleComplex yVal;

	if (cuDoubleComplex_isNotZero(beta))
		yVal = y[i];

	spgpuZhellspmv_ridx (i, yVal, i,
		z, y, alpha, cM, rP, hackSize, hackOffsets, rS, NULL, rows, x, beta, baseIndex);
}

// Force to recompile and optimize with llvm
__global__ void
spgpuZhellspmv_krn_b0 (cuDoubleComplex *z, const cuDoubleComplex *y, cuDoubleComplex alpha, const cuDoubleComplex* cM, const int* rP, int hackSize, const int* hackOffsets, const int* rS, int rows, const cuDoubleComplex *x, int baseIndex)
{
	spgpuZhellspmv_ (z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rows, x, make_cuDoubleComplex(0.0,0.0), baseIndex);
}

__global__ void
spgpuZhellspmv_krn (cuDoubleComplex *z, const cuDoubleComplex *y, cuDoubleComplex alpha, const cuDoubleComplex* cM, const int* rP, int hackSize, const int* hackOffsets, const int* rS, int rows, const cuDoubleComplex *x, cuDoubleComplex beta, int baseIndex)
{
	spgpuZhellspmv_ (z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rows, x, beta, baseIndex);
}

void
_spgpuZhellspmv (spgpuHandle_t handle, cuDoubleComplex* z, const cuDoubleComplex *y, cuDoubleComplex alpha, const cuDoubleComplex* cM, const int* rP, int hackSize, const int* hackOffsets, const int* rS, const int* rIdx, int rows, const cuDoubleComplex *x, cuDoubleComplex beta, int baseIndex)
{
	dim3 block (THREAD_BLOCK);
	dim3 grid ((rows + THREAD_BLOCK - 1) / THREAD_BLOCK);

	int warpsPerBlock = THREAD_BLOCK/handle->warpSize;

#ifdef ENABLE_CACHE
	bind_tex_x ((const int4 *) x);
#endif

	if (rIdx)
		spgpuZhellspmv_krn_ridx <<< grid, block, warpsPerBlock*sizeof(int), handle->currentStream >>> (z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rIdx, rows, x, beta, baseIndex);
	else
	{
		if (cuDoubleComplex_isNotZero(beta))
			spgpuZhellspmv_krn <<< grid, block, warpsPerBlock*sizeof(int), handle->currentStream >>> (z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rows, x, beta, baseIndex);
		else
			spgpuZhellspmv_krn_b0 <<< grid, block, warpsPerBlock*sizeof(int), handle->currentStream >>> (z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rows, x, baseIndex);
	}

	
	
#ifdef ENABLE_CACHE
  	unbind_tex_x ((const int4 *) x);
#endif

}



void 
spgpuZhellspmv (spgpuHandle_t handle,
	__device cuDoubleComplex *z, 
	const __device cuDoubleComplex *y, 
	cuDoubleComplex alpha, 
	const __device cuDoubleComplex* cM, 
	const __device int* rP,
	int hackSize,
	const __device int* hackOffsets, 
	const __device int* rS,
	const __device int* rIdx, 
	int rows, 
	const __device cuDoubleComplex *x, 
	cuDoubleComplex beta,
	int baseIndex)
{
	__assert(hackSize % 32 == 0, "Only hacks whose length is a multiple of 32 are supported...");

	
	while (rows > MAX_N_FOR_A_CALL) //managing large vectors
	{
		_spgpuZhellspmv (handle, z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rIdx, MAX_N_FOR_A_CALL, x, beta, baseIndex);

		y = y + MAX_N_FOR_A_CALL;
		rS = rS + MAX_N_FOR_A_CALL;
		hackOffsets += MAX_N_FOR_A_CALL/hackSize;
		
		rows -= MAX_N_FOR_A_CALL;
	}
	
	_spgpuZhellspmv (handle, z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rIdx, rows, x, beta, baseIndex);
	
	cudaCheckError("CUDA error on hell_zspmv");
}
