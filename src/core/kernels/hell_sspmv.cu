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
texture < float, 1, cudaReadModeElementType > x_tex;

#define bind_tex_x(x) cudaBindTexture(NULL, x_tex, x)
#define unbind_tex_x(x) cudaUnbindTexture(x_tex)
#endif

#define THREAD_BLOCK 128
#define MAX_N_FOR_A_CALL (THREAD_BLOCK*65535)

extern __shared__ int dynShrMem[]; 

__device__ void
spgpuShellspmv_ridx (int i, float yVal, int outRow,
	float *z, const float *y, float alpha, const float* cM, const int* rP, int hackSize, const int* hackOffsets, const int* rS, const int* rIdx, int rows, const float *x, float beta, int baseIndex)
{
	float zProd = 0.0f;

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
		float values1, values2, fetches1, fetches2;
		
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
#else
		fetches1 = x[pointers1];
		fetches2 = x[pointers2];
#endif

		// avoid MAD on pre-Fermi
		zProd = PREC_FADD(zProd, PREC_FMUL (values1, fetches1));
		zProd = PREC_FADD(zProd, PREC_FMUL (values2, fetches2));	
	}

	// odd row size
	if (rowSize % 2)
    {
      int pointer = rP[0] - baseIndex;
      float value = cM[0];
      float fetch;

#ifdef ENABLE_CACHE
      fetch = tex1Dfetch (x_tex, pointer);
#else
      fetch = x[pointer];
#endif

      zProd = PREC_FADD(zProd, PREC_FMUL (value, fetch));
    }

	// Since z and y are accessed with the same offset by the same thread,
	// and the write to z follows the y read, y and z can share the same base address (in-place computing).
	if (beta == 0.0f)
		z[outRow] = PREC_FMUL(alpha, zProd);
	else
		z[outRow] = PREC_FADD(PREC_FMUL (beta, yVal), PREC_FMUL (alpha, zProd));
}

__global__ void
spgpuShellspmv_krn_ridx (float *z, const float *y, float alpha, const float* cM, const int* rP, int hackSize, const int* hackOffsets, const int* rS, const int* rIdx, int rows, const float *x, float beta, int baseIndex)
{
	int i = threadIdx.x + blockIdx.x * (THREAD_BLOCK);
	if (i >= rows)
		return;

	int outRow = rIdx[i];
	float yVal;
	if (beta != 0.0f)
		yVal = y[outRow];

	spgpuShellspmv_ridx (i, yVal, outRow,
		z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rIdx, rows, x, beta, baseIndex);
}

__device__ void
spgpuShellspmv_ (float *z, const float *y, float alpha, const float* cM, const int* rP, int hackSize, const int* hackOffsets, const int* rS, int rows, const float *x, float beta, int baseIndex)
{
	int i = threadIdx.x + blockIdx.x * (THREAD_BLOCK);
	if (i >= rows)
		return;

	float yVal;

	if (beta != 0.0f)
		yVal = y[i];

	spgpuShellspmv_ridx (i, yVal, i,
		z, y, alpha, cM, rP, hackSize, hackOffsets, rS, NULL, rows, x, beta, baseIndex);
}

// Force to recompile and optimize with llvm
__global__ void
spgpuShellspmv_krn_b0 (float *z, const float *y, float alpha, const float* cM, const int* rP, int hackSize, const int* hackOffsets, const int* rS, int rows, const float *x, int baseIndex)
{
	spgpuShellspmv_ (z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rows, x, 0.0f, baseIndex);
}

__global__ void
spgpuShellspmv_krn (float *z, const float *y, float alpha, const float* cM, const int* rP, int hackSize, const int* hackOffsets, const int* rS, int rows, const float *x, float beta, int baseIndex)
{
	spgpuShellspmv_ (z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rows, x, beta, baseIndex);
}

void
_spgpuShellspmv (spgpuHandle_t handle, float* z, const float *y, float alpha, const float* cM, const int* rP, int hackSize, const int* hackOffsets, const int* rS, const int* rIdx, int rows, const float *x, float beta, int baseIndex)
{
	dim3 block (THREAD_BLOCK);
	dim3 grid ((rows + THREAD_BLOCK - 1) / THREAD_BLOCK);

	int warpsPerBlock = THREAD_BLOCK/handle->warpSize;

#ifdef ENABLE_CACHE
	bind_tex_x (x);
#endif

	
	if (rIdx)
		spgpuShellspmv_krn_ridx <<< grid, block, warpsPerBlock*sizeof(int), handle->currentStream >>> (z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rIdx, rows, x, beta, baseIndex);
	else
	{
		if (beta != 0.0f)
			spgpuShellspmv_krn <<< grid, block, warpsPerBlock*sizeof(int), handle->currentStream >>> (z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rows, x, beta, baseIndex);
		else
			spgpuShellspmv_krn_b0 <<< grid, block, warpsPerBlock*sizeof(int), handle->currentStream >>> (z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rows, x, baseIndex);
	}
	
#ifdef ENABLE_CACHE
  	unbind_tex_x (x);
#endif

	cudaCheckError("ERRORE (SSPVM)!");
}



void 
spgpuShellspmv (spgpuHandle_t handle,
	__device float *z, 
	const __device float *y, 
	float alpha, 
	const __device float* cM, 
	const __device int* rP,
	int hackSize,
	const __device int* hackOffsets, 
	const __device int* rS,
	const __device int* rIdx, 
	int rows, 
	const __device float *x, 
	float beta,
	int baseIndex)
{
	__assert(hackSize % 32 == 0, "Only hacks whose length is a multiple of 32 are supported...");

	
	while (rows > MAX_N_FOR_A_CALL) //managing large vectors
	{
		_spgpuShellspmv (handle, z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rIdx, MAX_N_FOR_A_CALL, x, beta, baseIndex);

		y = y + MAX_N_FOR_A_CALL;
		rS = rS + MAX_N_FOR_A_CALL;
		hackOffsets += MAX_N_FOR_A_CALL/hackSize;
		
		rows -= MAX_N_FOR_A_CALL;
	}
	
	_spgpuShellspmv (handle, z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rIdx, rows, x, beta, baseIndex);
	
	cudaCheckError("CUDA error on sspmv");
}
