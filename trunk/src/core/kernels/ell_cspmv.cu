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
#include "cuComplex.h"

extern "C"
{
#include "core.h"
#include "ell.h"
}


#include "debug.h"


#ifdef ENABLE_CACHE
// Texture cache management
texture < cuFloatComplex, 1, cudaReadModeElementType > x_tex;

#define bind_tex_x(x) cudaBindTexture(NULL, x_tex, x)
#define unbind_tex_x(x) cudaUnbindTexture(x_tex)
#endif

#define THREAD_BLOCK 128
#define MAX_N_FOR_A_CALL (THREAD_BLOCK*65535)

__device__ void
spgpuCellspmv_ridx (int i, cuFloatComplex yVal, int outRow,
	cuFloatComplex *z, const cuFloatComplex *y, cuFloatComplex alpha, const cuFloatComplex* cM, const int* rP, int cMPitch, int rPPitch, const int* rS, int rows, const cuFloatComplex *x, cuFloatComplex beta, int baseIndex)
{
	cuFloatComplex zProd = make_cuFloatComplex(0.0f, 0.0f);

	rS += i; rP += i; cM += i;

	int rowSize = rS[0];

	for (int j = 0; j < rowSize / 2; j++)
	{
		int pointers1, pointers2;
		cuFloatComplex values1, values2, fetches1, fetches2;
		
		pointers1 = rP[0] - baseIndex;
		rP += rPPitch;  
		pointers2 = rP[0] - baseIndex;
		rP += rPPitch;  

		values1 = cM[0];
		cM += cMPitch;
		values2 = cM[0];
		cM += cMPitch;

#ifdef ENABLE_CACHE
		fetches1 = tex1Dfetch (x_tex, pointers1);
		fetches2 = tex1Dfetch (x_tex, pointers2);
#else
		fetches1 = x[pointers1];
		fetches2 = x[pointers2];
#endif

		zProd = cuCfmaf(values1, fetches1, zProd);
		zProd = cuCfmaf(values2, fetches2, zProd);
	}

	// odd row size
	if (rowSize % 2)
    	{
      		int pointer = rP[0] - baseIndex;
     		cuFloatComplex value = cM[0];
     		cuFloatComplex fetch;

#ifdef ENABLE_CACHE
		fetch = tex1Dfetch (x_tex, pointer);
#else
		fetch = x[pointer];
#endif

		zProd = cuCfmaf(value, fetch, zProd);
	}

	// Since z and y are accessed with the same offset by the same thread,
	// and the write to z follows the y read, y and z can share the same base address (in-place computing).
	
	if (cuFloatComplex_isNotZero(beta))
		z[outRow] = cuCaddf(cuCmulf (beta, yVal), cuCmulf(alpha, zProd));
	else
		z[outRow] = cuCmulf(alpha, zProd);
}	

__global__ void
spgpuCellspmv_krn_ridx (cuFloatComplex *z, const cuFloatComplex *y, cuFloatComplex alpha, const cuFloatComplex* cM, const int* rP, int cMPitch, int rPPitch, const int* rS, const int* rIdx, int rows, const cuFloatComplex *x, cuFloatComplex beta, int baseIndex)
{
	int i = threadIdx.x + blockIdx.x * (THREAD_BLOCK);
	if (i >= rows)
		return;

	int outRow = rIdx[i];
	cuFloatComplex yVal;
	if (cuFloatComplex_isNotZero(beta))
		yVal = y[outRow];

	spgpuCellspmv_ridx (i, yVal, outRow,
		z, y, alpha, cM, rP, cMPitch, rPPitch, rS, rows, x, beta, baseIndex);
}


__device__ void
spgpuCellspmv_ (cuFloatComplex *z, const cuFloatComplex *y, cuFloatComplex alpha, const cuFloatComplex* cM, const int* rP, int cMPitch, int rPPitch, const int* rS, int rows, const cuFloatComplex *x, cuFloatComplex beta, int baseIndex)
{
	int i = threadIdx.x + blockIdx.x * (THREAD_BLOCK);
	if (i >= rows)
		return;

	cuFloatComplex yVal;

	if (cuFloatComplex_isNotZero(beta))
		yVal = y[i];

	spgpuCellspmv_ridx (i, yVal, i,
		z, y, alpha, cM, rP, cMPitch, rPPitch, rS, rows, x, beta, baseIndex);
}

// Force to recompile and optimize with llvm
__global__ void
spgpuCellspmv_krn_b0 (cuFloatComplex *z, const cuFloatComplex *y, cuFloatComplex alpha, const cuFloatComplex* cM, const int* rP, int cMPitch, int rPPitch, const int* rS, int rows, const cuFloatComplex *x, int baseIndex)
{
	spgpuCellspmv_ (z, y, alpha, cM, rP, cMPitch, rPPitch, rS, rows, x, make_cuFloatComplex(0.0f, 0.0f), baseIndex);
}

__global__ void
spgpuCellspmv_krn (cuFloatComplex *z, const cuFloatComplex *y, cuFloatComplex alpha, const cuFloatComplex* cM, const int* rP, int cMPitch, int rPPitch, const int* rS, int rows, const cuFloatComplex *x, cuFloatComplex beta, int baseIndex)
{
	spgpuCellspmv_ (z, y, alpha, cM, rP, cMPitch, rPPitch, rS, rows, x, beta, baseIndex);
}

void
_spgpuCellspmv (spgpuHandle_t handle, cuFloatComplex* z, const cuFloatComplex *y, cuFloatComplex alpha, const cuFloatComplex* cM, const int* rP, int cMPitch, int rPPitch, const int* rS, 
	const __device int* rIdx, int rows, const cuFloatComplex *x, cuFloatComplex beta, int baseIndex)
{
	dim3 block (THREAD_BLOCK);
	dim3 grid ((rows + THREAD_BLOCK - 1) / THREAD_BLOCK);

#ifdef ENABLE_CACHE
	bind_tex_x (x);
#endif

	if (rIdx)
		spgpuCellspmv_krn_ridx <<< grid, block, 0, handle->currentStream >>> (z, y, alpha, cM, rP, cMPitch, rPPitch, rS, rIdx, rows, x, beta, baseIndex);
	else
	{
		if (cuFloatComplex_isNotZero(beta))
			spgpuCellspmv_krn <<< grid, block, 0, handle->currentStream >>> (z, y, alpha, cM, rP, cMPitch, rPPitch, rS, rows, x, beta, baseIndex);
		else
			spgpuCellspmv_krn_b0 <<< grid, block, 0, handle->currentStream >>> (z, y, alpha, cM, rP, cMPitch, rPPitch, rS, rows, x, baseIndex);
	}

#ifdef ENABLE_CACHE
  	unbind_tex_x (x);
#endif

}

void 
spgpuCellspmv (spgpuHandle_t handle, 
	cuFloatComplex* z, 
	const cuFloatComplex *y, 
	cuFloatComplex alpha, 
	const cuFloatComplex* cM, 
	const int* rP, 
	int cMPitch, 
	int rPPitch, 
	const int* rS, 
	const __device int* rIdx, 
	int rows, 
	const cuFloatComplex *x, 
	cuFloatComplex beta, 
	int baseIndex)
{
	while (rows > MAX_N_FOR_A_CALL) //managing large vectors
	{
		_spgpuCellspmv (handle, z, y, alpha, cM, rP, cMPitch, rPPitch, rS, rIdx, MAX_N_FOR_A_CALL, x, beta, baseIndex);

		y = y + MAX_N_FOR_A_CALL;
		cM = cM + MAX_N_FOR_A_CALL;
		rP = rP + MAX_N_FOR_A_CALL;
		rS = rS + MAX_N_FOR_A_CALL;
		
		rows -= MAX_N_FOR_A_CALL;
	}
	
	_spgpuCellspmv (handle, z, y, alpha, cM, rP, cMPitch, rPPitch, rS, rIdx, rows, x, beta, baseIndex);
	
	cudaCheckError("CUDA error on ell_cspmv");
}

