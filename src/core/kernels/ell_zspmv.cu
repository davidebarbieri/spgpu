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
#include "ell.h"
}


#include "debug.h"


#ifdef ENABLE_CACHE
// Texture cache management
texture < int4, cudaTextureType1D, cudaReadModeElementType > x_tex;

#define bind_tex_x(x) cudaBindTexture(NULL, x_tex, x)
#define unbind_tex_x(x) cudaUnbindTexture(x_tex)
#endif

#define THREAD_BLOCK 128
#define MAX_N_FOR_A_CALL (THREAD_BLOCK*65535)

__device__ void
spgpuZellspmv_ridx (int i, cuDoubleComplex yVal, int outRow,
	cuDoubleComplex *z, const cuDoubleComplex *y, cuDoubleComplex alpha, const cuDoubleComplex* cM, const int* rP, int cMPitch, int rPPitch, const int* rS, int rows, const cuDoubleComplex *x, cuDoubleComplex beta, int baseIndex)
{
	cuDoubleComplex zProd = make_cuDoubleComplex(0.0, 0.0);

	rS += i; rP += i; cM += i;

	int rowSize = rS[0];

	for (int j = 0; j < rowSize / 2; j++)
	{
		int pointers1, pointers2;
		cuDoubleComplex values1, values2;
		cuDoubleComplex fetches1, fetches2;
		
		pointers1 = rP[0] - baseIndex;
		rP += rPPitch;  
		pointers2 = rP[0] - baseIndex;
		rP += rPPitch;  

		values1 = cM[0];
		cM += cMPitch;
		values2 = cM[0];
		cM += cMPitch;

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
spgpuZellspmv_krn_ridx (cuDoubleComplex *z, const cuDoubleComplex *y, cuDoubleComplex alpha, const cuDoubleComplex* cM, const int* rP, int cMPitch, int rPPitch, const int* rS, const int* rIdx, int rows, const cuDoubleComplex *x, cuDoubleComplex beta, int baseIndex)
{
	int i = threadIdx.x + blockIdx.x * (THREAD_BLOCK);
	if (i >= rows)
		return;

	int outRow = rIdx[i];
	cuDoubleComplex yVal;
	if (cuDoubleComplex_isNotZero(beta))
		yVal = y[outRow];

	spgpuZellspmv_ridx (i, yVal, outRow,
		z, y, alpha, cM, rP, cMPitch, rPPitch, rS, rows, x, beta, baseIndex);
}


__device__ void
spgpuZellspmv_ (cuDoubleComplex *z, const cuDoubleComplex *y, cuDoubleComplex alpha, const cuDoubleComplex* cM, const int* rP, int cMPitch, int rPPitch, const int* rS, int rows, const cuDoubleComplex *x, cuDoubleComplex beta, int baseIndex)
{
	int i = threadIdx.x + blockIdx.x * (THREAD_BLOCK);
	if (i >= rows)
		return;

	cuDoubleComplex yVal;

	if (cuDoubleComplex_isNotZero(beta))
		yVal = y[i];

	spgpuZellspmv_ridx (i, yVal, i,
		z, y, alpha, cM, rP, cMPitch, rPPitch, rS, rows, x, beta, baseIndex);
}

// Force to recompile and optimize with llvm
__global__ void
spgpuZellspmv_krn_b0 (cuDoubleComplex *z, const cuDoubleComplex *y, cuDoubleComplex alpha, const cuDoubleComplex* cM, const int* rP, int cMPitch, int rPPitch, const int* rS, int rows, const cuDoubleComplex *x, int baseIndex)
{
	spgpuZellspmv_ (z, y, alpha, cM, rP, cMPitch, rPPitch, rS, rows, x, make_cuDoubleComplex(0.0,0.0), baseIndex);
}

__global__ void
spgpuZellspmv_krn (cuDoubleComplex *z, const cuDoubleComplex *y, cuDoubleComplex alpha, const cuDoubleComplex* cM, const int* rP, int cMPitch, int rPPitch, const int* rS, int rows, const cuDoubleComplex *x, cuDoubleComplex beta, int baseIndex)
{
	spgpuZellspmv_ (z, y, alpha, cM, rP, cMPitch, rPPitch, rS, rows, x, beta, baseIndex);
}

void
_spgpuZellspmv (spgpuHandle_t handle, cuDoubleComplex* z, const cuDoubleComplex *y, cuDoubleComplex alpha, const cuDoubleComplex* cM, const int* rP, int cMPitch, int rPPitch, const int* rS,  
	const __device int* rIdx, int rows, const cuDoubleComplex *x, cuDoubleComplex beta, int baseIndex)
{
	dim3 block (THREAD_BLOCK);
	dim3 grid ((rows + THREAD_BLOCK - 1) / THREAD_BLOCK);

#ifdef ENABLE_CACHE
	bind_tex_x ((const int4 *) x);
#endif

	if (rIdx)
		spgpuZellspmv_krn_ridx <<< grid, block, 0, handle->currentStream >>> (z, y, alpha, cM, rP, cMPitch, rPPitch, rS, rIdx, rows, x, beta, baseIndex);
	else
	{
		if (cuDoubleComplex_isNotZero(beta))
			spgpuZellspmv_krn <<< grid, block, 0, handle->currentStream >>> (z, y, alpha, cM, rP, cMPitch, rPPitch, rS, rows, x, beta, baseIndex);
		else
			spgpuZellspmv_krn_b0 <<< grid, block, 0, handle->currentStream >>> (z, y, alpha, cM, rP, cMPitch, rPPitch, rS, rows, x, baseIndex);
	}

#ifdef ENABLE_CACHE
  	unbind_tex_x ((const int4 *) x);
#endif

}


void 
spgpuZellspmv (spgpuHandle_t handle, 
	cuDoubleComplex* z, 
	const cuDoubleComplex *y, 
	cuDoubleComplex alpha, 
	const cuDoubleComplex* cM, 
	const int* rP, 
	int cMPitch, 
	int rPPitch, 
	const int* rS, 
	const __device int* rIdx, 
	int rows, 
	const cuDoubleComplex *x, 
	cuDoubleComplex beta, 
	int baseIndex)
{
	while (rows > MAX_N_FOR_A_CALL) //managing large vectors
	{
		_spgpuZellspmv (handle, z, y, alpha, cM, rP, cMPitch, rPPitch, rS, rIdx, MAX_N_FOR_A_CALL, x, beta, baseIndex);

		y = y + MAX_N_FOR_A_CALL;
		cM = cM + MAX_N_FOR_A_CALL;
		rP = rP + MAX_N_FOR_A_CALL;
		rS = rS + MAX_N_FOR_A_CALL;
		
		rows -= MAX_N_FOR_A_CALL;
	}
	
	_spgpuZellspmv (handle, z, y, alpha, cM, rP, cMPitch, rPPitch, rS, rIdx, rows, x, beta, baseIndex);
	cudaCheckError("CUDA error on ell_zspmv");
}

