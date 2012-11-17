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
texture < float, 1, cudaReadModeElementType > x_tex;

#define bind_tex_x(x) cudaBindTexture(NULL, x_tex, x)
#define unbind_tex_x(x) cudaUnbindTexture(x_tex)
#endif

#define THREAD_BLOCK 128
#define MAX_N_FOR_A_CALL (THREAD_BLOCK*65535)

__global__ void
spgpuSellspmv_krn (float *z, const float *y, float alpha, const float* cM, const int* rP, int cMPitch, int rPPitch, const int* rS, int rows, const float *x, float beta, int baseIndex)
{
	int i = threadIdx.x + blockIdx.x * (THREAD_BLOCK);

	if (i >= rows)
		return;

	float yVal;
	
	if (beta != 0.0f)
		yVal = y[i];

	float zProd = 0.0f;

	rS += i; rP += i; cM += i;

	int rowSize = rS[0];

	for (int j = 0; j < rowSize / 2; j++)
	{
		int pointers1, pointers2;
		float values1, values2, fetches1, fetches2;
		
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

	if (beta == 0.0f)
		z[i] = PREC_FMUL(alpha, zProd);
	else
		z[i] = PREC_FADD(PREC_FMUL (beta, yVal), PREC_FMUL (alpha, zProd));
}	


void
_spgpuSellspmv (spgpuHandle_t handle, float* z, const float *y, float alpha, const float* cM, const int* rP, int cMPitch, int rPPitch, const int* rS, int rows, const float *x, float beta, int baseIndex)
{
	dim3 block (THREAD_BLOCK);
	dim3 grid ((rows + THREAD_BLOCK - 1) / THREAD_BLOCK);

#ifdef ENABLE_CACHE
	bind_tex_x (x);
#endif

	spgpuSellspmv_krn <<< grid, block, 0, handle->currentStream >>> (z, y, alpha, cM, rP, cMPitch, rPPitch, rS, rows, x, beta, baseIndex);
	
#ifdef ENABLE_CACHE
  	unbind_tex_x (x);
#endif

	cudaCheckError("ERRORE (SSPVM)!");
}

void 
spgpuSellspmv (spgpuHandle_t handle, float* z, const float *y, float alpha, const  float* cM, const  int* rP, int cMPitch, int rPPitch, const  int* rS, int rows, const float *x, float beta, int baseIndex)
{
	while (rows > MAX_N_FOR_A_CALL) //managing large vectors
	{
		_spgpuSellspmv (handle, z, y, alpha, cM, rP, cMPitch, rPPitch, rS, MAX_N_FOR_A_CALL, x, beta, baseIndex);

		y = y + MAX_N_FOR_A_CALL;
		cM = cM + MAX_N_FOR_A_CALL;
		rP = rP + MAX_N_FOR_A_CALL;
		rS = rS + MAX_N_FOR_A_CALL;
		
		rows -= MAX_N_FOR_A_CALL;
	}
	
	_spgpuSellspmv (handle, z, y, alpha, cM, rP, cMPitch, rPPitch, rS, rows, x, beta, baseIndex);
}


