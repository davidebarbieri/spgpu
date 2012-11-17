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
texture < int2, 1, cudaReadModeElementType > x_tex;

#define bind_tex_x(x) cudaBindTexture(NULL, x_tex, x)
#define unbind_tex_x(x) cudaUnbindTexture(x_tex)
#endif

#define THREAD_BLOCK 128
#define MAX_N_FOR_A_CALL (THREAD_BLOCK*65535)

__global__ void
spgpuDellspmv_krn (double *z, const double *y, double alpha, const double* cM, const int* rP, int cMPitch, int rPPitch, const int* rS, int rows, const double *x, double beta, int baseIndex)
{
	int i = threadIdx.x + blockIdx.x * (THREAD_BLOCK);

	if (i >= rows)
		return;

	double yVal;
	
	if (beta != 0.0)
		yVal = y[i];

	double zProd = 0.0;

	rS += i; rP += i; cM += i;

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

	if (beta == 0.0)
		z[i] = PREC_DMUL(alpha, zProd);
	else
		z[i] = PREC_DADD(PREC_DMUL (beta, yVal), PREC_DMUL (alpha, zProd));
}	


void
_spgpuDellspmv (spgpuHandle_t handle, double* z, const double *y, double alpha, const double* cM, const int* rP, int cMPitch, int rPPitch, const int* rS, int rows, const double *x, double beta, int baseIndex)
{
	dim3 block (THREAD_BLOCK);
	dim3 grid ((rows + THREAD_BLOCK - 1) / THREAD_BLOCK);

#ifdef ENABLE_CACHE
	bind_tex_x ((const int2 *) x);
#endif

	spgpuDellspmv_krn <<< grid, block, 0, handle->currentStream >>> (z, y, alpha, cM, rP, cMPitch, rPPitch, rS, rows, x, beta, baseIndex);
	
#ifdef ENABLE_CACHE
  	unbind_tex_x ((const int2 *) x);
#endif

	cudaCheckError("ERRORE (SSPVM)!");
}


void 
spgpuDellspmv (spgpuHandle_t handle, double* z, const double *y, double alpha, const  double* cM, const  int* rP, int cMPitch, int rPPitch, const  int* rS, int rows, const double *x, double beta, int baseIndex)
{
	while (rows > MAX_N_FOR_A_CALL) //managing large vectors
	{
		_spgpuDellspmv (handle, z, y, alpha, cM, rP, cMPitch, rPPitch, rS, MAX_N_FOR_A_CALL, x, beta, baseIndex);

		y = y + MAX_N_FOR_A_CALL;
		cM = cM + MAX_N_FOR_A_CALL;
		rP = rP + MAX_N_FOR_A_CALL;
		rS = rS + MAX_N_FOR_A_CALL;
		
		rows -= MAX_N_FOR_A_CALL;
	}
	
	_spgpuDellspmv (handle, z, y, alpha, cM, rP, cMPitch, rPPitch, rS, rows, x, beta, baseIndex);
}

