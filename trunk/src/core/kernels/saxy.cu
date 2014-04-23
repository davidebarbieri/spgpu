/*
 * spGPU - Sparse matrices on GPU library.
 * 
 * Copyright (C) 2010 - 2012 
 *     Salvatore Filippone - University of Rome Tor Vergata
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

#include "stdio.h"
#include "cudalang.h"
#include "cudadebug.h"

extern "C"
{
#include "core.h"
#include "vector.h"
}

#define BLOCK_SIZE 256
#define MAX_N_FOR_A_CALL (BLOCK_SIZE*65535)

__global__ void spgpuSaxy_kern(float *z, int n, float alpha, float* x, float* y)
{
	int id = threadIdx.x + BLOCK_SIZE*blockIdx.x;
	
	if (id < n)
	{
		// Since z, x and y are accessed with the same offset by the same thread,
		// and the write to z follows the x and y reads, x, y and z can share the same base address (in-place computing).

		z[id] = alpha*x[id]*y[id];
	}
}

void spgpuSaxy_(spgpuHandle_t handle, float *z, int n, float alpha, float* x, float* y)
{
  int msize = (n+BLOCK_SIZE-1)/BLOCK_SIZE;

  dim3 block(BLOCK_SIZE);
  dim3 grid(msize);

  spgpuSaxy_kern<<<grid, block, 0, handle->currentStream>>>(z, n, alpha, x, y);

}

void spgpuSaxy(spgpuHandle_t handle,
	__device float *z,
	int n,
	float alpha,
	__device float *x,
	__device float *y)
{

	while (n > MAX_N_FOR_A_CALL) //managing large vectors
    {
		spgpuSaxy_(handle, z, MAX_N_FOR_A_CALL, alpha, x, y);
		x = x + MAX_N_FOR_A_CALL;
		y = y + MAX_N_FOR_A_CALL;
		z = z + MAX_N_FOR_A_CALL;
		n -= MAX_N_FOR_A_CALL;
	}
	
    spgpuSaxy_(handle, z, n, alpha, x, y);
	
	cudaCheckError("CUDA error on saxy");
}

void spgpuSmaxy(spgpuHandle_t handle,
	__device float *z,
	int n,
	float alpha,
	__device float* x,
	__device float *y,
	int count,
	int pitch)
{
  for (int i=0; i<count; i++)
    {
      spgpuSaxy(handle, z, n, alpha, x, y);
		
      x += pitch;
      y += pitch;
      z += pitch;
    }
}

__global__ void spgpuSaxypbz_kern(float *w, int n, float beta, float* z, float alpha, float* x, float* y)
{
	int id = threadIdx.x + BLOCK_SIZE*blockIdx.x;
	
	if (id < n)
	{
		// Since w, x and y and z are accessed with the same offset by the same thread,
		// and the write to z follows the x, y and z reads, x, y, z and w can share the same base address (in-place computing).
		w[id] = PREC_FADD(PREC_FMUL(beta,z[id]), PREC_FMUL(alpha,PREC_FMUL(x[id],y[id])));
	}
}


void spgpuSaxypbz_(spgpuHandle_t handle, float *w, int n, float beta, float* z, float alpha, float* x, float* y)
{
  int msize = (n+BLOCK_SIZE-1)/BLOCK_SIZE;

  dim3 block(BLOCK_SIZE);
  dim3 grid(msize);

  spgpuSaxypbz_kern<<<grid, block, 0, handle->currentStream>>>(w, n, beta, z, alpha, x, y);

}

void spgpuSaxypbz(spgpuHandle_t handle,
	__device float *w,
	int n,
	float beta,
	__device float *z,
	float alpha,
	__device float* x,
	__device float *y
	)
{
	if (alpha == 0.0) {
		spgpuSscal(handle, w, n, beta, z);
	}
	else if (beta == 0.0) {
		spgpuSaxy(handle, w, n, alpha, x, y);
	} 
	else {

		while (n > MAX_N_FOR_A_CALL) //managing large vectors
		{
			spgpuSaxypbz_(handle, w, MAX_N_FOR_A_CALL, beta, z, alpha, x, y);
	
			x = x + MAX_N_FOR_A_CALL;
			y = y + MAX_N_FOR_A_CALL;
			z = z + MAX_N_FOR_A_CALL;
			w = w + MAX_N_FOR_A_CALL;
			n -= MAX_N_FOR_A_CALL;
		}
    
		spgpuSaxypbz_(handle, w, n, beta, z, alpha, x, y);
    }	
  
	cudaCheckError("CUDA error on saxypbz");
}



void spgpuSmaxypbz(spgpuHandle_t handle,
	__device float *w,
	int n,
	float beta,
	__device float *z,
	float alpha,
	__device float* x,
	__device float *y,
	int count,
	int pitch)
{
  for (int i=0; i<count; i++)
    {
      spgpuSaxypbz(handle, w, n, beta, z, alpha, x, y);
		
      x += pitch;
      y += pitch;
      z += pitch;
      w += pitch;
    }
}

