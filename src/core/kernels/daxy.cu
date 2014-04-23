/*
 * spGPU - Sparse matrices on GPU library.
 * 
 * Copyright (C) 2010 - 2014 
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

__global__ void spgpuDaxy_kern(double *z, int n, double alpha, double* x, double* y)
{
	int id = threadIdx.x + BLOCK_SIZE*blockIdx.x;
	
	if (id < n)
	{
		// Since z, x and y are accessed with the same offset by the same thread,
		// and the write to z follows the x and y reads, x, y and z can share the same base address (in-place computing).
		z[id] = alpha*x[id]*y[id];
	}
}

void spgpuDaxy_(spgpuHandle_t handle, double *z, int n, double alpha, double* x, double* y)
{
  int msize = (n+BLOCK_SIZE-1)/BLOCK_SIZE;

  dim3 block(BLOCK_SIZE);
  dim3 grid(msize);

  spgpuDaxy_kern<<<grid, block, 0, handle->currentStream>>>(z, n, alpha, x, y);

}

void spgpuDaxy(spgpuHandle_t handle,
	__device double *z,
	int n,
	double alpha,
	__device double *x,
	__device double *y)
{

	while (n > MAX_N_FOR_A_CALL) //managing large vectors
    	{
		spgpuDaxy_(handle, z, MAX_N_FOR_A_CALL, alpha, x, y);
		x = x + MAX_N_FOR_A_CALL;
		y = y + MAX_N_FOR_A_CALL;
		z = z + MAX_N_FOR_A_CALL;
		n -= MAX_N_FOR_A_CALL;
	}
	
   	spgpuDaxy_(handle, z, n, alpha, x, y);
	
	cudaCheckError("CUDA error on daxy");
}

void spgpuDmaxy(spgpuHandle_t handle,
	__device double *z,
	int n,
	double alpha,
	__device double* x,
	__device double *y,
	int count,
	int pitch)
{
  for (int i=0; i<count; i++)
    {
      spgpuDaxy(handle, z, n, alpha, x, y);
		
      x += pitch;
      y += pitch;
      z += pitch;
    }
}

__global__ void spgpuDaxypbz_kern(double *w, int n, double beta, double* z, double alpha, double* x, double* y)
{
	int id = threadIdx.x + BLOCK_SIZE*blockIdx.x;
	
	if (id < n)
	{
		// Since w, x and y and z are accessed with the same offset by the same thread,
		// and the write to z follows the x, y and z reads, x, y, z and w can share the same base address (in-place computing).
		w[id] = PREC_DADD(PREC_DMUL(beta,z[id]), PREC_DMUL(alpha,PREC_DMUL(x[id],y[id])));
	}
}


void spgpuDaxypbz_(spgpuHandle_t handle, double *w, int n, double beta, double* z, double alpha, double* x, double* y)
{
  int msize = (n+BLOCK_SIZE-1)/BLOCK_SIZE;

  dim3 block(BLOCK_SIZE);
  dim3 grid(msize);

  spgpuDaxypbz_kern<<<grid, block, 0, handle->currentStream>>>(w, n, beta, z, alpha, x, y);

}

void spgpuDaxypbz(spgpuHandle_t handle,
	__device double *w,
	int n,
	double beta,
	__device double *z,
	double alpha,
	__device double* x,
	__device double *y
	)
{
	if (alpha == 0.0) {
		spgpuDscal(handle, w, n, beta, z);
	}
	else if (beta == 0.0) {
		spgpuDaxy(handle, w, n, alpha, x, y);
	} 
	else {

		while (n > MAX_N_FOR_A_CALL) //managing large vectors
		{
			spgpuDaxypbz_(handle, w, MAX_N_FOR_A_CALL, beta, z, alpha, x, y);
	
			x = x + MAX_N_FOR_A_CALL;
			y = y + MAX_N_FOR_A_CALL;
			z = z + MAX_N_FOR_A_CALL;
			w = w + MAX_N_FOR_A_CALL;
			n -= MAX_N_FOR_A_CALL;
		}
    
		spgpuDaxypbz_(handle, w, n, beta, z, alpha, x, y);
    }	
  
	cudaCheckError("CUDA error on daxypbz");
}



void spgpuDmaxypbz(spgpuHandle_t handle,
	__device double *w,
	int n,
	double beta,
	__device double *z,
	double alpha,
	__device double* x,
	__device double *y,
	int count,
	int pitch)
{
  for (int i=0; i<count; i++)
    {
      spgpuDaxypbz(handle, w, n, beta, z, alpha, x, y);
		
      x += pitch;
      y += pitch;
      z += pitch;
      w += pitch;
    }
}

