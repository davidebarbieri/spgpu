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
#include "cuComplex.h"


extern "C"
{
#include "core.h"
#include "vector.h"
}

#define BLOCK_SIZE 512
#define MAX_N_FOR_A_CALL (BLOCK_SIZE*65535)

__global__ void spgpuZaxy_kern(cuDoubleComplex *z, int n, cuDoubleComplex alpha, cuDoubleComplex* x, cuDoubleComplex* y)
{
	int id = threadIdx.x + BLOCK_SIZE*blockIdx.x;
	
	if (id < n)
	{
		// Since z, x and y are accessed with the same offset by the same thread,
		// and the write to z follows the x and y reads, x, y and z can share the same base address (in-place computing).
		z[id] = cuCmul(alpha, cuCmul(x[id],y[id]));
	}
}

void spgpuZaxy_(spgpuHandle_t handle, cuDoubleComplex *z, int n, cuDoubleComplex alpha, cuDoubleComplex* x, cuDoubleComplex* y)
{
  int msize = (n+BLOCK_SIZE-1)/BLOCK_SIZE;

  dim3 block(BLOCK_SIZE);
  dim3 grid(msize);

  spgpuZaxy_kern<<<grid, block, 0, handle->currentStream>>>(z, n, alpha, x, y);

}

void spgpuZaxy(spgpuHandle_t handle,
	__device cuDoubleComplex *z,
	int n,
	cuDoubleComplex alpha,
	__device cuDoubleComplex *x,
	__device cuDoubleComplex *y)
{

	while (n > MAX_N_FOR_A_CALL) //managing large vectors
    {
		spgpuZaxy_(handle, z, MAX_N_FOR_A_CALL, alpha, x, y);
		x = x + MAX_N_FOR_A_CALL;
		y = y + MAX_N_FOR_A_CALL;
		z = z + MAX_N_FOR_A_CALL;
		n -= MAX_N_FOR_A_CALL;
	}
	
    spgpuZaxy_(handle, z, n, alpha, x, y);
	
	cudaCheckError("CUDA error on daxy");
}

void spgpuZmaxy(spgpuHandle_t handle,
	__device cuDoubleComplex *z,
	int n,
	cuDoubleComplex alpha,
	__device cuDoubleComplex* x,
	__device cuDoubleComplex *y,
	int count,
	int pitch)
{
  for (int i=0; i<count; i++)
    {
      spgpuZaxy(handle, z, n, alpha, x, y);
		
      x += pitch;
      y += pitch;
      z += pitch;
    }
}

__global__ void spgpuZaxypbz_kern(cuDoubleComplex *w, int n, cuDoubleComplex beta, cuDoubleComplex* z, cuDoubleComplex alpha, cuDoubleComplex* x, cuDoubleComplex* y)
{
	int id = threadIdx.x + BLOCK_SIZE*blockIdx.x;
	
	if (id < n)
	{
		// Since w, x and y and z are accessed with the same offset by the same thread,
		// and the write to z follows the x, y and z reads, x, y, z and w can share the same base address (in-place computing).
		w[id] = cuCfma(beta, z[id], cuCmul(alpha, cuCmul(x[id],y[id])));
	}
}


void spgpuZaxypbz_(spgpuHandle_t handle, cuDoubleComplex *w, int n, cuDoubleComplex beta, cuDoubleComplex* z, cuDoubleComplex alpha, cuDoubleComplex* x, cuDoubleComplex* y)
{
  int msize = (n+BLOCK_SIZE-1)/BLOCK_SIZE;

  dim3 block(BLOCK_SIZE);
  dim3 grid(msize);

  spgpuZaxypbz_kern<<<grid, block, 0, handle->currentStream>>>(w, n, beta, z, alpha, x, y);

}

void spgpuZaxypbz(spgpuHandle_t handle,
	__device cuDoubleComplex *w,
	int n,
	cuDoubleComplex beta,
	__device cuDoubleComplex *z,
	cuDoubleComplex alpha,
	__device cuDoubleComplex* x,
	__device cuDoubleComplex *y
	)
{
	if (cuDoubleComplex_isZero(alpha)) {
		spgpuZscal(handle, w, n, beta, z);
	}
	else if (cuDoubleComplex_isZero(beta)) {
		spgpuZaxy(handle, w, n, alpha, x, y);
	} 
	else {

		while (n > MAX_N_FOR_A_CALL) //managing large vectors
		{
			spgpuZaxypbz_(handle, w, MAX_N_FOR_A_CALL, beta, z, alpha, x, y);
	
			x = x + MAX_N_FOR_A_CALL;
			y = y + MAX_N_FOR_A_CALL;
			z = z + MAX_N_FOR_A_CALL;
			n -= MAX_N_FOR_A_CALL;
		}
    
		spgpuZaxypbz_(handle, w, MAX_N_FOR_A_CALL, beta, z, alpha, x, y);
    }	
  
	cudaCheckError("CUDA error on daxypbz");
}



void spgpuZmaxypbz(spgpuHandle_t handle,
	__device cuDoubleComplex *w,
	int n,
	cuDoubleComplex beta,
	__device cuDoubleComplex *z,
	cuDoubleComplex alpha,
	__device cuDoubleComplex* x,
	__device cuDoubleComplex *y,
	int count,
	int pitch)
{
  for (int i=0; i<count; i++)
    {
      spgpuZaxypbz(handle, w, n, beta, z, alpha, x, y);
		
      x += pitch;
      y += pitch;
      z += pitch;
    }
}

