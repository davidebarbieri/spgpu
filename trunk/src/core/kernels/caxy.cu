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

#define BLOCK_SIZE 256
#define MAX_N_FOR_A_CALL (BLOCK_SIZE*65535)

__global__ void spgpuCaxy_kern(cuFloatComplex *z, int n, cuFloatComplex alpha, cuFloatComplex* x, cuFloatComplex* y)
{
	int id = threadIdx.x + BLOCK_SIZE*blockIdx.x;
	
	if (id < n)
	{
		// Since z, x and y are accessed with the same offset by the same thread,
		// and the write to z follows the x and y reads, x, y and z can share the same base address (in-place computing).

		z[id] = cuCmulf(alpha,cuCmulf(x[id],y[id]));
	}
}

void spgpuCaxy_(spgpuHandle_t handle, cuFloatComplex *z, int n, cuFloatComplex alpha, cuFloatComplex* x, cuFloatComplex* y)
{
  int msize = (n+BLOCK_SIZE-1)/BLOCK_SIZE;

  dim3 block(BLOCK_SIZE);
  dim3 grid(msize);

  spgpuCaxy_kern<<<grid, block, 0, handle->currentStream>>>(z, n, alpha, x, y);

}

void spgpuCaxy(spgpuHandle_t handle,
	__device cuFloatComplex *z,
	int n,
	cuFloatComplex alpha,
	__device cuFloatComplex *x,
	__device cuFloatComplex *y)
{

	while (n > MAX_N_FOR_A_CALL) //managing large vectors
    {
		spgpuCaxy_(handle, z, MAX_N_FOR_A_CALL, alpha, x, y);
		x = x + MAX_N_FOR_A_CALL;
		y = y + MAX_N_FOR_A_CALL;
		z = z + MAX_N_FOR_A_CALL;
		n -= MAX_N_FOR_A_CALL;
	}
	
    spgpuCaxy_(handle, z, n, alpha, x, y);
	
	cudaCheckError("CUDA error on saxy");
}

void spgpuCmaxy(spgpuHandle_t handle,
	__device cuFloatComplex *z,
	int n,
	cuFloatComplex alpha,
	__device cuFloatComplex* x,
	__device cuFloatComplex *y,
	int count,
	int pitch)
{
  for (int i=0; i<count; i++)
    {
      spgpuCaxy(handle, z, n, alpha, x, y);
		
      x += pitch;
      y += pitch;
      z += pitch;
    }
}

__global__ void spgpuCaxypbz_kern(cuFloatComplex *w, int n, cuFloatComplex beta, cuFloatComplex* z, cuFloatComplex alpha, cuFloatComplex* x, cuFloatComplex* y)
{
	int id = threadIdx.x + BLOCK_SIZE*blockIdx.x;
	
	if (id < n)
	{
		// Since w, x and y and z are accessed with the same offset by the same thread,
		// and the write to z follows the x, y and z reads, x, y, z and w can share the same base address (in-place computing).
		w[id] = cuCfmaf(beta, z[id], cuCmulf(alpha, cuCmulf(x[id],y[id])));
	}
}


void spgpuCaxypbz_(spgpuHandle_t handle, cuFloatComplex *w, int n, cuFloatComplex beta, cuFloatComplex* z, cuFloatComplex alpha, cuFloatComplex* x, cuFloatComplex* y)
{
  int msize = (n+BLOCK_SIZE-1)/BLOCK_SIZE;

  dim3 block(BLOCK_SIZE);
  dim3 grid(msize);

  spgpuCaxypbz_kern<<<grid, block, 0, handle->currentStream>>>(w, n, beta, z, alpha, x, y);

}

void spgpuCaxypbz(spgpuHandle_t handle,
	__device cuFloatComplex *w,
	int n,
	cuFloatComplex beta,
	__device cuFloatComplex *z,
	cuFloatComplex alpha,
	__device cuFloatComplex* x,
	__device cuFloatComplex *y
	)
{
	if (cuFloatComplex_isZero(alpha)) {
		spgpuCscal(handle, w, n, beta, z);
	}
	else if (cuFloatComplex_isZero(beta)) {
		spgpuCaxy(handle, w, n, alpha, x, y);
	} 
	else {

		while (n > MAX_N_FOR_A_CALL) //managing large vectors
		{
			spgpuCaxypbz_(handle, w, MAX_N_FOR_A_CALL, beta, z, alpha, x, y);
	
			x = x + MAX_N_FOR_A_CALL;
			y = y + MAX_N_FOR_A_CALL;
			z = z + MAX_N_FOR_A_CALL;
			w = w + MAX_N_FOR_A_CALL;
			n -= MAX_N_FOR_A_CALL;
		}
    
		spgpuCaxypbz_(handle, w, n, beta, z, alpha, x, y);
    }	
  
	cudaCheckError("CUDA error on saxypbz");
}



void spgpuCmaxypbz(spgpuHandle_t handle,
	__device cuFloatComplex *w,
	int n,
	cuFloatComplex beta,
	__device cuFloatComplex *z,
	cuFloatComplex alpha,
	__device cuFloatComplex* x,
	__device cuFloatComplex *y,
	int count,
	int pitch)
{
  for (int i=0; i<count; i++)
    {
      spgpuCaxypbz(handle, w, n, beta, z, alpha, x, y);
		
      x += pitch;
      y += pitch;
      z += pitch;
      w += pitch;
    }
}

