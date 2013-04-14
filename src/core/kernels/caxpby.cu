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
#include "vector.h"
}


#include "debug.h"

#define BLOCK_SIZE 512
#define MAX_N_FOR_A_CALL (BLOCK_SIZE*65535)

__global__ void spgpuCaxpby_krn(cuFloatComplex *z, int n, cuFloatComplex beta, cuFloatComplex *y, cuFloatComplex alpha, cuFloatComplex* x)
{
	int id = threadIdx.x + BLOCK_SIZE*blockIdx.x;
	
	if (id < n)
	{
		// Since z, x and y are accessed with the same offset by the same thread,
		// and the write to z follows the x and y read, x, y and z can share the same base address (in-place computing).

		if (cuFloatComplex_isZero(beta))
			z[id] = cuCmulf(alpha,x[id]);
		else
			z[id] = cuCfmaf(beta, y[id], cuCmulf(alpha, x[id]));
	}
}


void spgpuCaxpby_(spgpuHandle_t handle,
	__device cuFloatComplex *z,
	int n,
	cuFloatComplex beta,
	__device cuFloatComplex *y,
	cuFloatComplex alpha,
	__device cuFloatComplex* x)
{
	int msize = (n+BLOCK_SIZE-1)/BLOCK_SIZE;

	dim3 block(BLOCK_SIZE);
	dim3 grid(msize);

	spgpuCaxpby_krn<<<grid, block, 0, handle->currentStream>>>(z, n, beta, y, alpha, x);
}

void spgpuCaxpby(spgpuHandle_t handle,
	__device cuFloatComplex *z,
	int n,
	cuFloatComplex beta,
	__device cuFloatComplex *y,
	cuFloatComplex alpha,
	__device cuFloatComplex* x)
{
	while (n > MAX_N_FOR_A_CALL) //managing large vectors
	{
		spgpuCaxpby_(handle, z, MAX_N_FOR_A_CALL, beta, y, alpha, x);
		
		x = x + MAX_N_FOR_A_CALL;
		y = y + MAX_N_FOR_A_CALL;
		n -= MAX_N_FOR_A_CALL;
	}
	
	spgpuCaxpby_(handle, z, n, beta, y, alpha, x);

	cudaCheckError("CUDA error on saxpby");
}

void spgpuCmaxpby(spgpuHandle_t handle,
		  __device cuFloatComplex *z,
		  int n,
		  cuFloatComplex beta,
		  __device cuFloatComplex *y,
		  cuFloatComplex alpha,
		  __device cuFloatComplex* x, 
		  int count, int pitch)
{

  for (int i=0; i<count; i++)
    spgpuCaxpby(handle, z+pitch*i, n, beta, y+pitch*i, alpha, x+pitch*i);
  
}
