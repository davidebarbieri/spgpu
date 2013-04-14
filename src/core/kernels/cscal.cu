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

__global__ void spgpuCscal_kern(cuFloatComplex *y, int n, cuFloatComplex alpha, cuFloatComplex* x)
{
	int id = threadIdx.x + BLOCK_SIZE*blockIdx.x;

	// Since x and y are accessed with the same offset by the same thread,
	// and the write to y follows the x read, x and y can share the same base address (in-place computing).	
	if (id < n)
	{
		y[id] = cuCmulf(alpha,x[id]);
	}
}

void spgpuCscal_(spgpuHandle_t handle, cuFloatComplex *y, int n, cuFloatComplex alpha, cuFloatComplex* x)
{
  int msize = (n+BLOCK_SIZE-1)/BLOCK_SIZE;

  dim3 block(BLOCK_SIZE);
  dim3 grid(msize);

  spgpuCscal_kern<<<grid, block, 0, handle->currentStream>>>(y, n, alpha, x);

}

void spgpuCscal(spgpuHandle_t handle,
	__device cuFloatComplex *y,
	int n,
	cuFloatComplex alpha,
	__device cuFloatComplex *x)
{

	while (n > MAX_N_FOR_A_CALL) //managing large vectors
    {
		spgpuCscal_(handle, y, MAX_N_FOR_A_CALL, alpha, x);
		x = x + MAX_N_FOR_A_CALL;
		y = y + MAX_N_FOR_A_CALL;
		n -= MAX_N_FOR_A_CALL;
	}
	
    spgpuCscal_(handle, y, n, alpha, x);
	
	cudaCheckError("CUDA error on cscal");
}
