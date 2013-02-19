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


extern "C"
{
#include "core.h"
#include "vector.h"
}


#define BLOCK_SIZE 512
#define MAX_N_FOR_A_CALL (BLOCK_SIZE*65535)

__global__ void spgpuSscal_kern(float *y, int n, float alpha, float* x)
{
	int id = threadIdx.x + BLOCK_SIZE*blockIdx.x;

	// Since x and y are accessed with the same offset by the same thread,
	// and the write to y follows the x read, x and y can share the same base address (in-place computing).	
	if (id < n)
	{
		y[id] = alpha*x[id];
	}
}

void spgpuSscal_(spgpuHandle_t handle, float *y, int n, float alpha, float* x)
{
  int msize = (n+BLOCK_SIZE-1)/BLOCK_SIZE;

  dim3 block(BLOCK_SIZE);
  dim3 grid(msize);

  spgpuSscal_kern<<<grid, block, 0, handle->currentStream>>>(y, n, alpha, x);

}

void spgpuSscal(spgpuHandle_t handle,
	__device float *y,
	int n,
	float alpha,
	__device float *x)
{

	while (n > MAX_N_FOR_A_CALL) //managing large vectors
    {
		spgpuSscal_(handle, y, MAX_N_FOR_A_CALL, alpha, x);
		x = x + MAX_N_FOR_A_CALL;
		y = y + MAX_N_FOR_A_CALL;
		n -= MAX_N_FOR_A_CALL;
	}
	
    spgpuSscal_(handle, y, n, alpha, x);
	
	cudaCheckError("CUDA error on sscal");
}
