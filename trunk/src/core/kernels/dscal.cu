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

__global__ void spgpuDscal_kern(double *y, int n, double alpha, double* x)
{
	int id = threadIdx.x + BLOCK_SIZE*blockIdx.x;
	
	if (id < n)
	{
		y[id] = alpha*x[id];
	}
}

void spgpuDscal_(spgpuHandle_t handle, double *y, int n, double alpha, double* x)
{
  int msize = (n+BLOCK_SIZE-1)/BLOCK_SIZE;

  dim3 block(BLOCK_SIZE);
  dim3 grid(msize);

  spgpuDscal_kern<<<grid, block, 0, handle->currentStream>>>(y, n, alpha, x);

}

void spgpuDscal(spgpuHandle_t handle,
	__device double *y,
	int n,
	double alpha,
	__device double *x)
{

	while (n > MAX_N_FOR_A_CALL) //managing large vectors
    {
		spgpuDscal_(handle, y, MAX_N_FOR_A_CALL, alpha, x);
		x = x + MAX_N_FOR_A_CALL;
		y = y + MAX_N_FOR_A_CALL;
		n -= MAX_N_FOR_A_CALL;
	}
	
    spgpuDscal_(handle, y, n, alpha, x);
	
	cudaCheckError("CUDA error on dscal");
}
