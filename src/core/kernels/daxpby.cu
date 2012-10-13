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
#include "vector.h"

extern "C"
{
#include "core.h"
#include "vector.h"
}


#include "debug.h"

#define BLOCK_SIZE 512
#define MAX_N_FOR_A_CALL (BLOCK_SIZE*65535)

__global__ void spgpuDaxpby_krn(double *z, int n, double beta, double *y, double alpha, double* x)
{
	int id = threadIdx.x + BLOCK_SIZE*blockIdx.x;
	
	if (id < n)
	{
		if (beta == 0.0)
			z[id] = PREC_FMUL(alpha,x[id]);
		else
			z[id] = PREC_FADD(PREC_FMUL(alpha, x[id]), PREC_FMUL(beta,y[id]));
	}
}


void spgpuDaxpby_(spgpuHandle_t handle,
	__device double *z,
	int n,
	double beta,
	__device double *y,
	double alpha,
	__device double* x)
{
	int msize = (n+BLOCK_SIZE-1)/BLOCK_SIZE;

	dim3 block(BLOCK_SIZE);
	dim3 grid(msize);

	spgpuDaxpby_krn<<<grid, block, 0, handle->currentStream>>>(z, n, beta, y, alpha, x);
}

void spgpuDaxpby(spgpuHandle_t handle,
	__device double *z,
	int n,
	double beta,
	__device double *y,
	double alpha,
	__device double* x)
{
	while (n > MAX_N_FOR_A_CALL) //managing large vectors
	{
		spgpuDaxpby_(handle, z, MAX_N_FOR_A_CALL, beta, y, alpha, x);
		
		x = x + MAX_N_FOR_A_CALL;
		y = y + MAX_N_FOR_A_CALL;
		n -= MAX_N_FOR_A_CALL;
	}
	
	spgpuDaxpby_(handle, z, n, beta, y, alpha, x);

	cudaCheckError("CUDA error on daxpby");
}
