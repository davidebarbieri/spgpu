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
#include "cuComplex.h"


extern "C"
{
#include "core.h"
#include "vector.h"
}

#define BLOCK_SIZE 512
#define MAX_N_FOR_A_CALL (BLOCK_SIZE*65535)

// Single Precision Indexed Scatter
__global__ void discat_gpu_kern(cuDoubleComplex* vector, int count, const int* indexes, const cuDoubleComplex* values, int firstIndex, cuDoubleComplex beta)
{
	int id = threadIdx.x + BLOCK_SIZE*blockIdx.x;
	
	if (id < count)
	{	
		int pos = indexes[id]-firstIndex;
		
		if (cuDoubleComplex_isNotZero(beta))
			vector[pos] = cuCfma(beta, vector[pos], values[id]);
		else
			vector[pos] = values[id];
	}
}

// Single Precision Indexed Gather
__global__ void digath_gpu_kern(const cuDoubleComplex* vector, int count, const int* indexes, cuDoubleComplex* values, int firstIndex)
{
	int id = threadIdx.x + BLOCK_SIZE*blockIdx.x;
	
	if (id < count)
	{
		values[id] = vector[indexes[id]-firstIndex];
	}
}




void spgpuZscat_(spgpuHandle_t handle,
	__device cuDoubleComplex* y,
	int xNnz,
	const __device cuDoubleComplex *xValues,
	const __device int *xIndices,
	int xBaseIndex,
	cuDoubleComplex beta)
{
	int msize = (xNnz+BLOCK_SIZE-1)/BLOCK_SIZE;

	dim3 block(BLOCK_SIZE);
	dim3 grid(msize);

	discat_gpu_kern<<<grid, block, 0, handle->currentStream>>>(y, xNnz, xIndices, xValues, xBaseIndex, beta);
}

void spgpuZgath_(spgpuHandle_t handle,
	__device cuDoubleComplex *xValues,
	int xNnz,
	const __device int *xIndices,
	int xBaseIndex,
	const __device cuDoubleComplex* y)
{
	int msize = (xNnz+BLOCK_SIZE-1)/BLOCK_SIZE;

	dim3 block(BLOCK_SIZE);
	dim3 grid(msize);

	digath_gpu_kern<<<grid, block, 0, handle->currentStream>>>(y, xNnz, xIndices, xValues, xBaseIndex);
}


void spgpuZscat(spgpuHandle_t handle,
	__device cuDoubleComplex* y,
	int xNnz,
	const __device cuDoubleComplex *xValues,
	const __device int *xIndices,
	int xBaseIndex,
	cuDoubleComplex beta)
{
	while (xNnz > MAX_N_FOR_A_CALL) //managing large vectors
	{
		spgpuZscat_(handle, y, MAX_N_FOR_A_CALL, xValues, xIndices, xBaseIndex, beta);
	
		xIndices += MAX_N_FOR_A_CALL;
		xValues += MAX_N_FOR_A_CALL;
		xNnz -= MAX_N_FOR_A_CALL;
	}
	
	spgpuZscat_(handle, y, xNnz, xValues, xIndices, xBaseIndex, beta);
}	
	
void spgpuZgath(spgpuHandle_t handle,
	__device cuDoubleComplex *xValues,
	int xNnz,
	const __device int *xIndices,
	int xBaseIndex,
	const __device cuDoubleComplex* y)	
{
	while (xNnz > MAX_N_FOR_A_CALL) //managing large vectors
	{
		spgpuZgath_(handle, xValues, MAX_N_FOR_A_CALL, xIndices, xBaseIndex, y);
	
		xIndices += MAX_N_FOR_A_CALL;
		xValues += MAX_N_FOR_A_CALL;
		xNnz -= MAX_N_FOR_A_CALL;
	}
	
	spgpuZgath_(handle, xValues, xNnz, xIndices, xBaseIndex, y);
}
