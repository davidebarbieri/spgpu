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

extern "C"
{
#include "core.h"
#include "vector.h"
}

#define BLOCK_SIZE 512

// Single Precision Indexed Scatter
__global__ void discat_gpu_kern(double* vector, int count, const int* indexes, const double* values, int firstIndex, double beta)
{
	int id = threadIdx.x + BLOCK_SIZE*blockIdx.x;
	
	if (id < count)
	{	
		int pos = indexes[id]-firstIndex;
		vector[pos] = beta*vector[pos]+values[id];
	}
}

// Single Precision Indexed Gather
__global__ void digath_gpu_kern(const double* vector, int count, const int* indexes, double* values, int firstIndex)
{
	int id = threadIdx.x + BLOCK_SIZE*blockIdx.x;
	
	if (id < count)
	{
		values[id] = vector[indexes[id]-firstIndex];
	}
}




void spgpuDscat_(spgpuHandle_t handle,
	__device double* y,
	int xNnz,
	const __device double *xValues,
	const __device int *xIndices,
	int xBaseIndex,
	double beta)
{
	int msize = (xNnz+BLOCK_SIZE-1)/BLOCK_SIZE;

	dim3 block(BLOCK_SIZE);
	dim3 grid(msize);

	discat_gpu_kern<<<grid, block, 0, handle->currentStream>>>(y, xNnz, xIndices, xValues, xBaseIndex, beta);
}

void spgpuDgath_(spgpuHandle_t handle,
	__device double *xValues,
	int xNnz,
	const __device int *xIndices,
	int xBaseIndex,
	const __device double* y)
{
	int msize = (xNnz+BLOCK_SIZE-1)/BLOCK_SIZE;

	dim3 block(BLOCK_SIZE);
	dim3 grid(msize);

	digath_gpu_kern<<<grid, block, 0, handle->currentStream>>>(y, xNnz, xIndices, xValues, xBaseIndex);
}


void spgpuDscat(spgpuHandle_t handle,
	__device double* y,
	int xNnz,
	const __device double *xValues,
	const __device int *xIndices,
	int xBaseIndex,
	double beta)
{
	int maxNForACall = max(handle->maxGridSizeX, BLOCK_SIZE*handle->maxGridSizeX);
	while (xNnz > maxNForACall) //managing large vectors
	{
		spgpuDscat_(handle, y, maxNForACall, xValues, xIndices, xBaseIndex, beta);
	
		xIndices += maxNForACall;
		xValues += maxNForACall;
		xNnz -= maxNForACall;
	}
	
	spgpuDscat_(handle, y, xNnz, xValues, xIndices, xBaseIndex, beta);
}	
	
void spgpuDgath(spgpuHandle_t handle,
	__device double *xValues,
	int xNnz,
	const __device int *xIndices,
	int xBaseIndex,
	const __device double* y)	
{
	int maxNForACall = max(handle->maxGridSizeX, BLOCK_SIZE*handle->maxGridSizeX);
	while (xNnz > maxNForACall) //managing large vectors
	{
		spgpuDgath_(handle, xValues, maxNForACall, xIndices, xBaseIndex, y);
	
		xIndices += maxNForACall;
		xValues += maxNForACall;
		xNnz -= maxNForACall;
	}
	
	spgpuDgath_(handle, xValues, xNnz, xIndices, xBaseIndex, y);
}
