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
#include "cudaprec.h"

//#define USE_CUBLAS

#define BLOCK_SIZE 512

__device__ float reductionResult[128];

__global__ void spgpuSdot_kern(int n, float* x, float* y)
{
	__shared__ float sSum[BLOCK_SIZE];

	float res = 0;

	float* lastX = x + n;

	x += threadIdx.x + blockIdx.x*BLOCK_SIZE;
	y += threadIdx.x + blockIdx.x*BLOCK_SIZE;

	int blockOffset = gridDim.x*BLOCK_SIZE;

	int numSteps = (lastX - x + blockOffset - 1)/blockOffset;

	// prefetching
	for (int j = 0; j < numSteps / 2; j++)
    {
		float x1 = x[0]; x += blockOffset;
		float y1 = y[0]; y += blockOffset;
		float x2 = x[0]; x += blockOffset;
		float y2 = y[0]; y += blockOffset;

		res = PREC_FADD(res, PREC_FMUL(x1,y1));
		res = PREC_FADD(res, PREC_FMUL(x2,y2));

	}

	if (numSteps % 2)
	{
		res = PREC_FADD(res, PREC_FMUL(*x,*y));
	}

	if (threadIdx.x >= 32)
		sSum[threadIdx.x] = res;

	__syncthreads();


	// Start reduction!

	if (threadIdx.x < 32) 
	{
		for (int i=1; i<BLOCK_SIZE/32; ++i)
		{
			res += sSum[i*32 + threadIdx.x];
		}

	//useless (because inter-warp)
	/*
	}
	__syncthreads(); 

	if (threadIdx.x < 32) 
	{
	*/

		volatile float* vsSum = sSum;
		vsSum[threadIdx.x] = res;

		if (threadIdx.x < 16) vsSum[threadIdx.x] += vsSum[threadIdx.x + 16];
		if (threadIdx.x < 8) vsSum[threadIdx.x] += vsSum[threadIdx.x + 8];
		if (threadIdx.x < 4) vsSum[threadIdx.x] += vsSum[threadIdx.x + 4];
		if (threadIdx.x < 2) vsSum[threadIdx.x] += vsSum[threadIdx.x + 2];
	
		if (threadIdx.x == 0)
			reductionResult[blockIdx.x] = vsSum[0] + vsSum[1];
	}
}

float spgpuSdot(spgpuHandle_t handle, int n, float* x, float* y)
{
#ifdef USE_CUBLAS
	return cublasSdot(n,x,1,y,1);
#else
	float res = 0;

	int device;
	cudaGetDevice(&device); 
	struct cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,device);	

	int blocks = min(128, min(prop.multiProcessorCount, (n+BLOCK_SIZE-1)/BLOCK_SIZE));
	
	float tRes[128];

	dotKernel<<<blocks, BLOCK_SIZE, 0, handle->currentStream>>>(d_x, d_y, n);
	cudaMemcpyFromSymbol(&tRes,"reductionResult",blocks*sizeof(float));

	for (int i=0; i<blocks; ++i)
	{
		res += tRes[i];
	}

	cudaCheckError("CUDA error on sdot");
	
	return res;
#endif
}