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


//#define USE_CUBLAS

#define BLOCK_SIZE 512

__device__ double reductionResult[128];

__global__ void spgpuDnrm2_kern(int n, double* x)
{
	__shared__ double sSum[BLOCK_SIZE];

	double res = 0;

	double* lastX = x + n;

	x += threadIdx.x + blockIdx.x*BLOCK_SIZE;

	int blockOffset = gridDim.x*BLOCK_SIZE;

	while (x < lastX)
    {
		double x1 = x[0];
		res = PREC_DADD(res, PREC_DMUL(x1, x1));
		
		x += blockOffset;

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

		volatile double* vsSum = sSum;
		vsSum[threadIdx.x] = res;

		if (threadIdx.x < 16) vsSum[threadIdx.x] += vsSum[threadIdx.x + 16];
		if (threadIdx.x < 8) vsSum[threadIdx.x] += vsSum[threadIdx.x + 8];
		if (threadIdx.x < 4) vsSum[threadIdx.x] += vsSum[threadIdx.x + 4];
		if (threadIdx.x < 2) vsSum[threadIdx.x] += vsSum[threadIdx.x + 2];
	
		if (threadIdx.x == 0)
			reductionResult[blockIdx.x] = vsSum[0] + vsSum[1];
	}
}

double spgpuDnrm2(spgpuHandle_t handle, int n, double* x)
{
#ifdef USE_CUBLAS
	double res;
	cublasDnrm2(n,x,1,&res);
	cudaDeviceSynchronize();
	
	return res;

#else
	double res = 0;

	int device;
	cudaGetDevice(&device); 
	struct cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,device);	

	int blocks = min(128, min(prop.multiProcessorCount, (n+BLOCK_SIZE-1)/BLOCK_SIZE));
	
	double tRes[128];

	spgpuDnrm2_kern<<<blocks, BLOCK_SIZE, 0, handle->currentStream>>>(n, x);;
	cudaMemcpyFromSymbol(&tRes,"reductionResult",blocks*sizeof(double));

	for (int i=0; i<blocks; ++i)
	{
		res += tRes[i];
	}

	cudaCheckError("CUDA error on dnrm2");
	
	return sqrt(res);
#endif
}

void spgpuDmnrm2(spgpuHandle_t handle, double *y, int n, __device double *x, int count, int pitch)
{
	for (int i=0; i < count; ++i)
	{
		y[i] = spgpuDnrm2(handle, n, x);
		x += pitch;
	}
}
