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
 

#include <stdio.h>
#include <stdlib.h>

#include "debug.h"
#include "cuda_runtime.h"

#include "core/ell_conv.h"
#include "core/hell_conv.h"

#include "core/core.h"
#include "core/ell.h"
#include "core/hell.h"
#include "vector.h"

int main(int argc, char** argv)
{

	int rowsCount = 100;
	int columnsCount = 100;
	int nonZerosCount= 200;

	float* values = (float*) malloc(nonZerosCount*sizeof(float));
	int* rows = (int*) malloc(nonZerosCount*sizeof(int));
	int* cols = (int*) malloc(nonZerosCount*sizeof(int));

	int i;
	for (i=0; i<nonZerosCount; ++i)
	{
		rows[i] = i % rowsCount;
		cols[i] = i % columnsCount;
		values[i] = 1.0f;
	}

	printf("Converting to ELL..\n");

	float *ellValues;
	int *ellIndices;
	int ellValuesPitch;
	int ellIndicesPitch;
	int ellMaxRowSize;

	int *ellRowLengths = (int*)malloc(rowsCount*sizeof(int));

	computeEllRowLenghts(ellRowLengths, &ellMaxRowSize, rowsCount, nonZerosCount, rows, 0);
	computeEllAllocPitch(&ellValuesPitch, &ellIndicesPitch, rowsCount, SPGPU_TYPE_FLOAT);

	ellValues = (float*)malloc(ellMaxRowSize*ellValuesPitch);
	ellIndices = (int*)malloc(ellMaxRowSize*ellIndicesPitch);

	memset((void*)ellValues, 0, ellMaxRowSize*ellValuesPitch);
	memset((void*)ellIndices, 0, ellMaxRowSize*ellIndicesPitch);

	cooToEll(ellValues, ellIndices, ellValuesPitch, 
		 ellIndicesPitch, ellMaxRowSize, 0,
		 rowsCount, nonZerosCount, rows, cols, values, 0, SPGPU_TYPE_FLOAT);

	printf("Conversion complete: ELL format is %i Bytes.\n", ellMaxRowSize*(ellValuesPitch + ellIndicesPitch) + rowsCount*sizeof(int));

	printf("Compute on GPU..\n");

	float *x = (float*) malloc(rowsCount*sizeof(float));
	float *y = (float*) malloc(rowsCount*sizeof(float));

	for (i=0; i<rowsCount; ++i)
	{
		x[i] = rand()/(float)RAND_MAX;
		y[i] = rand()/(float)RAND_MAX;
	}


	float *devX, *devY, *devZ;
	float *devCm;
	int *devRp, *devRs;

	cudaMalloc((void**)&devX,rowsCount*sizeof(float));
	cudaMalloc((void**)&devY,rowsCount*sizeof(float));
	cudaMalloc((void**)&devZ,rowsCount*sizeof(float));
	cudaMalloc((void**)&devRs,rowsCount*sizeof(int));

	cudaMemcpy(devX, x, rowsCount*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devY, y, rowsCount*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devRs, ellRowLengths, rowsCount*sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&devCm, ellMaxRowSize*ellValuesPitch);
	cudaMalloc((void**)&devRp, ellMaxRowSize*ellIndicesPitch);

	cudaMemcpy(devCm, ellValues, ellMaxRowSize*ellValuesPitch, cudaMemcpyHostToDevice);
	cudaMemcpy(devRp, ellIndices, ellMaxRowSize*ellIndicesPitch, cudaMemcpyHostToDevice);

	spgpuHandle_t spgpuHandle;
	spgpuCreate(&spgpuHandle, 0);

	struct cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	printf("Computing on %s\n", deviceProp.name);

	printf("Testing ELL format\n");

	spgpuSellspmv (spgpuHandle, devZ, devY, 2.0f, devCm, devRp, ellValuesPitch, ellIndicesPitch, devRs, rowsCount, devX, -3.0f, 0);
	

	
	float dotRes = spgpuSdot(spgpuHandle, rowsCount, devZ, devZ);
	cudaDeviceSynchronize();

	printf("dot res: %e\n", dotRes);

	spgpuSellspmv (spgpuHandle, devZ, devY, 2.0f, devCm, devRp, ellValuesPitch, ellIndicesPitch, devRs, rowsCount, devX, -3.0f, 0);
		
	cudaDeviceSynchronize();
	
	int hackSize = 32;
	int hellHeight;
	computeHellAllocSize(&hellHeight, hackSize, rowsCount,ellRowLengths);

	float* hellValues = (float*) malloc(hackSize*hellHeight*sizeof(float));
	int* hellIndices = (int*) malloc(hackSize*hellHeight*sizeof(int));
	int* hackOffsets =  (int*) malloc(((rowsCount+hackSize-1)/hackSize)*sizeof(int));

	printf("Converting to HELL format..\n");
	ellToHell(hellValues, hellIndices, hackOffsets, hackSize, ellValues, ellIndices,
		ellValuesPitch, ellIndicesPitch, ellRowLengths, rowsCount, SPGPU_TYPE_FLOAT);

	printf("Conversion complete: HELL format is %i Bytes.\n", hackSize*hellHeight*(sizeof(float) + sizeof(int)) + ((rowsCount+hackSize-1)/hackSize)*sizeof(int) + rowsCount*sizeof(int));

	float* devHellCm;
	int* devHellRp, *devHackOffsets;

	cudaMalloc((void**)&devHellCm, hackSize*hellHeight*sizeof(float));
	cudaMalloc((void**)&devHellRp, hackSize*hellHeight*sizeof(int));
	cudaMalloc((void**)&devHackOffsets, ((rowsCount+hackSize-1)/hackSize)*sizeof(int));

	cudaMemcpy(devHellCm, hellValues, hackSize*hellHeight*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devHellRp, hellIndices, hackSize*hellHeight*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devHackOffsets, hackOffsets, ((rowsCount+hackSize-1)/hackSize)*sizeof(int), cudaMemcpyHostToDevice);


	printf("Testing HELL format\n");

	spgpuShellspmv (spgpuHandle, devZ, devY, 2.0f, devHellCm, devHellRp, hackSize, devHackOffsets, devRs, NULL, rowsCount, devX, -3.0f, 0);
	dotRes = spgpuSdot(spgpuHandle, rowsCount, devZ, devZ);
	

	cudaDeviceSynchronize();
	spgpuDestroy(spgpuHandle);

	free(ellRowLengths);
	free(ellValues);
	free(ellIndices);

	cudaThreadSynchronize();
	cudaError_t lastError = cudaGetLastError();
	
	if (lastError != 0)
	{ 
	printf("Error: %i (%s)\n",lastError,cudaGetErrorString(lastError));
	}

	return 0;
}
