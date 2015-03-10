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
	int ellMaxRowSize;

	int *ellRowLengths = (int*)malloc(rowsCount*sizeof(int));

	computeEllRowLenghts(ellRowLengths, &ellMaxRowSize, rowsCount, nonZerosCount, rows, 0);

	int ellPitch = computeEllAllocPitch(rowsCount);

	ellValues = (float*)malloc(ellMaxRowSize*ellPitch*sizeof(float));
	ellIndices = (int*)malloc(ellMaxRowSize*ellPitch*sizeof(int));

	memset((void*)ellValues, 0, ellMaxRowSize*ellPitch*sizeof(float));
	memset((void*)ellIndices, 0, ellMaxRowSize*ellPitch*sizeof(int));

	cooToEll(ellValues, ellIndices, ellPitch, 
		 ellPitch, ellMaxRowSize, 0,
		 rowsCount, nonZerosCount, rows, cols, values, 0, SPGPU_TYPE_FLOAT);

	printf("Conversion complete: ELL format is %i Bytes.\n", ellMaxRowSize*(ellPitch*sizeof(float) + ellPitch*sizeof(int)) + rowsCount*sizeof(int));

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

	cudaMalloc((void**)&devCm, ellMaxRowSize*ellPitch*sizeof(float));
	cudaMalloc((void**)&devRp, ellMaxRowSize*ellPitch*sizeof(int));

	cudaMemcpy(devCm, ellValues, ellMaxRowSize*ellPitch*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devRp, ellIndices, ellMaxRowSize*ellPitch*sizeof(int), cudaMemcpyHostToDevice);

	spgpuHandle_t spgpuHandle;
	spgpuCreate(&spgpuHandle, 0);

	struct cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	printf("Computing on %s\n", deviceProp.name);

	printf("Testing ELL format\n");

	spgpuSellspmv (spgpuHandle, devZ, devY, 2.0f, devCm, devRp, ellPitch, ellPitch, devRs, NULL, ellMaxRowSize, ellMaxRowSize, rowsCount, devX, -3.0f, 0);
	

	
	float dotRes = spgpuSdot(spgpuHandle, rowsCount, devZ, devZ);
	cudaDeviceSynchronize();

	printf("dot res: %e\n", dotRes);

	spgpuSellspmv (spgpuHandle, devZ, devY, 2.0f, devCm, devRp, ellPitch, ellPitch, devRs, NULL, ellMaxRowSize, ellMaxRowSize, rowsCount, devX, -3.0f, 0);
		
	cudaDeviceSynchronize();
	
	int hackSize = 32;
	int hellHeight;
	computeHellAllocSize(&hellHeight, hackSize, rowsCount,ellRowLengths);

	float* hellValues = (float*) malloc(hackSize*hellHeight*sizeof(float));
	int* hellIndices = (int*) malloc(hackSize*hellHeight*sizeof(int));
	int* hackOffsets =  (int*) malloc(((rowsCount+hackSize-1)/hackSize)*sizeof(int));

	printf("Converting to HELL format..\n");
	ellToHell(hellValues, hellIndices, hackOffsets, hackSize, ellValues, ellIndices,
		ellPitch, ellPitch, ellRowLengths, rowsCount, SPGPU_TYPE_FLOAT);

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

	spgpuShellspmv (spgpuHandle, devZ, devY, 2.0f, devHellCm, devHellRp, hackSize, devHackOffsets, devRs, NULL, ellMaxRowSize, rowsCount, devX, -3.0f, 0);
	dotRes = spgpuSdot(spgpuHandle, rowsCount, devZ, devZ);
	cudaDeviceSynchronize();
	printf("dot res: %e\n", dotRes);

/*
	printf("Testing Vector functions..\n");
	spgpuSdot(spgpuHandle, rowsCount, devZ, devZ);
	spgpuSmdot(spgpuHandle, devZ,1,devZ,devZ,1,1);
	spgpuSnrm2(spgpuHandle, rowsCount, devZ);
	spgpuSmnrm2(spgpuHandle, devZ, 1, devZ, 1, 1);
	spgpuSscal(spgpuHandle, devZ, 1, 2.0f, devZ);
	spgpuSaxpby(spgpuHandle, devZ, 1, 2.0f, devZ, 1.0f, devZ);
	spgpuSaxy(spgpuHandle, devZ, 1, 3.0f, devZ, devZ);
	spgpuSaxypbz(spgpuHandle, devZ, 1, 4.0f, devZ, 1.0f, devZ, devZ);
	spgpuSmaxy(spgpuHandle, devZ, 1, 2.0f, devZ, devZ, 1, 1);
	spgpuSmaxypbz(spgpuHandle, devZ,1,1.0f,devZ,1.0f,devZ,devZ,1,1);
*/	

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
