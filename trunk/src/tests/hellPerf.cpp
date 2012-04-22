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

#include "mmread.hpp"
#include "debug.h"
#include "timing.hpp"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "string.h"

#include "core/ell_conv.hpp"
#include "core/hell_conv.hpp"

#include "core/core.h"
#include "core/ell.h"
#include "core/hell.h"

#define NUM_TESTS 2000


//#define TEST_DOUBLE

#ifdef TEST_DOUBLE
#define testType double
#else
#define testType float
#endif

void printUsage()
{
	printf("hellPerf <input>\n");
	printf("\tinput:\n");
	printf("\t\tMatrix Market format file\n");
}

int main(int argc, char** argv)
{

	if (argc != 2)
	{
		printUsage();
		return 0;
	}
	
	const char* input = argv[1];

	FILE* file;

	__assert ((file = fopen(input, "r")) != NULL, "File not found");

	int rowsCount;
	int columnsCount;
	int nonZerosCount;
	bool isStoredSparse;
	int  matrixStorage;
	int matrixType;

	bool res = loadMmProperties(&rowsCount, &columnsCount, &nonZerosCount, 
		&isStoredSparse, &matrixStorage, &matrixType, file);

	 __assert(res, "Error on file read");

	 printf("Input matrix is %s:\n", input);
	 printf("rows: %i:\n", rowsCount);
	 printf("columns: %i\n", columnsCount);
	 printf("non zeros: %i\n", nonZerosCount);

	 printf("Allocating COO matrix..\n");
	 testType* values = (testType*) malloc(nonZerosCount*sizeof(testType));
	 int* rows = (int*) malloc(nonZerosCount*sizeof(int));
	 int* cols = (int*) malloc(nonZerosCount*sizeof(int));

	 printf("Reading matrix from file..\n");

	 int rRes = loadMmMatrixToCoo(values, rows, cols, rowsCount, columnsCount, nonZerosCount, 
		 isStoredSparse, matrixStorage, file);
	 fclose(file);

	 __assert(rRes == MATRIX_READ_SUCCESS, "Error on file read");

	 printf("Converting to ELL..\n");

	testType *ellValues;
	int *ellIndices;
	int ellValuesPitch;
	int ellIndicesPitch;
	int ellMaxRowSize;

	int *ellRowLengths = (int*)malloc(rowsCount*sizeof(int));

	computeEllRowLenghts(ellRowLengths, &ellMaxRowSize, rowsCount, nonZerosCount, rows, 0);
	computeEllAllocPitch<testType>(&ellValuesPitch, &ellIndicesPitch, rowsCount);

	ellValues = (testType*)malloc(ellMaxRowSize*ellValuesPitch);
	ellIndices = (int*)malloc(ellMaxRowSize*ellIndicesPitch);

	memset((void*)ellValues, 0, ellMaxRowSize*ellValuesPitch);
	memset((void*)ellIndices, 0, ellMaxRowSize*ellIndicesPitch);

	cooToEll(ellValues, ellIndices, ellValuesPitch, 
		 ellIndicesPitch, ellMaxRowSize, 0,
		 rowsCount, nonZerosCount, rows, cols, values, 0);

	printf("Conversion complete: ELL format is %i Bytes.\n", ellMaxRowSize*(ellValuesPitch + ellIndicesPitch) + rowsCount*sizeof(int));

	printf("Compute on GPU..\n");

	testType *x = (testType*) malloc(rowsCount*sizeof(testType));
	testType *y = (testType*) malloc(rowsCount*sizeof(testType));

	for (int i=0; i<rowsCount; ++i)
	{
		x[i] = rand()/(testType)RAND_MAX;
		y[i] = rand()/(testType)RAND_MAX;
	}


	testType *devX, *devY, *devZ;
	testType *devCm;
	int *devRp, *devRs;

	cudaMalloc((void**)&devX,rowsCount*sizeof(testType));
	cudaMalloc((void**)&devY,rowsCount*sizeof(testType));
	cudaMalloc((void**)&devZ,rowsCount*sizeof(testType));
	cudaMalloc((void**)&devRs,rowsCount*sizeof(int));

	cudaMemcpy(devX, x, rowsCount*sizeof(testType), cudaMemcpyHostToDevice);
	cudaMemcpy(devY, y, rowsCount*sizeof(testType), cudaMemcpyHostToDevice);
	cudaMemcpy(devRs, ellRowLengths, rowsCount*sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&devCm, ellMaxRowSize*ellValuesPitch);
	cudaMalloc((void**)&devRp, ellMaxRowSize*ellIndicesPitch);

	cudaMemcpy(devCm, ellValues, ellMaxRowSize*ellValuesPitch, cudaMemcpyHostToDevice);
	cudaMemcpy(devRp, ellIndices, ellMaxRowSize*ellIndicesPitch, cudaMemcpyHostToDevice);

	spgpuHandle_t spgpuHandle;
	spgpuCreate(&spgpuHandle, 0);
	printf("Testing ELL format\n");

	Clock timer;

#ifdef TEST_DOUBLE
	spgpuDellspmv (spgpuHandle, devZ, devY, 2.0, devCm, devRp, ellValuesPitch, ellIndicesPitch, devRs, rowsCount, devX, -3.0, 0);
#else
	spgpuSellspmv (spgpuHandle, devZ, devY, 2.0f, devCm, devRp, ellValuesPitch, ellIndicesPitch, devRs, rowsCount, devX, -3.0f, 0);
#endif

	
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

	testType dotRes;
#ifdef TEST_DOUBLE
	cublasDdot(cublasHandle, rowsCount, devZ, 0, devZ, 0, &dotRes);
#else
	cublasSdot(cublasHandle, rowsCount, devZ, 0, devZ, 0, &dotRes);
#endif
	cudaDeviceSynchronize();

	printf("dot res: %e\n", dotRes);

	testType start = timer.getTime();

	for (int i=0; i<NUM_TESTS; ++i)
	{
#ifdef TEST_DOUBLE
		spgpuDellspmv (spgpuHandle, devZ, devY, 2.0, devCm, devRp, ellValuesPitch, ellIndicesPitch, devRs, rowsCount, devX, -3.0, 0);
#else
		spgpuSellspmv (spgpuHandle, devZ, devY, 2.0f, devCm, devRp, ellValuesPitch, ellIndicesPitch, devRs, rowsCount, devX, -3.0f, 0);
#endif
		
	}
	cudaDeviceSynchronize();

	testType time = (timer.getTime() - start)/NUM_TESTS;
	printf("elapsed time: %f seconds\n", time);

	testType gflops = (((nonZerosCount*2-1)) / time)*0.000000001f;
	printf("GFlop/s: %f\n", gflops);

	int hackSize = 32;
	int hellHeight;
	computeHellAllocSize(&hellHeight, hackSize, rowsCount,ellRowLengths);

	testType* hellValues = (testType*) malloc(hackSize*hellHeight*sizeof(testType));
	int* hellIndices = (int*) malloc(hackSize*hellHeight*sizeof(int));
	int* hackOffsets =  (int*) malloc(((rowsCount+hackSize-1)/hackSize)*sizeof(int));

	printf("Converting to HELL format..\n");
	ellToHell(hellValues, hellIndices, hackOffsets, hackSize, ellValues, ellIndices,
		ellValuesPitch, ellIndicesPitch, ellRowLengths, rowsCount);

	printf("Conversion complete: HELL format is %i Bytes.\n", hackSize*hellHeight*(sizeof(testType) + sizeof(int)) + ((rowsCount+hackSize-1)/hackSize)*sizeof(int) + rowsCount*sizeof(int));

	testType* devHellCm;
	int* devHellRp, *devHackOffsets;

	cudaMalloc((void**)&devHellCm, hackSize*hellHeight*sizeof(testType));
	cudaMalloc((void**)&devHellRp, hackSize*hellHeight*sizeof(int));
	cudaMalloc((void**)&devHackOffsets, ((rowsCount+hackSize-1)/hackSize)*sizeof(int));

	cudaMemcpy(devHellCm, hellValues, hackSize*hellHeight*sizeof(testType), cudaMemcpyHostToDevice);
	cudaMemcpy(devHellRp, hellIndices, hackSize*hellHeight*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devHackOffsets, hackOffsets, ((rowsCount+hackSize-1)/hackSize)*sizeof(int), cudaMemcpyHostToDevice);


	printf("Testing HELL format\n");

#ifdef TEST_DOUBLE
	spgpuDhellspmv (spgpuHandle, devZ, devY, 2.0, devHellCm, devHellRp, hackSize, devHackOffsets, devRs, NULL, rowsCount, devX, -3.0, 0);
	cublasDdot(cublasHandle,rowsCount,devZ, 0, devZ, 0, &dotRes);
#else
	spgpuShellspmv (spgpuHandle, devZ, devY, 2.0f, devHellCm, devHellRp, hackSize, devHackOffsets, devRs, NULL, rowsCount, devX, -3.0f, 0);
	cublasSdot(cublasHandle,rowsCount,devZ, 0, devZ, 0, &dotRes);
#endif

	
	cudaDeviceSynchronize();

	printf("dot res: %e\n", dotRes);

	printf("Timing HELL format\n");

	start = timer.getTime();

	for (int i=0; i<NUM_TESTS; ++i)
	{
#ifdef TEST_DOUBLE
		spgpuDhellspmv (spgpuHandle, devZ, devY, 2.0, devHellCm, devHellRp, hackSize, devHackOffsets, devRs, NULL, rowsCount, devX, -3.0, 0);
#else
		spgpuShellspmv (spgpuHandle, devZ, devY, 2.0f, devHellCm, devHellRp, hackSize, devHackOffsets, devRs, NULL, rowsCount, devX, -3.0f, 0);
#endif
	}
	cudaDeviceSynchronize();

	time = (timer.getTime() - start)/NUM_TESTS;
	printf("elapsed time: %f seconds\n", time);

	gflops = (((nonZerosCount*2-1)) / time)*0.000000001f;
	printf("GFlop/s: %f\n", gflops);

	cublasDestroy(cublasHandle);
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
