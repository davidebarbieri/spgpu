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
#include "string.h"

#include "core/dia_conv.h"
#include "core/hdia_conv.h"

#include "core/core.h"
#include "core/dia.h"
#include "core/hdia.h"
#include "vector.h"

#define NUM_TESTS 200

#ifdef TEST_DOUBLE
#define testType double
#else
#define testType float
#endif

void printUsage()
{
	printf("hdiaPerf <input>\n");
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

#ifdef TEST_DOUBLE
	spgpuType_t valuesTypeCode = SPGPU_TYPE_DOUBLE;
#else
	spgpuType_t valuesTypeCode = SPGPU_TYPE_FLOAT;
#endif

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

	 printf("Converting to DIA..\n");

	testType *diaValues;
	int *diaOffsets;
	int diagsCount = computeDiaDiagonalsCount(rowsCount, columnsCount, nonZerosCount, rows, cols);
	
	printf("Diagonals: %i\n", diagsCount);

	int diaPitch = computeDiaAllocPitch(rowsCount);

	diaValues = (testType*)malloc(diagsCount*diaPitch*sizeof(testType));
	diaOffsets = (int*)malloc(diagsCount*sizeof(int));

	memset((void*)diaValues, 0, diagsCount*diaPitch*sizeof(testType));
	memset((void*)diaOffsets, 0, diagsCount*sizeof(int));


	coo2dia(diaValues, diaOffsets, diaPitch, diagsCount, rowsCount,
	columnsCount, nonZerosCount, rows, cols, values, valuesTypeCode);

	printf("Conversion complete: DIA format is %i Bytes.\n", diagsCount*diaPitch*sizeof(testType) + diagsCount*sizeof(int));

	printf("Compute on GPU..\n");

	testType *x = (testType*) malloc(columnsCount*sizeof(testType));
	testType *y = (testType*) malloc(rowsCount*sizeof(testType));

	for (int i=0; i<columnsCount; ++i)
	{
		x[i] = rand()/(testType)RAND_MAX;
	}
	
	for (int i=0; i<rowsCount; ++i)
	{
		y[i] = rand()/(testType)RAND_MAX;
	}
	
	cudaError_t lastError;


	testType *devX, *devY, *devZ;
	testType *devDm;
	int *devOffsets;

	cudaMalloc((void**)&devX,columnsCount*sizeof(testType));
	cudaMalloc((void**)&devY,rowsCount*sizeof(testType));
	cudaMalloc((void**)&devZ,rowsCount*sizeof(testType));
	cudaMalloc((void**)&devOffsets,diagsCount*sizeof(int));

	cudaMemcpy(devX, x, columnsCount*sizeof(testType), cudaMemcpyHostToDevice);
	cudaMemcpy(devY, y, rowsCount*sizeof(testType), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&devDm, diagsCount*diaPitch*sizeof(testType));

	cudaMemcpy(devDm, diaValues, diagsCount*diaPitch*sizeof(testType), cudaMemcpyHostToDevice);
	cudaMemcpy(devOffsets, diaOffsets, diagsCount*sizeof(int), cudaMemcpyHostToDevice);

	spgpuHandle_t spgpuHandle;
	spgpuCreate(&spgpuHandle, 0);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	printf("Computing on %s\n", deviceProp.name);
	Clock timer;

	printf("Testing DIA format\n");


#ifdef TEST_DOUBLE
	spgpuDdiaspmv (spgpuHandle, devZ, devY, 2.0, devDm, devOffsets, diaPitch, rowsCount, columnsCount, diagsCount, devX, -3.0);
#else
	spgpuSdiaspmv (spgpuHandle, devZ, devY, 2.0f, devDm, devOffsets, diaPitch, rowsCount, columnsCount, diagsCount, devX, -3.0f);
#endif
	
	testType dotRes;
	testType time;
	testType gflops;
	testType start;

	
#ifdef TEST_DOUBLE
	dotRes = spgpuDdot(spgpuHandle, rowsCount, devZ, devZ);
#else
	dotRes = spgpuSdot(spgpuHandle, rowsCount, devZ, devZ);
#endif
	cudaDeviceSynchronize();

	printf("dot res: %e\n", dotRes);

	start = timer.getTime();

	for (int i=0; i<NUM_TESTS; ++i)
	{
#ifdef TEST_DOUBLE
		spgpuDdiaspmv (spgpuHandle, devZ, devY, 2.0, devDm, devOffsets, diaPitch, rowsCount, columnsCount, diagsCount, devX, -3.0);
#else
		spgpuSdiaspmv (spgpuHandle, devZ, devY, 2.0f, devDm, devOffsets, diaPitch, rowsCount, columnsCount, diagsCount, devX, -3.0f);
#endif
		
	}
	cudaDeviceSynchronize();

	time = (timer.getTime() - start)/NUM_TESTS;
	printf("elapsed time: %f seconds\n", time);

	gflops = (((nonZerosCount*2-1)) / time)*0.000000001f;
	printf("GFlop/s: %f\n", gflops);

	
	cudaFree(devOffsets);
	cudaFree(devDm);

	printf("Converting to HDIA..\n");

	int hackSize = 32;
	int hacksCount = getHdiaHacksCount(hackSize, rowsCount);

	int allocationHeight;
	int* hackOffsets = (int*)malloc((hacksCount+1)*sizeof(int)); 
	
	computeHdiaHackOffsets(
		&allocationHeight,
		hackOffsets,
		hackSize,
		diaValues,
		diaPitch,	
		diagsCount,
		rowsCount,
		valuesTypeCode);
	
	testType *hdiaValues = (testType*) malloc(hackSize*allocationHeight*sizeof(testType));
	int *hdiaOffsets = (int*) malloc(allocationHeight*sizeof(int));
	
	diaToHdia(
		hdiaValues,
		hdiaOffsets,
		hackOffsets,
		hackSize,
		diaValues,
		diaOffsets,
		diaPitch,	
		diagsCount,
		rowsCount,
		valuesTypeCode
	);
	
	printf("Conversion complete: HDIA format is %i Bytes.\n", hackSize*allocationHeight*sizeof(testType) + 
	(allocationHeight + (hacksCount+1))*sizeof(int));
	
	testType *devHdiaDm;
	int *devHdiaOffsets, *devHackOffsets;
	
	cudaMalloc((void**)&devHdiaDm, hackSize*allocationHeight*sizeof(testType));
	cudaMalloc((void**)&devHdiaOffsets, allocationHeight*sizeof(int));
	cudaMalloc((void**)&devHackOffsets,(hacksCount+1)*sizeof(int));

	cudaMemcpy(devHdiaDm, hdiaValues, hackSize*allocationHeight*sizeof(testType), cudaMemcpyHostToDevice);
	cudaMemcpy(devHdiaOffsets, hdiaOffsets, allocationHeight*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devHackOffsets, hackOffsets, (hacksCount+1)*sizeof(int), cudaMemcpyHostToDevice);

#ifdef TEST_DOUBLE
	spgpuDhdiaspmv (spgpuHandle, devZ, devY, 2.0, devHdiaDm, devHdiaOffsets, hackSize, devHackOffsets, rowsCount, columnsCount, devX, -3.0);
#else
	spgpuShdiaspmv (spgpuHandle, devZ, devY, 2.0f, devHdiaDm, devHdiaOffsets, hackSize, devHackOffsets, rowsCount, columnsCount, devX, -3.0f);
#endif
	
#ifdef TEST_DOUBLE
	dotRes = spgpuDdot(spgpuHandle, rowsCount, devZ, devZ);
#else
	dotRes = spgpuSdot(spgpuHandle, rowsCount, devZ, devZ);
#endif
	cudaDeviceSynchronize();

	printf("dot res: %e\n", dotRes);

	start = timer.getTime();

	for (int i=0; i<NUM_TESTS; ++i)
	{
#ifdef TEST_DOUBLE
		spgpuDhdiaspmv (spgpuHandle, devZ, devY, 2.0, devHdiaDm, devHdiaOffsets, hackSize, devHackOffsets, rowsCount, columnsCount, devX, -3.0);	
#else
		spgpuShdiaspmv (spgpuHandle, devZ, devY, 2.0f, devHdiaDm, devHdiaOffsets, hackSize, devHackOffsets, rowsCount, columnsCount, devX, -3.0f);
#endif
		
	}
	cudaDeviceSynchronize();

	time = (timer.getTime() - start)/NUM_TESTS;
	printf("elapsed time: %f seconds\n", time);

	gflops = (((nonZerosCount*2-1)) / time)*0.000000001f;
	printf("GFlop/s: %f\n", gflops);

	printf("Generating a Blocked HDIA..\n");

	int blockRows = 2;
	int blockCols = 2;
	int blockSize = blockRows*blockCols;
	
	testType *devXB, *devYB, *devZB, *devBhdiaDm;
	
	cudaMalloc((void**)&devBhdiaDm, blockSize*hackSize*allocationHeight*sizeof(testType));
	cudaMalloc((void**)&devXB, blockCols*columnsCount*sizeof(testType));
	cudaMalloc((void**)&devYB, blockRows*rowsCount*sizeof(testType));
	cudaMalloc((void**)&devZB, blockRows*rowsCount*sizeof(testType));
	
	cudaMemset(devBhdiaDm, 0, blockSize*hackSize*allocationHeight*sizeof(testType));
	cudaMemset(devXB, 0, blockCols*columnsCount*sizeof(testType));
	cudaMemset(devYB, 0, blockRows*rowsCount*sizeof(testType));
	cudaMemset(devZB, 0, blockRows*rowsCount*sizeof(testType));
	
	cudaMemcpy2D(devBhdiaDm, blockSize * sizeof(testType), devHdiaDm, sizeof(testType),
		sizeof(testType), hackSize*allocationHeight, cudaMemcpyDeviceToDevice);
	cudaMemcpy2D(devXB, blockCols * sizeof(testType), devX, sizeof(testType),
		sizeof(testType), columnsCount, cudaMemcpyDeviceToDevice);
	cudaMemcpy2D(devYB, blockRows * sizeof(testType), devY, sizeof(testType),
		sizeof(testType), rowsCount, cudaMemcpyDeviceToDevice);

#ifdef TEST_DOUBLE
	spgpuDbhdiaspmv (spgpuHandle, devZB, devYB, 2.0, blockRows, blockCols, devBhdiaDm, devHdiaOffsets, hackSize, devHackOffsets, rowsCount, columnsCount, devXB, -3.0);
#else
	spgpuSbhdiaspmv (spgpuHandle, devZB, devYB, 2.0f, blockRows, blockCols, devBhdiaDm, devHdiaOffsets, hackSize, devHackOffsets, rowsCount, columnsCount, devXB, -3.0f);
#endif
	
#ifdef TEST_DOUBLE
	dotRes = spgpuDdot(spgpuHandle, rowsCount*blockRows, devZB, devZB);
#else
	dotRes = spgpuSdot(spgpuHandle, rowsCount*blockRows, devZB, devZB);
#endif
	cudaDeviceSynchronize();

	printf("dot res: %e\n", dotRes);

	start = timer.getTime();

	for (int i=0; i<NUM_TESTS; ++i)
	{
#ifdef TEST_DOUBLE
		spgpuDbhdiaspmv (spgpuHandle, devZB, devYB, 2.0, blockRows, blockCols, devBhdiaDm, devHdiaOffsets, hackSize, devHackOffsets, rowsCount, columnsCount, devXB, -3.0);	
#else
		spgpuSbhdiaspmv (spgpuHandle, devZB, devYB, 2.0f, blockRows, blockCols, devBhdiaDm, devHdiaOffsets, hackSize, devHackOffsets, rowsCount, columnsCount, devXB, -3.0f);
#endif
		
	}
	cudaDeviceSynchronize();

	time = (timer.getTime() - start)/NUM_TESTS;
	printf("elapsed time: %f seconds\n", time);

	gflops = (((nonZerosCount*2-1)) / time)*0.000000001f;
	printf("GFlop/s: %f\n", gflops);

	spgpuDestroy(spgpuHandle);

	lastError = cudaGetLastError();
	if (lastError != 0)
		printf("Error: %i (%s)\n",lastError,cudaGetErrorString(lastError));


	cudaFree(devX);
	cudaFree(devY);
	cudaFree(devZ);
	cudaFree(devHdiaDm);
	cudaFree(devHdiaOffsets);
	cudaFree(devHackOffsets);

	free(diaValues);
	free(diaOffsets);
	free(hackOffsets);
	free(hdiaValues);
	free(hdiaOffsets);
	free(rows);
	free(cols);
	
	return 0;
}
