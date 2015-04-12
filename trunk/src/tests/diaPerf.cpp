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
#include "mmutils.hpp"
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

#define ALPHA 1.0
#define BETA 0.0
#define NUM_TESTS 10000

#ifdef TEST_DOUBLE
#define testType double
#else
#define testType float
#endif

void printUsage()
{
	printf("diaPerf <input>\n");
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
	 
	 // Deal with the symmetric property of the matrix
	if (matrixType == MATRIX_TYPE_SYMMETRIC)
	{
		int unfoldedNonZerosCount = 0;
		getUnfoldedMmSymmetricSize(&unfoldedNonZerosCount, values, rows, cols, nonZerosCount);
		
		int *unfoldedRows = (int*) malloc(unfoldedNonZerosCount*sizeof(int));
		int *unfoldedCols = (int*) malloc(unfoldedNonZerosCount*sizeof(int));
		testType *unfoldedValues = (testType*) malloc(unfoldedNonZerosCount*sizeof(testType));
		
		unfoldMmSymmetricReal(unfoldedRows, unfoldedCols, unfoldedValues, rows, cols, values, nonZerosCount);
		
		free(rows);
		free(cols);
		free(values);
		
		nonZerosCount = unfoldedNonZerosCount;
		rows = unfoldedRows;
		cols = unfoldedCols;
		values = unfoldedValues;
	}
	 

	printf("Input matrix is %s:\n", input);
	printf("rows: %i:\n", rowsCount);
	printf("columns: %i\n", columnsCount);
	printf("symmetric: %s\n", matrixType == MATRIX_TYPE_SYMMETRIC ? "true" : "false");
	printf("non zeros: %i\n", nonZerosCount);

	spgpuHandle_t spgpuHandle;
	spgpuCreate(&spgpuHandle, 0);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	printf("Computing on %s\n", deviceProp.name);
	Clock timer;
	
	testType dotRes;
	testType time;
	testType gflops;
	testType start;

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

	cudaMalloc((void**)&devX,columnsCount*sizeof(testType));
	cudaMalloc((void**)&devY,rowsCount*sizeof(testType));
	cudaMalloc((void**)&devZ,rowsCount*sizeof(testType));

	cudaMemcpy(devX, x, columnsCount*sizeof(testType), cudaMemcpyHostToDevice);
	cudaMemcpy(devY, y, rowsCount*sizeof(testType), cudaMemcpyHostToDevice);
	
	printf("Converting to DIA..\n");

	testType *diaValues;
	int *diaOffsets;
	int diagsCount = computeDiaDiagonalsCount(rowsCount, columnsCount, nonZerosCount, rows, cols);
	
	printf("Diagonals: %i\n", diagsCount);

	int diaPitch = computeDiaAllocPitch(rowsCount);

	printf("DIA format needs %li Bytes.\n", (long int)diagsCount*(long int)diaPitch*sizeof(testType) + diagsCount*sizeof(int));

	long int diavsize = (long int)diagsCount*(long int)diaPitch*sizeof(testType);

	size_t freeMem;
	size_t totMem;

	cudaMemGetInfo(&freeMem, &totMem);

	if (diavsize > freeMem)
	{
		diaValues = 0;
		printf("DIA format needs too much memory! (free memory on device: %li MB)\n", freeMem/1000000);
	}
	else
	{
	diaValues = (testType*)malloc(diavsize);
	
	if (diaValues)
	{
		testType *devDm;
		int *devOffsets;
		
		diaOffsets = (int*)malloc(diagsCount*sizeof(int));

		memset((void*)diaValues, 0, diagsCount*diaPitch*sizeof(testType));
		memset((void*)diaOffsets, 0, diagsCount*sizeof(int));

		coo2dia(diaValues, diaOffsets, diaPitch, diagsCount, rowsCount,
		columnsCount, nonZerosCount, rows, cols, values, 0, valuesTypeCode);

		printf("Conversion complete.\n");

		printf("Compute on GPU..\n");
		printf("Testing with alpha = %f and beta = %f\n", (float)ALPHA, (float)BETA);
	
		cudaMalloc((void**)&devOffsets,diagsCount*sizeof(int));
		cudaMalloc((void**)&devDm, diagsCount*diaPitch*sizeof(testType));

		cudaMemcpy(devDm, diaValues, diagsCount*diaPitch*sizeof(testType), cudaMemcpyHostToDevice);
		cudaMemcpy(devOffsets, diaOffsets, diagsCount*sizeof(int), cudaMemcpyHostToDevice);

		printf("Testing DIA format\n");

#ifdef TEST_DOUBLE
		spgpuDdiaspmv (spgpuHandle, devZ, devY, ALPHA, devDm, devOffsets, diaPitch, rowsCount, columnsCount, diagsCount, devX, BETA);
#else
		spgpuSdiaspmv (spgpuHandle, devZ, devY, ALPHA, devDm, devOffsets, diaPitch, rowsCount, columnsCount, diagsCount, devX, (float)BETA);
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
			spgpuDdiaspmv (spgpuHandle, devZ, devY, ALPHA, devDm, devOffsets, diaPitch, rowsCount, columnsCount, diagsCount, devX, BETA);
#else
			spgpuSdiaspmv (spgpuHandle, devZ, devY, (float)ALPHA, devDm, devOffsets, diaPitch, rowsCount, columnsCount, diagsCount, devX, (float)BETA);
#endif		
		}
		cudaDeviceSynchronize();

		time = (timer.getTime() - start)/NUM_TESTS;
		printf("elapsed time: %f seconds\n", time);

		gflops = (((nonZerosCount*2-1)) / time)*0.000000001f;
		printf("GFlop/s: %f\n", gflops);
	
		cudaFree(devOffsets);
		cudaFree(devDm);
	}
	else
		printf("Error on DIA format allocation..\n");
	}

	printf("Converting to HDIA..\n");

	int hackSize = 32;
	int hacksCount = getHdiaHacksCount(hackSize, rowsCount);

	int allocationHeight;
	int* hackOffsets = (int*)malloc((hacksCount+1)*sizeof(int)); 
	
	computeHdiaHackOffsetsFromCoo(
		&allocationHeight,
		hackOffsets,
		hackSize,
		rowsCount,
		columnsCount, 
		nonZerosCount,
		rows, 
		cols,
		0
		);
			
	printf("HDIA format needs %li Bytes.\n", (long int)hackSize*(long int)allocationHeight*sizeof(testType) + (allocationHeight
		+ (hacksCount+1))*sizeof(int));
	
	testType *hdiaValues = (testType*) malloc(hackSize*allocationHeight*sizeof(testType));
	int *hdiaOffsets = (int*) malloc(allocationHeight*sizeof(int));
	
	cooToHdia(
		hdiaValues,
		hdiaOffsets,
		hackOffsets,
		hackSize,
		rowsCount,
		columnsCount,
		nonZerosCount,
		rows,
		cols,
		values,
		0,
		valuesTypeCode
	);
	
	printf("Conversion complete.\n");
		
	testType *devHdiaDm;
	int *devHdiaOffsets, *devHackOffsets;
	
	cudaMalloc((void**)&devHdiaDm, hackSize*allocationHeight*sizeof(testType));
	cudaMalloc((void**)&devHdiaOffsets, allocationHeight*sizeof(int));
	cudaMalloc((void**)&devHackOffsets,(hacksCount+1)*sizeof(int));

	cudaMemcpy(devHdiaDm, hdiaValues, hackSize*allocationHeight*sizeof(testType), cudaMemcpyHostToDevice);
	cudaMemcpy(devHdiaOffsets, hdiaOffsets, allocationHeight*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devHackOffsets, hackOffsets, (hacksCount+1)*sizeof(int), cudaMemcpyHostToDevice);

#ifdef TEST_DOUBLE
	spgpuDhdiaspmv (spgpuHandle, devZ, devY, ALPHA, devHdiaDm, devHdiaOffsets, hackSize, devHackOffsets, rowsCount, columnsCount, devX, BETA);
#else
	spgpuShdiaspmv (spgpuHandle, devZ, devY, (float)ALPHA, devHdiaDm, devHdiaOffsets, hackSize, devHackOffsets, rowsCount, columnsCount, devX, (float)BETA);
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
		spgpuDhdiaspmv (spgpuHandle, devZ, devY, ALPHA, devHdiaDm, devHdiaOffsets, hackSize, devHackOffsets, rowsCount, columnsCount, devX, BETA);	
#else
		spgpuShdiaspmv (spgpuHandle, devZ, devY, (float)ALPHA, devHdiaDm, devHdiaOffsets, hackSize, devHackOffsets, rowsCount, columnsCount, devX, (float)BETA);
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

	if (diaValues)
	{
		free(diaValues);
		free(diaOffsets);
	}

	free(hackOffsets);
	free(hdiaValues);
	free(hdiaOffsets);
	free(rows);
	free(cols);
	
	return 0;
}
