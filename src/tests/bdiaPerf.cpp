/*
 * spGPU - Sparse matrices on GPU library.
 * 
 * Copyright (C) 2010 - 2014
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
#include "core/coo_conv.h"

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
	printf("bdiaPerf <block_rows> <block_cols> <input>\n");
	printf("\tinput:\n");
	printf("\t\tMatrix Market format file\n");
}

int main(int argc, char** argv)
{

	if (argc != 4)
	{
		printUsage();
		return 0;
	}
	
	int blockRows = atoi(argv[1]);
	int blockCols = atoi(argv[2]);
	int blockSize = blockRows*blockCols;
	
	const char* input = argv[3];

	FILE* file;

	__assert ((file = fopen(input, "r")) != NULL, "File not found");

	int rowsCount;
	int columnsCount;
	int nonZerosCount;
	bool isStoredSparse;
	int matrixStorage;
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

	printf("COO format needs %li Bytes.\n", 2*nonZerosCount*sizeof(int) 
		+ nonZerosCount*sizeof(testType));
		
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

	int paddedRowsCount = ((rowsCount + blockRows - 1)/blockRows)*blockRows;
	int paddedColumnsCount = ((columnsCount + blockCols - 1)/blockCols)*blockCols;

	testType *x = (testType*) malloc(paddedColumnsCount*sizeof(testType));
	testType *y = (testType*) malloc(paddedRowsCount*sizeof(testType));

	for (int i=0; i<columnsCount; ++i)
	{
		x[i] = rand()/(testType)RAND_MAX;
	}
	
	for (int i=columnsCount; i<paddedColumnsCount; ++i)
		x[i] = 0;
	
	for (int i=0; i<rowsCount; ++i)
	{
		y[i] = rand()/(testType)RAND_MAX;
	}
	
	for (int i=rowsCount; i<paddedRowsCount; ++i)
		y[i] = 0;
	
	cudaError_t lastError;
	
	testType *devX, *devY, *devZ;

	cudaMalloc((void**)&devX,paddedColumnsCount*sizeof(testType));
	cudaMalloc((void**)&devY,paddedRowsCount*sizeof(testType));
	cudaMalloc((void**)&devZ,paddedRowsCount*sizeof(testType));

	cudaMemcpy(devX, x, paddedColumnsCount*sizeof(testType), cudaMemcpyHostToDevice);
	cudaMemcpy(devY, y, paddedRowsCount*sizeof(testType), cudaMemcpyHostToDevice);
	
	printf("Converting to BCOO..\n");
	
	int blockedNonZerosCount = computeBcooSize(blockRows, blockCols, rows, cols, nonZerosCount);

	printf("BCOO format needs %li Bytes.\n", 2*blockedNonZerosCount*sizeof(int) 
		+ blockedNonZerosCount*blockSize*sizeof(testType));
		
	int *bRows = (int*) malloc(blockedNonZerosCount*sizeof(int));
	int *bCols = (int*) malloc(blockedNonZerosCount*sizeof(int));
	testType *bValues = (testType*) malloc(blockedNonZerosCount*blockSize*sizeof(testType));

	memset(bValues, 0, blockedNonZerosCount*blockSize*sizeof(testType));

	// column-major format for blocks
	cooToBcoo(bRows, bCols, bValues, blockRows, blockCols, rows, cols, values, nonZerosCount, valuesTypeCode);
	
	printf("Conversion complete.\n");
	
	printf("Converting to BHDIA..\n");
	
	int blockedRowsCount = paddedRowsCount/blockRows;
	int blockedColsCount = paddedColumnsCount/blockCols;
	
	int hackSize = 32;
	int hacksCount = getHdiaHacksCount(hackSize, blockedRowsCount);

	int allocationHeight;
	int* hackOffsets = (int*)malloc((hacksCount+1)*sizeof(int)); 
	
	computeHdiaHackOffsetsFromCoo(
		&allocationHeight,
		hackOffsets,
		hackSize,
		blockedRowsCount,
		blockedColsCount, 
		blockedNonZerosCount,
		bRows, 
		bCols
		);
			
	printf("BHDIA format needs %li Bytes.\n", hackSize*allocationHeight*blockSize*sizeof(testType) + (allocationHeight
		+ (hacksCount+1))*sizeof(int));
	
	testType *bhdiaValues = (testType*) malloc(hackSize*allocationHeight*blockSize*sizeof(testType));
	int *hdiaOffsets = (int*) malloc(allocationHeight*sizeof(int));
	
	memset(bhdiaValues, 0, hackSize*allocationHeight*blockSize*sizeof(testType));
	
	bcooToBhdia(
		bhdiaValues,
		hdiaOffsets,
		hackOffsets,
		hackSize,
		blockedRowsCount, blockedColsCount,
		blockedNonZerosCount, bRows, bCols, bValues, valuesTypeCode, blockSize);
	
	printf("Conversion complete.\n");
		
	testType *devBhdiaDm;
	int *devHdiaOffsets, *devHackOffsets;
	
	cudaMalloc((void**)&devBhdiaDm, hackSize*allocationHeight*blockSize*sizeof(testType));
	cudaMalloc((void**)&devHdiaOffsets, allocationHeight*sizeof(int));
	cudaMalloc((void**)&devHackOffsets,(hacksCount+1)*sizeof(int));

	cudaMemcpy(devBhdiaDm, bhdiaValues, hackSize*allocationHeight*blockSize*sizeof(testType), cudaMemcpyHostToDevice);
	cudaMemcpy(devHdiaOffsets, hdiaOffsets, allocationHeight*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devHackOffsets, hackOffsets, (hacksCount+1)*sizeof(int), cudaMemcpyHostToDevice);


#ifdef TEST_DOUBLE
	spgpuDbhdiaspmv (spgpuHandle, devZ, devY, 2.0, blockRows, blockCols, devBhdiaDm, devHdiaOffsets, hackSize, devHackOffsets, blockedRowsCount, blockedColsCount, devX, -3.0);
#else
	spgpuSbhdiaspmv (spgpuHandle, devZ, devY, 2.0f, blockRows, blockCols, devBhdiaDm, devHdiaOffsets, hackSize, devHackOffsets, blockedRowsCount, blockedColsCount, devX, -3.0f);
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
		spgpuDbhdiaspmv (spgpuHandle, devZ, devY, 2.0, blockRows, blockCols, devBhdiaDm, devHdiaOffsets, hackSize, devHackOffsets, blockedRowsCount, blockedColsCount, devX, -3.0);	
#else
		spgpuSbhdiaspmv (spgpuHandle, devZ, devY, 2.0f, blockRows, blockCols, devBhdiaDm, devHdiaOffsets, hackSize, devHackOffsets, blockedRowsCount, blockedColsCount, devX, -3.0f);
#endif
		
	}
	cudaDeviceSynchronize();

	time = (timer.getTime() - start)/NUM_TESTS;
	printf("elapsed time: %f seconds\n", time);

	gflops = (((nonZerosCount*2-1)) / time)*0.000000001f;
	printf("GFlop/s: %f\n", gflops);
	printf("GFlop/s: %f (considering block overhead)\n", (((blockedNonZerosCount*blockRows*blockCols*2-1)) / time)*0.000000001f);


	spgpuDestroy(spgpuHandle);

	lastError = cudaGetLastError();
	if (lastError != 0)
		printf("Error: %i (%s)\n",lastError,cudaGetErrorString(lastError));


	cudaFree(devX);
	cudaFree(devY);
	cudaFree(devZ);
	cudaFree(devBhdiaDm);
	cudaFree(devHdiaOffsets);
	cudaFree(devHackOffsets);

	free(hackOffsets);
	free(bhdiaValues);
	free(hdiaOffsets);
	free(bRows);
	free(bCols);
	free(bValues);
	free(rows);
	free(cols);
	free(values);
	
	return 0;
}
