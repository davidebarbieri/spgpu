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
#include "cublas_v2.h"
#include "cusparse_v2.h"
#include "string.h"

#include "core/ell_conv.h"
#include "core/hell_conv.h"

#include "core/core.h"
#include "core/ell.h"
#include "core/hell.h"
#include "vector.h"

#define ALPHA 1.0
#define BETA 0.0

#define NUM_TESTS 1000

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



int *app_rows;
int *app_cols;
testType *app_vals;

void merge(int *rows, int *cols, testType *vals, int start, int center, int end, int size) {
	int i, j, k; 

	i = start;
	j = center+1;
	k = 0;
 
	while ((i<=center) && (j<=end)) {
	
		if((rows[i] < rows[j]) ||
	 		((rows[i] == rows[j]) && cols[i] <= cols[j])) {
			app_rows[k] = rows[i];
			app_cols[k] = cols[i];
			app_vals[k] = vals[i];
			
			++k; ++i;
		} else {
			app_rows[k] = rows[j];
			app_cols[k] = cols[j];
			app_vals[k] = vals[j];
			
			++k; ++j;
		}
	}
 
	while (i<=center) 
	{
		app_rows[k] = rows[i];
		app_cols[k] = cols[i];
		app_vals[k] = vals[i];
			
		++k; ++i;
	}
 
	while (j<=end) 
	{
		app_rows[k] = rows[j];
		app_cols[k] = cols[j];
		app_vals[k] = vals[j];
			
		++k; ++j;
	}
 
	for (k=start; k<=end; k++)
	{
		rows[k] = app_rows[k-start];
		cols[k] = app_cols[k-start];
		vals[k] = app_vals[k-start];
	}
}
 
void mergecoosort(int *rows, int *cols, testType *vals, int size) { 
	app_rows = (int*)malloc(size*sizeof(int));
	app_cols = (int*)malloc(size*sizeof(int));
	app_vals = (testType*)malloc(size*sizeof(testType));

	int sizetomerge=size-1;
	size--;
	int i;
	int n=2;

	while (n<sizetomerge*2) {
		for (i=0; (i+n-1)<=sizetomerge; i+=n) {
			merge(rows,cols,vals,i,(i+i+n-1)/2,i+(n-1),sizetomerge); 
		}
 
		i--;
		if ((sizetomerge+1)%n!=0) {
			if (size>sizetomerge)
				merge (rows,cols,vals,sizetomerge -((sizetomerge)%n),sizetomerge,size,size);
			sizetomerge=sizetomerge-((sizetomerge+1)%n);}
		n=n*2;
	}
 
	if (size>sizetomerge) 
		merge (rows,cols,vals,0,size-(size-sizetomerge),size,size);
		

	free(app_rows);
	free(app_cols);
	free(app_vals);

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
	 
	// Sort COO for cusparse ////////////////////
	
	printf("Sorting COO for cusparse\n");
	printf("Input matrix is %s:\n", input);
	printf("rows: %i:\n", rowsCount);
	printf("columns: %i\n", columnsCount);
	printf("symmetric: %s\n", matrixType == MATRIX_TYPE_SYMMETRIC ? "true" : "false");
	printf("non zeros: %i\n", nonZerosCount);


	mergecoosort(rows, cols, values, nonZerosCount);
	

	
	
	printf("Compute on GPU..\n");
	printf("Testing with alpha = %f and beta = %f\n", (float)ALPHA, (float)BETA);

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

	testType *devX, *devY, *devZ;
	
	cudaMalloc((void**)&devX,columnsCount*sizeof(testType));
	cudaMalloc((void**)&devY,rowsCount*sizeof(testType));
	cudaMalloc((void**)&devZ,rowsCount*sizeof(testType));

	cudaMemcpy(devX, x, columnsCount*sizeof(testType), cudaMemcpyHostToDevice);
	cudaMemcpy(devY, y, rowsCount*sizeof(testType), cudaMemcpyHostToDevice);
	
		spgpuHandle_t spgpuHandle;
	spgpuCreate(&spgpuHandle, 0);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	printf("Computing on %s\n", deviceProp.name);
	Clock timer;

	
	
	
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);
	
	
	testType dotRes;
	testType time;
	testType gflops;
	testType start;

	

	///////////////////////////////////////////////////////////////////////////



	cusparseStatus_t status;
	cusparseHandle_t cusparseHandle;
	status = cusparseCreate(&cusparseHandle);
	if (status != CUSPARSE_STATUS_SUCCESS) { printf("CUSPARSE Library initialization failed"); return 1; }

	cusparseMatDescr_t descr=0;
	
	cusparseCreateMatDescr(&descr);
	cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    	cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);  

	int *csrRowPtrDev;
	int *rowsDev;
	int *colsDev;
	testType* valuesDev;
	
	printf("CSR needs %li bytes\n", ((long int)(rowsCount+1) + 2l * (long int)nonZerosCount)*sizeof(int) 
		+ (long int)nonZerosCount*sizeof(testType));

	cudaMalloc((void**)&csrRowPtrDev, (rowsCount+1)*sizeof(int));
	cudaMalloc((void**)&rowsDev, nonZerosCount*sizeof(int));
	cudaMalloc((void**)&colsDev, nonZerosCount*sizeof(int));
	cudaMalloc((void**)&valuesDev, nonZerosCount*sizeof(testType));
	
	cudaMemcpy(rowsDev, rows, nonZerosCount*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(colsDev, cols, nonZerosCount*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(valuesDev, values, nonZerosCount*sizeof(testType), cudaMemcpyHostToDevice);
	
	printf("Converting to CSR..\n");
	status = cusparseXcoo2csr(cusparseHandle, rowsDev, nonZerosCount, rowsCount, csrRowPtrDev, CUSPARSE_INDEX_BASE_ZERO);
	if (status != CUSPARSE_STATUS_SUCCESS) { printf("Conversion from COO to CSR format failed\n"); return 1; } 

	cudaDeviceSynchronize();
	printf("Converted.\n");
	
	testType alpha = ALPHA;
	testType beta = BETA;
	
	testType* alphaDev;
	testType* betaDev;

	cudaMalloc((void**)&alphaDev, sizeof(testType));
	cudaMalloc((void**)&betaDev, sizeof(testType));
	cudaMemcpy(alphaDev, &alpha, sizeof(testType), cudaMemcpyHostToDevice);
	cudaMemcpy(betaDev, &beta, sizeof(testType), cudaMemcpyHostToDevice);

	printf("Timing cuSPARSE CSR.\n");
	start = timer.getTime();

	for (int i=0; i<NUM_TESTS; ++i)
	{
#ifdef TEST_DOUBLE
		status= cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
			rowsCount, columnsCount, nonZerosCount, &alpha, descr, valuesDev, csrRowPtrDev, 
			colsDev, devX, &beta, devY);
#else
		status= cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
			rowsCount, columnsCount, nonZerosCount, &alpha, descr, valuesDev, csrRowPtrDev, 
			colsDev, devX, &beta, devY);
#endif

	}
	cudaDeviceSynchronize();
	if (status != CUSPARSE_STATUS_SUCCESS) { printf("Matrix-vector multiplication failed\n"); return 1; } 


	time = (timer.getTime() - start)/NUM_TESTS;
	printf("elapsed time: %f seconds\n", time);

	gflops = (((nonZerosCount*2-1)) / time)*0.000000001f;
	printf("GFlop/s: %f\n", gflops);
	
#ifdef TEST_DOUBLE	
	dotRes = spgpuDdot(spgpuHandle, rowsCount, devY, devY);
#else
	dotRes = spgpuSdot(spgpuHandle, rowsCount, devY, devY);
#endif
	cudaDeviceSynchronize();

	printf("dot res: %e\n", dotRes);
	
	cusparseHybMat_t hybA;
	cusparseCreateHybMat(&hybA);

	printf("Converting to HYB..\n");
#ifdef TEST_DOUBLE	
	status= cusparseDcsr2hyb(cusparseHandle, rowsCount, columnsCount, descr, 
		valuesDev, csrRowPtrDev, colsDev, hybA, 
		0, CUSPARSE_HYB_PARTITION_AUTO);
#else	
	status= cusparseScsr2hyb(cusparseHandle, rowsCount, columnsCount, descr, 
		valuesDev, csrRowPtrDev, colsDev, hybA, 
		0, CUSPARSE_HYB_PARTITION_AUTO);
	
#endif	

	cudaDeviceSynchronize();
	printf("Converted.\n");
	
	if (status != CUSPARSE_STATUS_SUCCESS) { printf("Conversion from CSR to HYB format failed\n"); return 1; } 
		
	printf("Timing cuSPARSE HYB.\n");
	start = timer.getTime();

	for (int i=0; i<NUM_TESTS; ++i)
	{
#ifdef TEST_DOUBLE
		cusparseDhybmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
			&alpha, descr, hybA, devX, &beta, devY);
#else
		cusparseShybmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
			&alpha, descr, hybA, devX, &beta, devY);
#endif

	}
	cudaDeviceSynchronize();

	time = (timer.getTime() - start)/NUM_TESTS;
	printf("elapsed time: %f seconds\n", time);

	gflops = (((nonZerosCount*2-1)) / time)*0.000000001f;
	printf("GFlop/s: %f\n", gflops);
	
#ifdef TEST_DOUBLE	
	dotRes = spgpuDdot(spgpuHandle, rowsCount, devY, devY);
#else
	dotRes = spgpuSdot(spgpuHandle, rowsCount, devY, devY);
#endif
	cudaDeviceSynchronize();

	printf("dot res: %e\n", dotRes);



	
	cusparseDestroyHybMat(hybA);
	
	cusparseHybMat_t ellA;
	cusparseCreateHybMat(&ellA);

	printf("Converting to ELL..\n");
#ifdef TEST_DOUBLE	
	status=cusparseDcsr2hyb(cusparseHandle, rowsCount, columnsCount, descr, 
		valuesDev, csrRowPtrDev, colsDev, ellA, 
		0, CUSPARSE_HYB_PARTITION_MAX);
#else	
	status=cusparseScsr2hyb(cusparseHandle, rowsCount, columnsCount, descr, 
		valuesDev, csrRowPtrDev, colsDev, ellA, 
		0, CUSPARSE_HYB_PARTITION_MAX);
	
#endif	
	cudaDeviceSynchronize();
	printf("Converted.\n");
	if (status != CUSPARSE_STATUS_SUCCESS) { printf("Conversion from CSR to ELL format failed\n"); return 1; } 

	
	printf("Timing cuSPARSE ELL.\n");
	start = timer.getTime();

	for (int i=0; i<NUM_TESTS; ++i)
	{
#ifdef TEST_DOUBLE
		status=cusparseDhybmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
			&alpha, descr, ellA, devX, &beta, devY);
#else
		status=cusparseShybmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
			&alpha, descr, ellA, devX, &beta, devY);
#endif

	}
	cudaDeviceSynchronize();
	if (status != CUSPARSE_STATUS_SUCCESS) { printf("Matrix-vector multiplication failed\n"); return 1; } 

	time = (timer.getTime() - start)/NUM_TESTS;
	printf("elapsed time: %f seconds\n", time);

	gflops = (((nonZerosCount*2-1)) / time)*0.000000001f;
	printf("GFlop/s: %f\n", gflops);
	
	
#ifdef TEST_DOUBLE	
	dotRes = spgpuDdot(spgpuHandle, rowsCount, devY, devY);
#else
	dotRes = spgpuSdot(spgpuHandle, rowsCount, devY, devY);
#endif
	cudaDeviceSynchronize();

	printf("dot res: %e\n", dotRes);
	
	cusparseDestroyHybMat(ellA);
	
	cusparseDestroyMatDescr(descr); 
	cusparseDestroy(cusparseHandle);

	cublasDestroy(cublasHandle);
	
	spgpuDestroy(spgpuHandle);


	return 0;
}
