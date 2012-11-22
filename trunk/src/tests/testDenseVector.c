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

#include "vector.h"
#include "cublas_v2.h"

//#define TEST_DOUBLE
#ifdef TEST_DOUBLE
#define testType double
#else
#define testType float
#endif

#define TEST_SIZE 1234

int main(int argc, char** argv)
{
	testType *xHost = (testType*) malloc(TEST_SIZE*sizeof(testType));	
	testType *xDevice;
	
	int i;
	for (i=0; i<TEST_SIZE; ++i)
		xHost[i] = (testType)i;
	
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);
   
	cudaMalloc((void**)&xDevice, TEST_SIZE*sizeof(testType));
	cudaMemcpy(xDevice, xHost, TEST_SIZE*sizeof(testType), cudaMemcpyHostToDevice);

	
	spgpuHandle_t spgpuHandle;
	spgpuCreate(&spgpuHandle, 0);

	struct cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	printf("Computing on %s\n", deviceProp.name);

	testType spgpuRes;
	testType cublasRes;
	
#ifdef TEST_DOUBLE
	spgpuRes = spgpuDdot(spgpuHandle, TEST_SIZE, xDevice, xDevice);
	cublasDdot (cublasHandle, TEST_SIZE, xDevice, 1, xDevice, 1, &cublasRes);
#else	
	spgpuRes = spgpuSdot(spgpuHandle, TEST_SIZE, xDevice, xDevice);
	cublasSdot (cublasHandle, TEST_SIZE, xDevice, 1, xDevice, 1, &cublasRes);
#endif	
	
	printf("Spgpu Dot res: %f, Cublas res: %f\n", spgpuRes, cublasRes);	
	
	if (spgpuRes == cublasRes)
		printf("Test Passed (Dot operation)\n");
		
	// Testing Nrm2
#ifdef TEST_DOUBLE
	spgpuRes = spgpuDnrm2(spgpuHandle, TEST_SIZE, xDevice);
	cublasDnrm2 (cublasHandle, TEST_SIZE, xDevice, 1, &cublasRes);
#else	
	spgpuRes = spgpuSnrm2(spgpuHandle, TEST_SIZE, xDevice);
	cublasSnrm2 (cublasHandle, TEST_SIZE, xDevice, 1, &cublasRes);
#endif	
	
	printf("Spgpu Nrm2 res: %f, Cublas res: %f\n", spgpuRes, cublasRes);	
	
	if (spgpuRes == cublasRes)
		printf("Test Passed (Nrm2 operation)\n");

	
	spgpuDestroy(spgpuHandle);

	cublasDestroy(cublasHandle);
	free(xHost);

	cudaError_t lastError = cudaGetLastError();
	
	if (lastError != 0)
	{ 
		printf("Error: %i (%s)\n",lastError,cudaGetErrorString(lastError));
		return -1;
	}


	return 0;
}
