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

//#define TEST_DOUBLE
#ifdef TEST_DOUBLE
#define testType double
#else
#define testType float
#endif

#define TEST_SIZE 1234
#define SPARSE_SIZE 123

int main(int argc, char** argv)
{
	testType *xHost = (testType*) malloc(TEST_SIZE*sizeof(testType));
	testType *resHost = (testType*) malloc(TEST_SIZE*sizeof(testType));
	testType *resDeviceHost = (testType*) malloc(TEST_SIZE*sizeof(testType));
	
	testType *xDevice;
	int i;
	for (i=0; i<TEST_SIZE; ++i)
		xHost[i] = (testType)i;
	
	cudaMalloc((void**)&xDevice, TEST_SIZE*sizeof(testType));
	cudaMemcpy(xDevice, xHost, TEST_SIZE*sizeof(testType), cudaMemcpyHostToDevice);

	spgpuHandle_t spgpuHandle;
	spgpuCreate(&spgpuHandle, 0);

	struct cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	printf("Computing on %s\n", deviceProp.name);

	printf("Scatter test...\n");
	int* yIndicesHost = (int*) malloc(SPARSE_SIZE*sizeof(int));		
	testType* yValuesHost = (testType*) malloc(SPARSE_SIZE*sizeof(testType));
	
	for (i=0; i<SPARSE_SIZE; ++i)
	{
		yIndicesHost[i] = (i*17) % TEST_SIZE;
		yValuesHost[i] = 1.111f*(SPARSE_SIZE - i);
	}
	
	
	int* yIndicesDevice;
	testType* yValuesDevice;
	
	cudaMalloc((void**)&yIndicesDevice, SPARSE_SIZE*sizeof(int));
	cudaMalloc((void**)&yValuesDevice, SPARSE_SIZE*sizeof(testType));

	cudaMemcpy(yIndicesDevice, yIndicesHost, SPARSE_SIZE*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(yValuesDevice, yValuesHost, SPARSE_SIZE*sizeof(testType), cudaMemcpyHostToDevice);
#ifdef TEST_DOUBLE
	spgpuDscat(spgpuHandle, xDevice, SPARSE_SIZE, yValuesDevice, yIndicesDevice, 0, 2.0);	
#else	
	spgpuSscat(spgpuHandle, xDevice, SPARSE_SIZE, yValuesDevice, yIndicesDevice, 0, 2.0f);	
#endif	
	cudaMemcpy(resDeviceHost, xDevice, TEST_SIZE*sizeof(testType), cudaMemcpyDeviceToHost);
	
	// Scatter on host
	for (i=0; i<TEST_SIZE; ++i)
	{
		resHost[i] = xHost[i];
	}
	
	for (i=0; i<SPARSE_SIZE; ++i)
	{
		int index = yIndicesHost[i];
		resHost[index] = 2.0f*resHost[index] + yValuesHost[i];
	}
		
	// Verify correctness
	for (i=0; i<TEST_SIZE; ++i)
	{
		if (resHost[i] != resDeviceHost[i])
		{
			printf("Test Failed (Scatter operation): %i (%f - %f)\n", i, resHost[i], resDeviceHost[i]);
			return -1;
		}
	}	
	
	printf("Test Passed (Scatter operation)\n");
		
	printf("Gather test...\n");

#ifdef TEST_DOUBLE
	spgpuDgath(spgpuHandle, yValuesDevice, SPARSE_SIZE, yIndicesDevice, 0, xDevice);
#else
	spgpuSgath(spgpuHandle, yValuesDevice, SPARSE_SIZE, yIndicesDevice, 0, xDevice);
#endif
	cudaMemcpy(resDeviceHost, yValuesDevice, SPARSE_SIZE*sizeof(testType), cudaMemcpyDeviceToHost);	

	// host version
	for (i=0; i<SPARSE_SIZE; ++i)
	{
		int index = yIndicesHost[i];
		yValuesHost[i] = resHost[index];
	}


	// Verify correctness
	for (i=0; i<SPARSE_SIZE; ++i)
	{
		if (yValuesHost[i] != resDeviceHost[i])
		{
			printf("Test Failed (Gather operation): %i (%f - %f)\n", i, yValuesHost[i], resDeviceHost[i]);
			return -1;
		}
	}	

	
	printf("Test Passed (Gather operation)\n");
	
	spgpuDestroy(spgpuHandle);

	free(xHost);
	free(resHost);

	cudaError_t lastError = cudaGetLastError();
	
	if (lastError != 0)
	{ 
		printf("Error: %i (%s)\n",lastError,cudaGetErrorString(lastError));
		return -1;
	}


	return 0;
}
