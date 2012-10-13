#include "vector.h"
#include "stdio.h"

#define BLOCK_SIZE 512
#define MAX_N_FOR_A_CALL (BLOCK_SIZE*65535)

// Single Precision Indexed Scatter
__global__ void discat_gpu_kern(double* vector, int count, const int* indexes, const double* values, int firstIndex)
{
	int id = threadIdx.x + BLOCK_SIZE*blockIdx.x;
	
	if (id < count)
	{	
		vector[indexes[id]-firstIndex] = values[id];
	}
}

// Single Precision Indexed Gather
__global__ void digath_gpu_kern(const double* vector, int count, const int* indexes, double* values, int firstIndex)
{
	int id = threadIdx.x + BLOCK_SIZE*blockIdx.x;
	
	if (id < count)
	{
		values[id] = vector[indexes[id]-firstIndex];
	}
}




void spgpuDscat_(spgpuHandle_t handle,
	__device double* y,
	int xNnz,
	const __device double *xValues,
	const __device int *xIndices,
	int xBaseIndex)
{
	int msize = (xNnz+BLOCK_SIZE-1)/BLOCK_SIZE;

	dim3 block(BLOCK_SIZE);
	dim3 grid(msize);

	discat_gpu_kern<<<grid, block, 0, handle->currentStream>>>(y, xNnz, xIndices, xValues, xBaseIndex);
}

void spgpuDgath_(spgpuHandle_t handle,
	__device double *xValues,
	int xNnz,
	const __device int *xIndices,
	int xBaseIndex,
	const __device double* y)
{
	int msize = (xNnz+BLOCK_SIZE-1)/BLOCK_SIZE;

	dim3 block(BLOCK_SIZE);
	dim3 grid(msize);

	digath_gpu_kern<<<grid, block, 0, handle->currentStream>>>(y, xNnz, xIndices, xValues, xBaseIndex);
}


void spgpuDscat(spgpuHandle_t handle,
	__device double* y,
	int xNnz,
	const __device double *xValues,
	const __device int *xIndices,
	int xBaseIndex)
{
	while (xNnz > MAX_N_FOR_A_CALL) //managing large vectors
	{
		spgpuDscat_(handle, y, MAX_N_FOR_A_CALL, xValues, xIndices, xBaseIndex);
	
		xIndices += MAX_N_FOR_A_CALL;
		xValues += MAX_N_FOR_A_CALL;
		xNnz -= MAX_N_FOR_A_CALL;
	}
	
	spgpuDscat_(handle, y, xNnz, xValues, xIndices, xBaseIndex);
}	
	
void spgpuDgath(spgpuHandle_t handle,
	__device double *xValues,
	int xNnz,
	const __device int *xIndices,
	int xBaseIndex,
	const __device double* y)	
{
	while (xNnz > MAX_N_FOR_A_CALL) //managing large vectors
	{
		spgpuDgath_(handle, xValues, MAX_N_FOR_A_CALL, xIndices, xBaseIndex, y);
	
		xIndices += MAX_N_FOR_A_CALL;
		xValues += MAX_N_FOR_A_CALL;
		xNnz -= MAX_N_FOR_A_CALL;
	}
	
	spgpuDgath_(handle, xValues, xNnz, xIndices, xBaseIndex, y);
}
