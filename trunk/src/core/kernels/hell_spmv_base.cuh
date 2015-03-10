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


#define PRE_CONCAT(A, B) A ## B
#define CONCAT(A, B) PRE_CONCAT(A, B)

#undef GEN_SPGPU_HELL_NAME
#undef X_TEX
#define GEN_SPGPU_HELL_NAME(x) CONCAT(CONCAT(spgpu,x),hellspmv)
#define X_TEX CONCAT(x_tex_, FUNC_SUFFIX)

#ifdef ENABLE_CACHE
// Texture cache management
texture < TEX_FETCH_TYPE, 1, cudaReadModeElementType > X_TEX;

#define bind_tex_x(x) cudaBindTexture(NULL, X_TEX, x)
#define unbind_tex_x(x) cudaUnbindTexture(X_TEX)
#endif

#define THREAD_BLOCK 128

#if __CUDA_ARCH__ < 300
extern __shared__ int dynShrMem[]; 
#endif

// Define:
//#define USE_PREFETCHING
//#define VALUE_TYPE
//#define TYPE_SYMBOL
//#define TEX_FETCH_TYPE


__device__ __host__ static float zero_float() { return 0.0f; }
__device__ __host__ static cuFloatComplex zero_cuFloatComplex() { return make_cuFloatComplex(0.0, 0.0); }
__device__ __host__ static bool float_isNotZero(float x) { return x != 0.0f; }

__device__ static float float_fma(float a, float b, float c) { return PREC_FADD(PREC_FMUL (a, b), c); }
__device__ static float float_add(float a, float b) { return PREC_FADD (a, b); }
__device__ static float float_mul(float a, float b) { return PREC_FMUL (a, b); }

__device__ static cuFloatComplex cuFloatComplex_fma(cuFloatComplex a, cuFloatComplex b, cuFloatComplex c) { return cuCfmaf(a, b, c); } 
__device__ static cuFloatComplex cuFloatComplex_add(cuFloatComplex a, cuFloatComplex b) { return cuCaddf(a, b); }
__device__ static cuFloatComplex cuFloatComplex_mul(cuFloatComplex a, cuFloatComplex b) { return cuCmulf(a, b); }

__device__ static float readValue_float(float fetch) { return fetch; }
__device__ static cuFloatComplex readValue_cuFloatComplex(cuFloatComplex fetch) { return fetch; }

// host or c.c >= 1.3 
#if (__CUDA_ARCH__ >= 130) || (!__CUDA_ARCH__)
__device__ __host__ static double zero_double() { return 0.0; }
__device__ __host__ static cuDoubleComplex zero_cuDoubleComplex() { return make_cuDoubleComplex(0.0, 0.0); }
__device__ __host__ static bool double_isNotZero(double x) { return x != 0.0; }

__device__ static double double_fma(double a, double b, double c) { return PREC_DADD(PREC_DMUL (a, b), c); }
__device__ static double double_add(double a, double b) { return PREC_DADD (a, b); }
__device__ static double double_mul(double a, double b) { return PREC_DMUL (a, b); }

__device__ static cuDoubleComplex cuDoubleComplex_fma(cuDoubleComplex a, cuDoubleComplex b, cuDoubleComplex c) { return cuCfma(a, b, c); }
__device__ static cuDoubleComplex cuDoubleComplex_add(cuDoubleComplex a, cuDoubleComplex b) { return cuCadd(a, b); }
__device__ static cuDoubleComplex cuDoubleComplex_mul(cuDoubleComplex a, cuDoubleComplex b) { return cuCmul(a, b); }

__device__ static double readValue_double(int2 fetch) { return __hiloint2double (fetch.y, fetch.x); }
__device__ static cuDoubleComplex readValue_cuDoubleComplex(int4 fetch) 
{
	cuDoubleComplex c;
	c.x = __hiloint2double (fetch.y, fetch.x);
	c.y = __hiloint2double (fetch.w, fetch.z);
	return c;
}
#endif

__device__ static VALUE_TYPE fetchTex(int pointer)
{
	TEX_FETCH_TYPE fetch = tex1Dfetch (X_TEX, pointer);
	return CONCAT(readValue_,VALUE_TYPE) (fetch);
}

__device__ void
CONCAT(GEN_SPGPU_HELL_NAME(TYPE_SYMBOL), _ridx_2)
(int i, VALUE_TYPE yVal, int outRow,
	VALUE_TYPE *z, const VALUE_TYPE *y, VALUE_TYPE alpha, const VALUE_TYPE* cM, const int* rP, int hackSize, const int* hackOffsets, const int* rS, int rows, const VALUE_TYPE *x, VALUE_TYPE beta, int baseIndex)
{
	VALUE_TYPE zProd = CONCAT(zero_,VALUE_TYPE)();

	__shared__ VALUE_TYPE temp[THREAD_BLOCK];
	
	if (i < rows)
	{
		int hackId = i / hackSize;
		int hackLaneId = i % hackSize;
		
		int hackOffset;
		unsigned int laneId = threadIdx.x % 32;
#if __CUDA_ARCH__ < 300
        	// "volatile" used to avoid __syncthreads()
        	volatile int* warpHackOffset = dynShrMem;

        	unsigned int warpId = threadIdx.x / 32;

        	if (laneId == 0)
              		warpHackOffset[warpId] = hackOffsets[hackId];

        	hackOffset = warpHackOffset[warpId] + hackLaneId;
#else
     		if (laneId == 0)
                	hackOffset = hackOffsets[hackId];
        	hackOffset = __shfl(hackOffset, 0) + hackLaneId;
#endif

		rP += hackOffset; 
		cM += hackOffset; 

		int rowSize = rS[i]; 
		int rowSizeM = rowSize / 2;
		
		if (threadIdx.y == 0)
		{
			if (rowSize % 2)
				++rowSizeM;
		}
		else
		{
			rP += hackSize; 
			cM += hackSize;
		}
		
		
		for (int j = 0; j < rowSizeM; j++)
		{
			int pointer;
			VALUE_TYPE value;
			VALUE_TYPE fetch;
		
			pointer = rP[0] - baseIndex;
			rP += hackSize; 
			rP += hackSize;

			value = cM[0];
			cM += hackSize;
			cM += hackSize;

#ifdef ENABLE_CACHE
			fetch = fetchTex(pointer);
#else
			fetch = x[pointer];
#endif	

			// avoid MAD on pre-Fermi
			zProd = CONCAT(VALUE_TYPE, _fma)(value, fetch, zProd);
		}

		// Reduction
		if (threadIdx.y == 1)
			temp[threadIdx.x] = zProd;
	}
	
	__syncthreads();
	
	if (i < rows)
	{
		if (threadIdx.y == 0)	
		{
			zProd = CONCAT(VALUE_TYPE, _add)(zProd, temp[threadIdx.x]);
		
			// Since z and y are accessed with the same offset by the same thread,
			// and the write to z follows the y read, y and z can share the same base address (in-place computing).
	
			if (CONCAT(VALUE_TYPE, _isNotZero(beta)))
				z[outRow] = CONCAT(VALUE_TYPE, _fma)(beta, yVal, CONCAT(VALUE_TYPE, _mul) (alpha, zProd));
			else
				z[outRow] = CONCAT(VALUE_TYPE, _mul)(alpha, zProd);
		}
	}
}	

__device__ void
CONCAT(GEN_SPGPU_HELL_NAME(TYPE_SYMBOL), _ridx)
(int i, VALUE_TYPE yVal, int outRow,
	VALUE_TYPE *z, const VALUE_TYPE *y, VALUE_TYPE alpha, const VALUE_TYPE* cM, const int* rP,  int hackSize, const int* hackOffsets, const int* rS, int rows, const VALUE_TYPE *x, VALUE_TYPE beta, int baseIndex)
{
	VALUE_TYPE zProd = CONCAT(zero_,VALUE_TYPE)();

	if (i < rows)
	{
		
		int hackId = i / hackSize;
		int hackLaneId = i % hackSize;
		
		int hackOffset;
		unsigned int laneId = threadIdx.x % 32;
#if __CUDA_ARCH__ < 300
        	// "volatile" used to avoid __syncthreads()
        	volatile int* warpHackOffset = dynShrMem;

        	unsigned int warpId = threadIdx.x / 32;

        	if (laneId == 0)
              		warpHackOffset[warpId] = hackOffsets[hackId];

        	hackOffset = warpHackOffset[warpId] + hackLaneId;
#else
     		if (laneId == 0)
                	hackOffset = hackOffsets[hackId];
        	hackOffset = __shfl(hackOffset, 0) + hackLaneId;
#endif

		rP += hackOffset; 
		cM += hackOffset; 

		int rowSize = rS[i];

#ifdef USE_PREFETCHING		
		for (int j = 0; j < rowSize / 2; j++)
		{
			int pointers1, pointers2;
			VALUE_TYPE values1, values2;
			VALUE_TYPE fetches1, fetches2;
		
			pointers1 = rP[0] - baseIndex;
			rP += hackSize; 
			pointers2 = rP[0] - baseIndex;
			rP += hackSize; 

			values1 = cM[0];
			cM += hackSize;
			
			values2 = cM[0];
			cM += hackSize;

#ifdef ENABLE_CACHE
			fetches1 = fetchTex(pointers1);
			fetches2 = fetchTex(pointers2);
#else
			fetches1 = x[pointers1];
			fetches2 = x[pointers2];	
#endif

			// avoid MAD on pre-Fermi
			zProd = CONCAT(VALUE_TYPE, _fma)(values1, fetches1, zProd);
			zProd = CONCAT(VALUE_TYPE, _fma)(values2, fetches2, zProd);
		}

		// odd row size
		if (rowSize % 2)
	    	{
	     		int pointer = rP[0] - baseIndex;
	      		VALUE_TYPE value = cM[0];
			VALUE_TYPE fetch;
	      		
#ifdef ENABLE_CACHE
			fetch = fetchTex (pointer);
#else
			fetch = x[pointer];
#endif
			zProd = CONCAT(VALUE_TYPE, _fma)(value, fetch, zProd);
	   	}
#else
		for (int j = 0; j < rowSize; j++)
		{
			int pointer;
			VALUE_TYPE value;
			VALUE_TYPE fetch;
		
			pointer = rP[0] - baseIndex;
			rP += hackSize;

			value = cM[0];
			cM += hackSize;

#ifdef ENABLE_CACHE
			fetch = fetchTex (pointer);
#else
			fetch = x[pointer];
#endif
			zProd = CONCAT(VALUE_TYPE, _fma)(value, fetch, zProd);
	   	}
#endif	   	

		// Since z and y are accessed with the same offset by the same thread,
		// and the write to z follows the y read, y and z can share the same base address (in-place computing).
	
		if (CONCAT(VALUE_TYPE, _isNotZero(beta)))
			z[outRow] = CONCAT(VALUE_TYPE, _fma)(beta, yVal, CONCAT(VALUE_TYPE, _mul) (alpha, zProd));
		else
			z[outRow] = CONCAT(VALUE_TYPE, _mul)(alpha, zProd);
	}
}

__global__ void
CONCAT(GEN_SPGPU_HELL_NAME(TYPE_SYMBOL), _krn_ridx)
(VALUE_TYPE *z, const VALUE_TYPE *y, VALUE_TYPE alpha, const VALUE_TYPE* cM, const int* rP,  int hackSize, const int* hackOffsets, const int* rS, const int* rIdx, int rows, const VALUE_TYPE *x, VALUE_TYPE beta, int baseIndex)
{
	int i = threadIdx.x + blockIdx.x * (THREAD_BLOCK);
	
	VALUE_TYPE yVal = CONCAT(zero_,VALUE_TYPE)();
	int outRow = 0;
	if (i < rows)
	{

		outRow = rIdx[i];
		if (CONCAT(VALUE_TYPE, _isNotZero(beta)))
			yVal = y[outRow];
	}
	
	if (blockDim.y == 1)
		CONCAT(GEN_SPGPU_HELL_NAME(TYPE_SYMBOL), _ridx)
			(i, yVal, outRow, z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rows, x, beta, baseIndex);
	else
		CONCAT(GEN_SPGPU_HELL_NAME(TYPE_SYMBOL), _ridx_2)
			(i, yVal, outRow, z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rows, x, beta, baseIndex);
			
}


__device__ void
CONCAT(GEN_SPGPU_HELL_NAME(TYPE_SYMBOL), _)
(VALUE_TYPE *z, const VALUE_TYPE *y, VALUE_TYPE alpha, const VALUE_TYPE* cM, const int* rP,  int hackSize, const int* hackOffsets, const int* rS, int rows, const VALUE_TYPE *x, VALUE_TYPE beta, int baseIndex)
{
	int i = threadIdx.x + blockIdx.x * (THREAD_BLOCK);
	
	VALUE_TYPE yVal = CONCAT(zero_,VALUE_TYPE)();

	if (i < rows)
	{
		if (CONCAT(VALUE_TYPE, _isNotZero(beta)))
			yVal = y[i];
	
	}
	
	if (blockDim.y == 1)
		CONCAT(GEN_SPGPU_HELL_NAME(TYPE_SYMBOL), _ridx)
			(i, yVal, i, z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rows, x, beta, baseIndex);
	else
		CONCAT(GEN_SPGPU_HELL_NAME(TYPE_SYMBOL), _ridx_2)
			(i, yVal, i, z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rows, x, beta, baseIndex);
			
}

// Force to recompile and optimize with llvm
__global__ void
CONCAT(GEN_SPGPU_HELL_NAME(TYPE_SYMBOL), _krn_b0) 
(VALUE_TYPE *z, const VALUE_TYPE *y, VALUE_TYPE alpha, const VALUE_TYPE* cM, const int* rP,  int hackSize, const int* hackOffsets, const int* rS, int rows, const VALUE_TYPE *x, int baseIndex)
{
	CONCAT(GEN_SPGPU_HELL_NAME(TYPE_SYMBOL), _)
		(z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rows, x, CONCAT(zero_,VALUE_TYPE)(), baseIndex);
}

__global__ void
CONCAT(GEN_SPGPU_HELL_NAME(TYPE_SYMBOL), _krn)
(VALUE_TYPE *z, const VALUE_TYPE *y, VALUE_TYPE alpha, const VALUE_TYPE* cM, const int* rP, int hackSize, const int* hackOffsets, const int* rS, int rows, const VALUE_TYPE *x, VALUE_TYPE beta, int baseIndex)
{
	CONCAT(GEN_SPGPU_HELL_NAME(TYPE_SYMBOL), _)
		(z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rows, x, beta, baseIndex);
}

void
CONCAT(_,GEN_SPGPU_HELL_NAME(TYPE_SYMBOL))
(spgpuHandle_t handle, VALUE_TYPE* z, const VALUE_TYPE *y, VALUE_TYPE alpha, 
	const VALUE_TYPE* cM, const int* rP, int hackSize, const int* hackOffsets, const int* rS,  
	const __device int* rIdx, int avgNnzPerRow, int rows, const VALUE_TYPE *x, VALUE_TYPE beta, int baseIndex)
{
	dim3 block (THREAD_BLOCK, avgNnzPerRow >= 64 ? 2 : 1);

	dim3 grid ((rows + THREAD_BLOCK - 1) / THREAD_BLOCK);

	int shrMemSize;
#if __CUDA_ARCH__ < 300
       	int warpsPerBlock = THREAD_BLOCK/handle->warpSize;
        shrMemSize = warpsPerBlock*sizeof(int);
#else
       	shrMemSize = 0;
#endif

#ifdef ENABLE_CACHE
	bind_tex_x ((const TEX_FETCH_TYPE *) x);
#endif

	if (rIdx)
		CONCAT(GEN_SPGPU_HELL_NAME(TYPE_SYMBOL), _krn_ridx)
			<<< grid, block, shrMemSize, handle->currentStream >>> (z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rIdx, rows, x, beta, baseIndex);
	else
	{
		if (CONCAT(VALUE_TYPE, _isNotZero(beta)))
			CONCAT(GEN_SPGPU_HELL_NAME(TYPE_SYMBOL), _krn) 
				<<< grid, block, shrMemSize, handle->currentStream >>> (z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rows, x, beta, baseIndex);
		else
			CONCAT(GEN_SPGPU_HELL_NAME(TYPE_SYMBOL), _krn_b0)
				<<< grid, block, shrMemSize, handle->currentStream >>> (z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rows, x, baseIndex);
	}

#ifdef ENABLE_CACHE
  	unbind_tex_x ((const TEX_FETCH_TYPE *) x);
#endif

}

void
GEN_SPGPU_HELL_NAME(TYPE_SYMBOL)
(spgpuHandle_t handle, 
	VALUE_TYPE* z, 
	const VALUE_TYPE *y, 
	VALUE_TYPE alpha, 
	const VALUE_TYPE* cM, 
	const int* rP, 
	int hackSize,
	const __device int* hackOffsets, 
	const __device int* rS,
	const __device int* rIdx, 
	int avgNnzPerRow,	
	int rows, 
	const VALUE_TYPE *x, 
	VALUE_TYPE beta, 
	int baseIndex)
{

	int maxNForACall = max(handle->maxGridSizeX, THREAD_BLOCK*handle->maxGridSizeX);

	while (rows > maxNForACall) //managing large vectors
	{
		CONCAT(_,GEN_SPGPU_HELL_NAME(TYPE_SYMBOL)) (handle, z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rIdx, avgNnzPerRow, maxNForACall, x, beta, baseIndex);

		y = y + maxNForACall;
		z = z + maxNForACall;
		cM = cM + maxNForACall;
		rP = rP + maxNForACall;
		rS = rS + maxNForACall;
		
		rows -= maxNForACall;
	}
	
	CONCAT(_,GEN_SPGPU_HELL_NAME(TYPE_SYMBOL)) (handle, z, y, alpha, cM, rP, hackSize, hackOffsets, rS, rIdx, avgNnzPerRow, rows, x, beta, baseIndex);
	
	cudaCheckError("CUDA error on hell_spmv");
}

