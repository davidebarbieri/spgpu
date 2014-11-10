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

#undef GEN_SPGPU_ELL_NAME
#define GEN_SPGPU_ELL_NAME(x) CONCAT(CONCAT(spgpu,x),ellcsput)

#define THREAD_BLOCK 512

// Define:
//#define VALUE_TYPE
//#define TYPE_SYMBOL

#include "mathbase.cuh"


__global__ void
CONCAT(GEN_SPGPU_ELL_NAME(TYPE_SYMBOL), _krn)
(VALUE_TYPE alpha, VALUE_TYPE* cM, const int* rP, int cMPitch, int rPPitch, const int* rS,
	int nnz, int *aI, int *aJ, VALUE_TYPE *aVal, int baseIndex)
{
	int i = threadIdx.x + blockIdx.x * (THREAD_BLOCK);
#if 1
	if (i < nnz)
	{
		int row = aI[i] - baseIndex;
		int column = aJ[i];
		VALUE_TYPE value = aVal[i];
		if (row < 0) return;
		if (row < 0)
			return;
		
		// Select row
		cM += row;
		rP += row;
		rS += row;
		
		// Binary search
		int lower = 0;
		int upper = (*rS - 1);
		
		while(lower <= upper) 
		{
			int medium = (lower + upper) / 2;
			
			int currentColumn = rP[medium*rPPitch];
			
			if(currentColumn == column) 
			{
				cM[medium*cMPitch] = value;
				break;
			}
			else if(currentColumn < column)
			    lower = medium + 1;
			else
			    upper = medium - 1;
		}
	}
#else
	if (i < nnz)
	  {
	    int row = aI[i] - baseIndex;
	    int column = aJ[i];
	    VALUE_TYPE value = aVal[i];
	    if (row < 0) return;
	    // Select row
	    cM += row;
	    rP += row;
	    rS += row;
	    
	    // Binary search
	    int lower = 0;
	    int upper = (*rS - 1);
	    
	    for (; lower <= upper; lower += 1)
	      {
		int pos=lower*rPPitch;
		int currentColumn = rP[pos];
		
		if(currentColumn == column) 
		  {
		    cM[pos] = value;
		    break;
		  }
	      }
	    
	  }
#endif
}



void
CONCAT(_,GEN_SPGPU_ELL_NAME(TYPE_SYMBOL))
(spgpuHandle_t handle, VALUE_TYPE alpha, VALUE_TYPE* cM, const int* rP, int cMPitch, 
 int rPPitch, const int* rS,
 int nnz, int *aI, int *aJ, VALUE_TYPE *aVal, int baseIndex)
{
	dim3 block (THREAD_BLOCK, 1);
	
	dim3 grid ((nnz + THREAD_BLOCK - 1) / THREAD_BLOCK);

	CONCAT(GEN_SPGPU_ELL_NAME(TYPE_SYMBOL), _krn)
		<<< grid, block, 0, handle->currentStream >>> (alpha, cM, rP, cMPitch, rPPitch, rS, 
							       nnz, aI, aJ, aVal, baseIndex);
}

void
GEN_SPGPU_ELL_NAME(TYPE_SYMBOL)
	(spgpuHandle_t handle, 
	VALUE_TYPE alpha, 
	VALUE_TYPE* cM, 
	const int* rP, 
	int cMPitch, 
	int rPPitch, 
	const int* rS,
	int nnz, 
	int *aI, 
	int *aJ, 
	VALUE_TYPE *aVal, 
	int baseIndex)
{
	int maxNForACall = max(handle->maxGridSizeX, THREAD_BLOCK*handle->maxGridSizeX);

	while (nnz > maxNForACall) //managing large vectors
	{
		CONCAT(_,GEN_SPGPU_ELL_NAME(TYPE_SYMBOL))
			(handle, alpha, cM, rP, cMPitch, rPPitch, rS, maxNForACall, aI, aJ, aVal, baseIndex);

		aI = aI + maxNForACall;
		aJ = aJ + maxNForACall;
		aVal = aVal + maxNForACall;
		
		nnz -= maxNForACall;
	}
	
	CONCAT(_,GEN_SPGPU_ELL_NAME(TYPE_SYMBOL))
		(handle, alpha, cM, rP, cMPitch, rPPitch, rS, nnz, aI, aJ, aVal, baseIndex);	
	
	cudaCheckError("CUDA error on ell_csput");
}




__global__ void
CONCAT(CONCAT(GEN_SPGPU_ELL_NAME(TYPE_SYMBOL), Idx), _krn)
(VALUE_TYPE alpha, VALUE_TYPE* cM, const int* rP, int cMPitch, int rPPitch, const int* rS,
 int nnz, int *aI, int *aJ, VALUE_TYPE *aVal, int baseIndex, int *upd)
{
	int i = threadIdx.x + blockIdx.x * (THREAD_BLOCK);
#if 0
	if (i < nnz)
	{
		int row = aI[i] - baseIndex;
		int column = aJ[i];
		VALUE_TYPE value = aVal[i];
		if (row < 0) return;
		// Select row
		cM += row;
		rP += row;
		rS += row;
		
		// Binary search
		int lower = 0;
		int upper = (*rS - 1);
		
		while(lower <= upper) 
		{
			int medium = (lower + upper) / 2;
			
			int currentColumn = rP[medium*rPPitch];
			
			if(currentColumn == column) 
			{
				cM[medium*cMPitch] = value;
				break;
			}
			else if(currentColumn < column)
			    lower = medium + 1;
			else
			    upper = medium - 1;
		}
	}
#else
	if (i < nnz)
	  {
	    VALUE_TYPE value = aVal[i];
	    int idx=upd[i];
	    if (idx < 0) { 
	      int row = aI[i] - baseIndex;
	      if (row < 0) return;
	      int column = aJ[i];
	      // Select row
	      cM += row;
	      rP += row;
	      rS += row;
	      // Binary search
	      int lower = 0;
	      int upper = (*rS - 1);
	      
	      for (; lower <= upper; lower += 1)
		{
		  int pos=lower*rPPitch;
		  int currentColumn = rP[pos];
		  
		  if(currentColumn == column) 
		    {
		      cM[pos] = value;
		      upd[i]  = lower*rPPitch+row;
		      break;
		    }
		}
	      
	    } else {
	      cM[idx] = value;
	    } 
	  }
#endif
}



void
CONCAT(_,CONCAT(GEN_SPGPU_ELL_NAME(TYPE_SYMBOL),Idx))
(spgpuHandle_t handle, VALUE_TYPE alpha, VALUE_TYPE* cM, const int* rP, int cMPitch, 
 int rPPitch, const int* rS,
 int nnz, int *aI, int *aJ, VALUE_TYPE *aVal, int baseIndex, int *upd)
{
	dim3 block (THREAD_BLOCK, 1);
	
	dim3 grid ((nnz + THREAD_BLOCK - 1) / THREAD_BLOCK);

	CONCAT(CONCAT(GEN_SPGPU_ELL_NAME(TYPE_SYMBOL), Idx), _krn)
		<<< grid, block, 0, handle->currentStream >>> (alpha, cM, rP, cMPitch, rPPitch, rS, 
							       nnz, aI, aJ, aVal, baseIndex, upd);
}

void
CONCAT(GEN_SPGPU_ELL_NAME(TYPE_SYMBOL),Idx)
  (spgpuHandle_t handle, 
   VALUE_TYPE alpha, 
   VALUE_TYPE* cM, 
   const int* rP, 
   int cMPitch, 
   int rPPitch, 
   const int* rS,
   int nnz, 
   int *aI, 
   int *aJ, 
   VALUE_TYPE *aVal, 
   int baseIndex,
   int *updIdx,
   int cnt)
{ 
#if 0
    GEN_SPGPU_ELL_NAME(TYPE_SYMBOL)
      (handle, alpha, cM, rP, cMPitch, rPPitch, rS, nnz, aI, aJ, aVal, baseIndex);	
    return;
#endif
  int *updSec=updIdx, i;
  if (updSec != NULL) {
    updSec += cnt;
    //int maxNForACall = max(handle->maxGridSizeX, THREAD_BLOCK*handle->maxGridSizeX);
    int maxNForACall = 65536;
    //int maxNForACall = 1024;
    //fprintf(stderr,"Going through new version %d %p\n",cnt,updSec);
    
    while (nnz > maxNForACall) //managing large vectors
      {    CONCAT(_,CONCAT(GEN_SPGPU_ELL_NAME(TYPE_SYMBOL),Idx))
	  (handle, alpha, cM, rP, cMPitch, rPPitch, rS, maxNForACall, aI, aJ, aVal, baseIndex, updSec);
	
	aI = aI + maxNForACall;
	aJ = aJ + maxNForACall;
	aVal = aVal + maxNForACall;
	updSec = updSec + maxNForACall;
	
	nnz -= maxNForACall;
      }
    CONCAT(_,CONCAT(GEN_SPGPU_ELL_NAME(TYPE_SYMBOL),Idx))
      (handle, alpha, cM, rP, cMPitch, rPPitch, rS, nnz, aI, aJ, aVal, baseIndex, updSec);
    
    cudaCheckError("CUDA error on ell_csput");
    
  } else {
    GEN_SPGPU_ELL_NAME(TYPE_SYMBOL)
      (handle, alpha, cM, rP, cMPitch, rPPitch, rS, nnz, aI, aJ, aVal, baseIndex);	
  }
}
