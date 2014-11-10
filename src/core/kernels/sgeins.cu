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

#include "stdio.h"
#include "cudadebug.h"
#include "cudalang.h"

extern "C"
{
#include "core.h"
#include "vector.h"
}


#include "debug.h"

#define BLOCK_SIZE 512
#define MAX_N_FOR_A_CALL (BLOCK_SIZE*65535)
#define OVERWRITE  0
#define ADD        1

__global__ void spgpuSgeins_ovw_krn(int n, int *irl, float *val,
				int xBaseIndex, float* x)
{
  int id = threadIdx.x + BLOCK_SIZE*blockIdx.x;
	
  if (id < n)
    {
      int ix = irl[id]-xBaseIndex;
      if (ix < 0) return;
      x[ix] = val[id];
    }
}


void spgpuSgeins_ovw_(spgpuHandle_t handle,
		      int n,
		      int *irl,
		      float *val,
		      int xBaseIndex,
		      float* x)

{
	int msize = (n+BLOCK_SIZE-1)/BLOCK_SIZE;

	dim3 block(BLOCK_SIZE);
	dim3 grid(msize);

	spgpuSgeins_ovw_krn<<<grid, block, 0, handle->currentStream>>>(n, irl,val, 
								   xBaseIndex,x);
}

__global__ void spgpuSgeins_add_krn(int n, int *irl, float *val,
				int xBaseIndex, float* x)
{
  int id = threadIdx.x + BLOCK_SIZE*blockIdx.x;
	
  if (id < n)
    {
      int ix = irl[id]-xBaseIndex;
      if (ix < 0) return;
      x[ix] += val[id];
    }
}


void spgpuSgeins_add_(spgpuHandle_t handle,
		      int n,
		      int *irl,
		      float *val,
		      int xBaseIndex,
		      float* x)

{
	int msize = (n+BLOCK_SIZE-1)/BLOCK_SIZE;

	dim3 block(BLOCK_SIZE);
	dim3 grid(msize);

	spgpuSgeins_add_krn<<<grid, block, 0, handle->currentStream>>>(n, irl,val, 
								   xBaseIndex,x);

}

void spgpuSgeins(spgpuHandle_t handle,
		 int n,
		 __device int *irl,
		 __device float *val,
		 int dupl,
		 int xBaseIndex,
		 __device float* x)
{
  cudaError_t err;

  //fprintf(stderr,"spgpuSgeins: entry %d %d %p %p %p\n",n,dupl,irl,val,x);
  if (dupl == OVERWRITE) {
    while (n > MAX_N_FOR_A_CALL) //managing large vectors
      {
	//	fprintf(stderr,"spgpuSgeins: OVERWRITE: calling with %d\n",MAX_N_FOR_A_CALL); 	    
	spgpuSgeins_ovw_(handle, MAX_N_FOR_A_CALL, irl, val, xBaseIndex, x);
	
	irl = irl + MAX_N_FOR_A_CALL;
	val = val + MAX_N_FOR_A_CALL;
	n -= MAX_N_FOR_A_CALL;
      }
    //    fprintf(stderr,"spgpuSgeins: OVERWRITE: calling with  %d\n",n); 
    spgpuSgeins_ovw_(handle, n, irl, val, xBaseIndex, x);
  } else if (dupl == ADD) {
    while (n > MAX_N_FOR_A_CALL) //managing large vectors
      {
	//	fprintf(stderr,"spgpuSgeins: ADD: calling with %d\n",MAX_N_FOR_A_CALL); 	    
	spgpuSgeins_add_(handle, MAX_N_FOR_A_CALL, irl, val, xBaseIndex, x);
	
	irl = irl + MAX_N_FOR_A_CALL;
	val = val + MAX_N_FOR_A_CALL;
	n -= MAX_N_FOR_A_CALL;
      }
    //    fprintf(stderr,"spgpuSgeins: ADD: calling with  %d\n",n); 
    spgpuSgeins_add_(handle, n, irl, val, xBaseIndex, x);
  } else  {
    fprintf(stderr,"Error in geins\n"); 
  }
  /* err = cudaDeviceSynchronize(); */
  /* if (err != cudaSuccess){ */
  /*   fprintf(stderr,"CUDA Error from geins/Sync: %s\n", cudaGetErrorString(err)); */
  /* } */
  cudaCheckError("CUDA error on geins");
}

/* void spgpuDmaxpby(spgpuHandle_t handle, */
/* 		  __device float *z, */
/* 		  int n, */
/* 		  float beta, */
/* 		  __device float *y, */
/* 		  float alpha, */
/* 		  __device float* x,  */
/* 		  int count, int pitch) */
/* { */

/*   for (int i=0; i<count; i++) */
/*     spgpuDaxpby(handle, z+pitch*i, n, beta, y+pitch*i, alpha, x+pitch*i); */
  
/* } */
