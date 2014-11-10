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

__global__ void spgpuDgeins_ovw_krn(int n, int *irl, double *val,
				int xBaseIndex, double* x)
{
  int id = threadIdx.x + BLOCK_SIZE*blockIdx.x;
	
  if (id < n)
    {
      int ix = irl[id]-xBaseIndex;
      if (ix < 0) return;
      x[ix] = val[id];
    }
}


void spgpuDgeins_ovw_(spgpuHandle_t handle,
		      int n,
		      int *irl,
		      double *val,
		      int xBaseIndex,
		      double* x)

{
	int msize = (n+BLOCK_SIZE-1)/BLOCK_SIZE;

	dim3 block(BLOCK_SIZE);
	dim3 grid(msize);

	spgpuDgeins_ovw_krn<<<grid, block, 0, handle->currentStream>>>(n, irl,val, 
								   xBaseIndex,x);
}

__global__ void spgpuDgeins_add_krn(int n, int *irl, double *val,
				int xBaseIndex, double* x)
{
  int id = threadIdx.x + BLOCK_SIZE*blockIdx.x;
	
  if (id < n)
    {
      int ix = irl[id]-xBaseIndex;
      if (ix < 0) return;
      x[ix] += val[id];
    }
}


void spgpuDgeins_add_(spgpuHandle_t handle,
		      int n,
		      int *irl,
		      double *val,
		      int xBaseIndex,
		      double* x)

{
	int msize = (n+BLOCK_SIZE-1)/BLOCK_SIZE;

	dim3 block(BLOCK_SIZE);
	dim3 grid(msize);

	spgpuDgeins_add_krn<<<grid, block, 0, handle->currentStream>>>(n, irl,val, 
								   xBaseIndex,x);

}

void spgpuDgeins(spgpuHandle_t handle,
		 int n,
		 __device int *irl,
		 __device double *val,
		 int dupl,
		 int xBaseIndex,
		 __device double* x)
{
  cudaError_t err;

  //fprintf(stderr,"spgpuDgeins: entry %d %d %p %p %p\n",n,dupl,irl,val,x);
  if (dupl == OVERWRITE) {
    while (n > MAX_N_FOR_A_CALL) //managing large vectors
      {
	//	fprintf(stderr,"spgpuDgeins: OVERWRITE: calling with %d\n",MAX_N_FOR_A_CALL); 	    
	spgpuDgeins_ovw_(handle, MAX_N_FOR_A_CALL, irl, val, xBaseIndex, x);
	
	irl = irl + MAX_N_FOR_A_CALL;
	val = val + MAX_N_FOR_A_CALL;
	n -= MAX_N_FOR_A_CALL;
      }
    //    fprintf(stderr,"spgpuDgeins: OVERWRITE: calling with  %d\n",n); 
    spgpuDgeins_ovw_(handle, n, irl, val, xBaseIndex, x);
  } else if (dupl == ADD) {
    while (n > MAX_N_FOR_A_CALL) //managing large vectors
      {
	//	fprintf(stderr,"spgpuDgeins: ADD: calling with %d\n",MAX_N_FOR_A_CALL); 	    
	spgpuDgeins_add_(handle, MAX_N_FOR_A_CALL, irl, val, xBaseIndex, x);
	
	irl = irl + MAX_N_FOR_A_CALL;
	val = val + MAX_N_FOR_A_CALL;
	n -= MAX_N_FOR_A_CALL;
      }
    //    fprintf(stderr,"spgpuDgeins: ADD: calling with  %d\n",n); 
    spgpuDgeins_add_(handle, n, irl, val, xBaseIndex, x);
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
/* 		  __device double *z, */
/* 		  int n, */
/* 		  double beta, */
/* 		  __device double *y, */
/* 		  double alpha, */
/* 		  __device double* x,  */
/* 		  int count, int pitch) */
/* { */

/*   for (int i=0; i<count; i++) */
/*     spgpuDaxpby(handle, z+pitch*i, n, beta, y+pitch*i, alpha, x+pitch*i); */
  
/* } */
