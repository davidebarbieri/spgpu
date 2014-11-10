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
#include "cudalang.h"
#include "cudadebug.h"

extern "C"
{
#include "core.h"
#include "vector.h"
}



void spgpuDmamax(cublasHandle_t handle, double *y, int n, __device double *x, int count, int pitch)
{
  int i,j;
  for (i=0; i < count; ++i)
    {
      //fprintf(stderr," Calling cublasIdamax %d %p, %d\n",n,x,count);
      cublasIdamax(handle,n,x,1,&j);
      //fprintf(stderr," Exit from  cublasIdamax %d %p\n",j,x);
      cudaError_t err = cudaMemcpy(&(y[i]), &(x[j-1]), sizeof(double), cudaMemcpyDeviceToHost);
      y[i] = fabs(y[i]);
      x += pitch;
    }
}
