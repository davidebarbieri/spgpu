#pragma once

/*
 * spGPU - Sparse matrices on GPU library.
 * 
 * Copyright (C) 2010 - 2012 
 *     Davide Barbieri - University of Rome Tor Vergata
 *     Salvatore Filippone - University of Rome Tor Vergata
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
 
#include "core.h"

/* Single precision dot(x,y)
*/
float spgpuSdot(spgpuHandle_t handle, 
	int n, 
	__device float* a, 
	__device float* b);

void spgpuSmdot(spgpuHandle_t handle, 
	float* y, 
	int n, 
	__device float* a, 
	__device float* b, 
	int count, 
	int pitch);

/* Single precision nrm2(x)
*/
float spgpuSnrm2(spgpuHandle_t handle, 
	int n, 
	__device float* x);
	
float spgpuSmnrm2(spgpuHandle_t handle, 
	float *y, 
	int n, 
	__device float *x, 
	int count, 
	int pitch);

/* Single precision y = alpha * x
	y could be exactly x (without offset) or another vector
*/
void spgpuSscal(spgpuHandle_t handle,
	__device float *y,
	int n,
	float alpha,
	__device float *x);

/* Single precision z = beta * y + alpha * x
	z could be exactly x or y (without offset) or another vector
*/
void spgpuSaxpby(spgpuHandle_t handle,
	__device float *z,
	int n,
	float beta,
	__device float *y,
	float alpha,
	__device float* x);
	
/* Single precision z = alpha * x * y
	z could be exactly x or y (without offset) or another vector
	*/
void spgpuSaxy(spgpuHandle_t handle,
	__device float *z,
	int n,
	float alpha,
	__device float *x,
	__device float *y);

/* Single precision w = beta * z + alpha * x * y
	w could be exactly x, y or z (without offset) or another vector
	*/
void spgpuSaxypbz(spgpuHandle_t handle,
	__device float *w,
	int n,
	float beta,
	__device float *z,
	float alpha,
	__device float* x,
	__device float *y);
	
void spgpuSmaxy(spgpuHandle_t handle,
	__device float *z,
	int n,
	float alpha,
	__device float* x,
	__device float *y,
	int count,
	int pitch);
	
	
void spgpuSmaxypbz(spgpuHandle_t handle,
	__device float *w,
	int n,
	float beta,
	__device float *z,
	float alpha,
	__device float* x,
	__device float *y,
	int count,
	int pitch);	
	
	

/* Double precision y = alpha * x
	y could be exactly x (without offset) or another vector
*/
void spgpuDscal(spgpuHandle_t handle,
	__device double *y,
	int n,
	double alpha,
	__device double *x);

/* Double precision dot(x,y)
*/
double spgpuDdot(spgpuHandle_t handle, 
	int n, 
	__device double* a, 
	__device double* b);

/* Double precision nrm2(x)
*/
double spgpuDnrm2(spgpuHandle_t handle, 
	int n, 
	__device double* x);

	
double spgpuDmnrm2(spgpuHandle_t handle, 
	double *y, 
	int n, 
	__device double *x, 
	int count, 
	int pitch)
	
/* Double precision z = beta * y + alpha * x	
	z could be exactly x or y (without offset) or another vector
*/
void spgpuDaxpby(spgpuHandle_t handle,
	__device double *z,
	int n,
	double beta,
	__device double *y,
	double alpha,
	__device double* x);
	

/* Double precision z = alpha * x * y
	z could be exactly x or y (without offset) or another vector
	*/
void spgpuDaxy(spgpuHandle_t handle,
	__device double *z,
	int n,
	double alpha,
	__device double *x,
	__device double *y);

/* Double precision w = beta * z + alpha * x * y
	w could be exactly x, y or z (without offset) or another vector
	*/
void spgpuDaxypbz(spgpuHandle_t handle,
	__device double *w,
	int n,
	double beta,
	__device double *z,
	double alpha,
	__device double* x,
	__device double *y);
	
void spgpuDmaxy(spgpuHandle_t handle,
	__device double *z,
	int n,
	double alpha,
	__device double* x,
	__device double *y,
	int count,
	int pitch);
	
	
void spgpuDmaxypbz(spgpuHandle_t handle,
	__device double *w,
	int n,
	double beta,
	__device double *z,
	double alpha,
	__device double* x,
	__device double *y,
	int count,
	int pitch);