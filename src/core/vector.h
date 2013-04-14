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

/** \addtogroup vecFun Vectors and sparse vectors routines
 *  @{
 */


#ifdef __cplusplus
extern "C" {
#endif

/** 
* \fn float spgpuSdot (spgpuHandle_t handle, int n, __device float* a, __device float* b)
 * Computes single precision dot product of a and b vectors.
 * \param handle The spgpu handle used to call this routine
 * \param n the vectors length
 * \param a the first input vector
 * \param b the second input vector
 * \return the dot product
 */
float spgpuSdot(spgpuHandle_t handle, 
	int n, 
	__device float* a, 
	__device float* b);

/** 
* \fn float spgpuSmdot (spgpuHandle_t handle, float* y, int n, __device float* a, __device float* b, int count, int pitch)
 * Computes single precision dot product of a and b multivectors.
 * \param handle the spgpu handle used to call this routine
 * \param y the result, made by dot products of every vector couples from the multivectors a and b
 * \param n the vectors' length
 * \param a the first input multivector
 * \param b the second input multivector
 * \param count the number of vectors in every multivector
 * \param pitch the pitch, in number of elements, of every multivectors (so the second element of the first vector in a will be a[pitch], the third a[2*pitch], etc.).
 */
void spgpuSmdot(spgpuHandle_t handle, 
	float* y, 
	int n, 
	__device float* a, 
	__device float* b, 
	int count, 
	int pitch);


/** 
* \fn float spgpuSnrm2(spgpuHandle_t handle, int n, __device float* x)
 * Computes the single precision Euclidean vector norm of x. 
 * \param handle the spgpu handle used to call this routine
 * \param n the vector's length
 * \param x the input vector
 * \return the euclidean vector norm
 */
float spgpuSnrm2(spgpuHandle_t handle, 
	int n, 
	__device float* x);

/** 
* \fn void spgpuSmnrm2(spgpuHandle_t handle, float *y, int n, __device float *x, int count, int pitch)
 * Computes the single precision Euclidean vector norm for every vector in the multivector x. 
 * \param handle the spgpu handle used to call this routine
 * \param y the array of results
 * \param n the vectors' length in the x multivector
 * \param x the input multivector
 * \param count the number of vectors in x
 * \param pitch the multivector's pitch
 */	
void spgpuSmnrm2(spgpuHandle_t handle, 
	float *y, 
	int n, 
	__device float *x, 
	int count, 
	int pitch);


/** 
* \fn void spgpuSscal(spgpuHandle_t handle, __device float *y, int n, float alpha, __device float *x)
 * Computes the single precision y = alpha * x. y could be exactly x (without offset) or another vector.
 * \param handle the spgpu handle used to call this routine
 * \param y the resulting vector
 * \param n the vectors' length
 * \param alpha the alpha value
 * \param x the input vector
 */	
void spgpuSscal(spgpuHandle_t handle,
	__device float *y,
	int n,
	float alpha,
	__device float *x);

/** 
* \fn void spgpuSaxpby(spgpuHandle_t handle, __device float *z, int n, float beta, __device float *y, float alpha, __device float* x)
 * Computes the single precision z = beta * y + alpha * x. z could be exactly x or y (without offset) or another vector.
 * \param handle the spgpu handle used to call this routine
 * \param z the resulting vector
 * \param n the vectors' length
 * \param beta the beta value
 * \param y the first input vector
 * \param alpha the alpha value
 * \param x the second input vector
 */
void spgpuSaxpby(spgpuHandle_t handle,
	__device float *z,
	int n,
	float beta,
	__device float *y,
	float alpha,
	__device float* x);

/** 
* \fn void spgpuSmaxpby(spgpuHandle_t handle, __device float *z, int n, float beta, __device float *y, float alpha, __device float* x, int count, int pitch)
 * Computes the single precision z = beta * y + alpha * x of x and y multivectors. z could be exactly x or y (without offset) or another vector.
 * \param handle the spgpu handle used to call this routine
 * \param z the resulting vector
 * \param n the vectors' length
 * \param beta the beta value
 * \param y the first input vector
 * \param alpha the alpha value
 * \param x the second input vector
 * \param count the number of vectors in z,x and y multivectors
 * \param pitch the multivectors pitch
 */

void spgpuSmaxpby(spgpuHandle_t handle,
		  __device float *z,
		  int n,
		  float beta,
		  __device float *y,
		  float alpha,
		  __device float* x, 
		  int count, int pitch);

/** 
* \fn void spgpuSaxy(spgpuHandle_t handle, __device float *z, int n, float alpha, __device float *x, __device float* y)
 * Computes the single precision z = alpha * x * y. z could be exactly x or y (without offset) or another vector.
 * \param handle the spgpu handle used to call this routine
 * \param z the resulting vector
 * \param n the vectors' length
 * \param alpha the alpha value
 * \param x the first input vector
 * \param y the second input vector
 */
void spgpuSaxy(spgpuHandle_t handle,
	__device float *z,
	int n,
	float alpha,
	__device float *x,
	__device float *y);

/** 
* \fn void spgpuSaxypbz(spgpuHandle_t handle, __device float *w, int n, float beta, __device float *z, float alpha, __device float* x, __device float *y)
 * Computes the single precision w = beta * z + alpha * x * y. w could be exactly x, y or z (without offset) or another vector.
 * \param handle the spgpu handle used to call this routine
 * \param w the resulting vector
 * \param n the vectors' length
 * \param beta the beta value
 * \param z the first input vector
 * \param alpha the alpha value
 * \param x the second input vector
 * \param y the third input vector
 */
void spgpuSaxypbz(spgpuHandle_t handle,
	__device float *w,
	int n,
	float beta,
	__device float *z,
	float alpha,
	__device float* x,
	__device float *y);

/** 
* \fn void spgpuSmaxy(spgpuHandle_t handle, __device float *z, int n, float alpha, __device float *x, __device float* y, int count, int pitch)
 * Computes the single precision z = alpha * x * y for z,x and y multivectors. z could be exactly x or y (without offset) or another vector.
 * \param handle the spgpu handle used to call this routine
 * \param z the resulting multivector
 * \param n the vectors' length in the multivectors
 * \param alpha the alpha value
 * \param x the first input multivector
 * \param y the second input multivector
 * \param count the number of vectors in z,x and y multivectors
 * \param pitch the multivectors pitch
 */
void spgpuSmaxy(spgpuHandle_t handle,
	__device float *z,
	int n,
	float alpha,
	__device float* x,
	__device float *y,
	int count,
	int pitch);
	
/** 
* \fn void spgpuSmaxypbz(spgpuHandle_t handle, __device float *w, int n, float beta, __device float *z, float alpha, __device float* x, __device float *y, int count, int pitch)
 * Computes the single precision w = beta * z + alpha * x * y. w could be exactly x, y or z (without offset) or another vector.
 * \param handle the spgpu handle used to call this routine
 * \param w the resulting vector
 * \param n the vectors' length
 * \param beta the beta value
 * \param z the first input vector
 * \param alpha the alpha value
 * \param x the second input vector
 * \param y the third input vector
 * \param count the number of vectors in w,z,x and y multivectors
 * \param pitch the multivectors' pitch
 */
	
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
	
/** 
* \fn void spgpuSgath(spgpuHandle_t handle, __device float *xValues, int xNnz, const __device int *xIndices, int xBaseIndex, const __device float* y)
 * Single precision gather from y to sparse(x). Computes the single precision gather from y to xValues (using xIndices).
 * \param handle the spgpu handle used to call this routine
 * \param xValues the destination array for gathered values
 * \param xNnz the number of elements to gather
 * \param xIndices the array of indices for the elements to be gathered
 * \param xBaseIndex the base index used in xIndices (i.e. 0 for C, 1 for Fortran).
 * \param y the source vector (from which the elements will be gathered)
 */
void spgpuSgath(spgpuHandle_t handle,
	__device float *xValues,
	int xNnz,
	const __device int *xIndices,
	int xBaseIndex,
	const __device float* y);
	
/** 
* \fn void spgpuSscat(spgpuHandle_t handle, __device float* y, int xNnz, const __device float *xValues, const __device int *xIndices, int xBaseIndex, float beta)
 * Single precision scatter from sparse(x) to y. Computes the single precision scatter from xValues to y (using xIndices).
 * The scattered element will be, for i in [0,xNnz), y[xIndices[i]] = beta*y[xIndices[i]] + xValues[i] (to be noted that
 * y values will be multiplied with beta just for scattered values).
 * \param handle the spgpu handle used to call this routine
 * \param y the destination vector (to which the elements will be scattered)
 * \param xNnz the number of elements to scatter
 * \param xValues the source array from which the values will be read
 * \param xIndices the array of indices for the elements to be scattered
 * \param xBaseIndex the base index used in xIndices (i.e. 0 for C, 1 for Fortran).
 * \param beta the beta value
 */
void spgpuSscat(spgpuHandle_t handle,
	__device float* y,
	int xNnz,
	const __device float *xValues,
	const __device int *xIndices,
	int xBaseIndex, float beta);	

/** 
* \fn void spgpuDscal(spgpuHandle_t handle, __device double *y, int n, double alpha, __device double *x)
 * Computes the Double precision y = alpha * x. y could be exactly x (without offset) or another vector.
 * \param handle the spgpu handle used to call this routine
 * \param y the resulting vector
 * \param n the vectors' length
 * \param alpha the alpha value
 * \param x the input vector
 */
void spgpuDscal(spgpuHandle_t handle,
	__device double *y,
	int n,
	double alpha,
	__device double *x);

/** 
* \fn float spgpuDdot (spgpuHandle_t handle, int n, __device double* a, __device double* b)
 * Computes double precision dot product of a and b vectors.
 * \param handle The spgpu handle used to call this routine
 * \param n the vectors length
 * \param a the first input vector
 * \param b the second input vector
 * \return the dot product
 */
 double spgpuDdot(spgpuHandle_t handle, 
	int n, 
	__device double* a, 
	__device double* b);

/** 
* \fn float spgpuDmdot (spgpuHandle_t handle, double* y, int n, __device double* a, __device double* b, int count, int pitch)
 * Computes double precision dot product of a and b multivectors.
 * \param handle the spgpu handle used to call this routine
 * \param y the result, made by dot products of every vector couples from the multivectors a and b
 * \param n the vectors' length
 * \param a the first input multivector
 * \param b the second input multivector
 * \param count the number of vectors in every multivector
 * \param pitch the pitch, in number of elements, of every multivectors (so the second element of the first vector in a will be a[pitch], the third a[2*pitch], etc.).
 */
void spgpuDmdot(spgpuHandle_t handle, 
	double* y, 
	int n, 
	__device double* a, 
	__device double* b, 
	int count, 
	int pitch);


/** 
* \fn double spgpuDnrm2(spgpuHandle_t handle, int n, __device double* x)
 * Computes the double precision Euclidean vector norm of x. 
 * \param handle the spgpu handle used to call this routine
 * \param n the vector's length
 * \param x the input vector
 * \return the euclidean vector norm
 */
 double spgpuDnrm2(spgpuHandle_t handle, 
	int n, 
	__device double* x);

/** 
* \fn void spgpuDmnrm2(spgpuHandle_t handle, double *y, int n, __device double *x, int count, int pitch)
 * Computes the double precision Euclidean vector norm for every vector in the multivector x. 
 * \param handle the spgpu handle used to call this routine
 * \param y the array of results
 * \param n the vectors' length in the x multivector
 * \param x the input multivector
 * \param count the number of vectors in x
 * \param pitch the multivector's pitch
 */	
	
void spgpuDmnrm2(spgpuHandle_t handle, 
	double *y, 
	int n, 
	__device double *x, 
	int count, 
	int pitch);
	
/** 
* \fn void spgpuDaxpby(spgpuHandle_t handle, __device double *z, int n, double beta, __device double *y, double alpha, __device double* x)
 * Computes the double precision z = beta * y + alpha * x. z could be exactly x or y (without offset) or another vector.
 * \param handle the spgpu handle used to call this routine
 * \param z the resulting vector
 * \param n the vectors' length
 * \param beta the beta value
 * \param y the first input vector
 * \param alpha the alpha value
 * \param x the second input vector
 */
void spgpuDaxpby(spgpuHandle_t handle,
	__device double *z,
	int n,
	double beta,
	__device double *y,
	double alpha,
	__device double* x);
	
/** 
* \fn void spgpuDmaxpby(spgpuHandle_t handle, __device double *z, int n, double beta, __device double *y, double alpha, __device double* x, int count, int pitch)
 * Computes the double precision z = beta * y + alpha * x of x and y multivectors. z could be exactly x or y (without offset) or another vector.
 * \param handle the spgpu handle used to call this routine
 * \param z the resulting vector
 * \param n the vectors' length
 * \param beta the beta value
 * \param y the first input vector
 * \param alpha the alpha value
 * \param x the second input vector
 * \param count the number of vectors in z,x and y multivectors
 * \param pitch the multivector's pitch
 */

  void spgpuDmaxpby(spgpuHandle_t handle,
		    __device double *z,
		    int n,
		    double beta,
		    __device double *y,
		    double alpha,
		    __device double* x,
		    int count, int pitch);

/** 
* \fn void spgpuDaxy(spgpuHandle_t handle, __device double *z, int n, double alpha, __device double *x, __device double* y)
 * Computes the double precision z = alpha * x * y. z could be exactly x or y (without offset) or another vector.
 * \param handle the spgpu handle used to call this routine
 * \param z the resulting vector
 * \param n the vectors' length in the x multivector
 * \param alpha the alpha value
 * \param x the second input vector
 * \param y the first input vector
 */
void spgpuDaxy(spgpuHandle_t handle,
	__device double *z,
	int n,
	double alpha,
	__device double *x,
	__device double *y);

/** 
* \fn void spgpuDaxypbz(spgpuHandle_t handle, __device double *w, int n, double beta, __device double *z, double alpha, __device double* x, __device double *y)
 * Computes the double precision w = beta * z + alpha * x * y. w could be exactly x, y or z (without offset) or another vector.
 * \param handle the spgpu handle used to call this routine
 * \param w the resulting vector
 * \param n the vectors' length
 * \param beta the beta value
 * \param z the first input vector
 * \param alpha the alpha value
 * \param x the second input vector
 * \param y the third input vector
 */void spgpuDaxypbz(spgpuHandle_t handle,
	__device double *w,
	int n,
	double beta,
	__device double *z,
	double alpha,
	__device double* x,
	__device double *y);

/** 
* \fn void spgpuDmaxy(spgpuHandle_t handle, __device double *z, int n, double alpha, __device double *x, __device double* y, int count, int pitch)
 * Computes the double precision z = alpha * x * y for z,x and y multivectors. z could be exactly x or y (without offset) or another vector.
 * \param handle the spgpu handle used to call this routine
 * \param z the resulting multivector
 * \param n the vectors' length in the multivectors
 * \param alpha the alpha value
 * \param x the first input multivector
 * \param y the second input multivector
 * \param count the number of vectors in z,x and y multivectors
 * \param pitch the multivectors pitch
 */	
void spgpuDmaxy(spgpuHandle_t handle,
	__device double *z,
	int n,
	double alpha,
	__device double* x,
	__device double *y,
	int count,
	int pitch);
	
/** 
* \fn void spgpuDmaxypbz(spgpuHandle_t handle, __device double *w, int n, double beta, __device double *z, double alpha, __device double* x, __device double *y, int count, int pitch)
 * Computes the double precision w = beta * z + alpha * x * y. w could be exactly x, y or z (without offset) or another vector.
 * \param handle the spgpu handle used to call this routine
 * \param w the resulting vector
 * \param n the vectors' length
 * \param beta the beta value
 * \param z the first input vector
 * \param alpha the alpha value
 * \param x the second input vector
 * \param y the third input vector
 * \param count the number of vectors in w,z,x and y multivectors
 * \param pitch the multivectors' pitch
 */
	
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

/** 
* \fn void spgpuDgath(spgpuHandle_t handle, __device double *xValues, int xNnz, const __device int *xIndices, int xBaseIndex, const __device double* y)
 * Double precision gather from y to sparse(x). Computes the double precision gather from y to xValues (using xIndices).
 * \param handle the spgpu handle used to call this routine
 * \param xValues the destination array for gathered values
 * \param xNnz the number of elements to gather
 * \param xIndices the array of indices for the elements to be gathered
 * \param xBaseIndex the base index used in xIndices (i.e. 0 for C, 1 for Fortran).
 * \param y the source vector (from which the elements will be gathered)
 */
void spgpuDgath(spgpuHandle_t handle,
	__device double *xValues,
	int xNnz,
	const __device int *xIndices,
	int xBaseIndex,
	const __device double* y);
	
/** 
* \fn void spgpuDscat(spgpuHandle_t handle, __device double* y, int xNnz, const __device double *xValues, const __device int *xIndices, int xBaseIndex, double beta)
 * Double precision scatter from sparse(x) to y. Computes the single precision scatter from xValues to y (using xIndices).
 * The scattered element will be, for i in [0,xNnz), y[xIndices[i]] = beta*y[xIndices[i]] + xValues[i] (to be noted that
 * y values will be multiplied with beta just for scattered values).
 * \param handle the spgpu handle used to call this routine
 * \param y the destination vector (to which the elements will be scattered)
 * \param xNnz the number of elements to scatter
 * \param xValues the source array from which the values will be read
 * \param xIndices the array of indices for the elements to be scattered
 * \param xBaseIndex the base index used in xIndices (i.e. 0 for C, 1 for Fortran).
 * \param beta the beta value
 */
void spgpuDscat(spgpuHandle_t handle,
	__device double* y,
	int xNnz,
	const __device double *xValues,
	const __device int *xIndices,
	int xBaseIndex, double beta);
	
/** @}*/
		
#ifdef __cplusplus
}
#endif
