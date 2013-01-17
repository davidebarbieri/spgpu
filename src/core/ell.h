#pragma once

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

#include "core.h"

/** \addtogroup ellFun ELL/HELL Format
 *  @{
 */
 
#ifdef __cplusplus
extern "C" {
#endif


// ELL/HELL Compressed Matrix Format routines

/// This is the pitch alignment that must be fullfilled by the coefficients and the row pointers allocations.
#define ELL_PITCH_ALIGN_BYTE 128

/** 
* \fn void spgpuSellspmv (spgpuHandle_t handle,__device float *z,const __device float *y, float alpha, const __device float* cM, const __device int* rP, int cMPitch, int rPPitch, const __device int* rS, int rows, const __device float *x, float beta,int baseIndex)
 * Computes single precision z = alpha*A*x + beta*y, with A stored in ELLpack Format on GPU.
 * \param handle The spgpu handle used to call this routine
 * \param z The output vector of the routine. z could be y, but not y + k (i.e. an overlapping area over y, but starting from a base index different from y).
 * \param y The y input vector
 * \param alpha The alpha scalar
 * \param cM The ELL non zero values allocation pointer
 * \param rP The ELL column indices allocation pointer
 * \param cMPitch the pitch (in number of elements) of the allocation containing the matrix non zero values
 * \param rPPitch  the pitch (in number of elements) of the allocation containing the matrix non zero column indices
 * \param rS the array containing the row sized (in non zero elements)
 * \param rIdx (optional) An array containing the row index per every row (i.e. the reorder array) of the Ell matrix. Pass NULL if you don't use a reorder array (i.e. the k-th row is stored in the k-th position in the ELL format).
 * \param rows the rows count
 * \param x the x vector
 * \param beta the beta scalar
 * \param baseIndex the ELL format base index used (i.e. 0 for C, 1 for Fortran).
 */
void spgpuSellspmv (spgpuHandle_t handle,
	__device float *z,
	const __device float *y, 
	float alpha, 
	const __device float* cM, 
	const __device int* rP, 
	int cMPitch, 
	int rPPitch, 
	const __device int* rS, 
	const __device int* rIdx, 
	int rows, 
	const __device float *x, 
	float beta,
	int baseIndex);

/** 
* \fn void spgpuDellspmv (spgpuHandle_t handle,__device double *z,const __device double *y, double alpha, const __device double* cM, const __device int* rP, int cMPitch, int rPPitch, const __device int* rS, int rows, const __device double *x, double beta,int baseIndex)
 * Computes double precision z = alpha*A*x + beta*y, with A stored in ELLpack Format on GPU.
 * \param handle The spgpu handle used to call this routine
 * \param z The output vector of the routine. z could be y, but not y + k (i.e. an overlapping area over y, but starting from a base index different from y).
 * \param y The y input vector
 * \param alpha The alpha scalar
 * \param cM The ELL non zero values allocation pointer
 * \param rP The ELL column indices allocation pointer
 * \param cMPitch the pitch (in number of elements) of the allocation containing the matrix non zero values
 * \param rPPitch the pitch (in number of elements) of the allocation containing the matrix non zero column indices
 * \param rS the array containing the row sized (in non zero elements)
 * \param rIdx (optional) An array containing the row index per every row (i.e. the reorder array) of the Ell matrix. Pass NULL if you don't use a reorder array (i.e. the k-th row is stored in the k-th position in the ELL format).
 * \param rows the rows count
 * \param x the x vector
 * \param beta the beta scalar
 * \param baseIndex the ELL format base index used (i.e. 0 for C, 1 for Fortran).
 */
void spgpuDellspmv (spgpuHandle_t handle,
	__device double *z,
	const __device double *y, 
	double alpha, 
	const __device double* cM, 
	const __device int* rP, 
	int cMPitch, 
	int rPPitch, 
	const __device int* rS, 
	const __device int* rIdx, 
	int rows, 
	const __device double *x, 
	double beta,
	int baseIndex);

/** @}*/

#ifdef __cplusplus
}
#endif
