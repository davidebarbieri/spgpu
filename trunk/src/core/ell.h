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
* \fn template<typename T> void getEllAllocAlignment(int* ellValuesAlignment, int* ellIndicesAlignment)
 * This function returns the ELL format alignment to be provided by values array and column indices memory layout.
 * Use these to compute the size of values and indices pitch, respectively, ((rows*sizeof(T) + ellValuesAlignment - 1)/ellValuesAlignment)*ellValuesAlignment
 * and ((rows*sizeof(int) + ellIndicesAlignment - 1)/ellIndicesAlignment)*ellIndicesAlignment.
 * T is the type of every value element (i.e. float or double for real values).
 * \param ellValuesAlignment outputs the values memory layout alignment
 * \param ellIndicesAlignment outputs the indices memory layout alignment
 * \param rowsCount the rows count
*/
inline void getEllAllocAlignment(
	int* ellValuesAlignment,
	int* ellIndicesAlignment)
{
	// Compute ellValues and ellIndices pitch (in bytes)
	*ellValuesAlignment = ELL_PITCH_ALIGN_BYTE;
	*ellIndicesAlignment = ELL_PITCH_ALIGN_BYTE;
}


/** 
* \fn void spgpuSellspmv (spgpuHandle_t handle,__device float *z,const __device float *y, float alpha, const __device float* cM, const __device int* rP, int cMPitch, int rPPitch, const __device int* rS, int rows, const __device float *x, float beta,int baseIndex)
 * Computes single precision z = alpha*A*x + beta*y, with A stored in ELLpack Format on GPU.
 * \param handle The spgpu handle used to call this routine
 * \param z The output vector of the routine. z could be y, but not y + k (i.e. an overlapping area over y, but starting from a base index different from y).
 * \param y The y input vector
 * \param alpha The alpha scalar
 * \param cM The ELL non zero values allocation pointer
 * \param rP The ELL column indices allocation pointer
 * \param cMPitch the pitch of the allocation containing the matrix non zero values
 * \param rPPitch  the pitch of the allocation containing the matrix non zero column indices
 * \param rS the array containing the row sized (in non zero elements)
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
 * \param cMPitch the pitch of the allocation containing the matrix non zero values
 * \param rPPitch  the pitch of the allocation containing the matrix non zero column indices
 * \param rS the array containing the row sized (in non zero elements)
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
	int rows, 
	const __device double *x, 
	double beta,
	int baseIndex);

/** @}*/

#ifdef __cplusplus
}
#endif
