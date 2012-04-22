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

/// if you change this pitch, assure it's divisible by 16 (2 doubles) or I'll kill you
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
