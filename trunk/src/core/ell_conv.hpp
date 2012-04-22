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
 
#include "ell.h"

/** \addtogroup conversionRoutines Conversion Routines
 *  @{
 */

/** 
* \fn void computeEllRowLenghts(int *ellRowLengths, int *ellMaxRowSize, int rowsCount, int nonZerosCount, const int* cooRowIndices, int cooBaseIndex)
 * Compute the Ell row lengths array (and the greatest row size) from the COO matrix format.
 * \param ellRowLengths Array of length rowsCount to be filled by the non zeros count for every matrix row
 * \param ellMaxRowSize outputs the greatest row size (in non zeros)
 * \param rowsCount the number of rows of the coo matrix to convert
 * \param nonZerosCount the non zeros count of the coo matrix to convert
 * \param cooRowIndices the row indices array for the coo matrix to convert
 * \param cooBaseIndex the input base index (e.g. 0 for C, 1 for Fortran)
 */
inline void computeEllRowLenghts(
	int *ellRowLengths,
	int *ellMaxRowSize,
	int rowsCount,
	int nonZerosCount,
	const int* cooRowIndices,
	int cooBaseIndex
	)
{
	// find the max number of non zero per row
	int maxRowSize = 0;
	for (int i=0; i<rowsCount; i++) 
		ellRowLengths[i] = 0;

	for (int i=0; i<nonZerosCount; i++)
		++ellRowLengths[cooRowIndices[i] - cooBaseIndex];

	for (int i=0; i<rowsCount; i++)
	{
		int currCount = ellRowLengths[i];
		if (currCount > maxRowSize) 
			maxRowSize = currCount;
	}

	*ellMaxRowSize = maxRowSize;
}

/** 
* \fn void computeEllAllocPitch(int* ellValuesPitch, int* ellIndicesPitch, int rowsCount, int valueElementSize)
 * Compute the ELL format pitch for values array and column indices pitch.
 * Use these to compute the size of values and indices allocations (respectively, ellValuesPitch*maxRowSize and ellIndicesPitch*maxRowSize).
 * \param ellValuesPitch outputs the values allocation pitch
 * \param ellIndicesPitch outputs the indices allocation pitch
 * \param rowsCount the rows count
 * \param valueElementSize the size of every value element (i.e. sizeof(float) or sizeof(double) for real values)
*/
template<typename T>
void computeEllAllocPitch(
	int* ellValuesPitch,
	int* ellIndicesPitch,
	int rowsCount)
{
	// Compute ellValues and ellIndices pitch (in bytes)
	*ellValuesPitch = ((rowsCount*sizeof(T) + ELL_PITCH_ALIGN_BYTE - 1)/ELL_PITCH_ALIGN_BYTE)*ELL_PITCH_ALIGN_BYTE;
	*ellIndicesPitch = ((rowsCount*sizeof(int) + ELL_PITCH_ALIGN_BYTE - 1)/ELL_PITCH_ALIGN_BYTE)*ELL_PITCH_ALIGN_BYTE;
}


/** 
* \fn void cooToEll(ValueType *ellValues,int *ellIndices,int ellValuesPitch,int ellIndicesPitch,int ellMaxRowSize,int ellBaseIndex,int rowsCount,int nonZerosCount,const int* cooRowIndices,const int* cooColsIndices,const ValueType* cooValues,int cooBaseIndex)
 * Convert a matrix in COO format to a matrix in ELL format.
 * The matrix is stored in column-major format.  The ellValues and ellIndices sizes are ellMaxRowSize * pitch (pitch is in bytes).
 * \param ellValues pointer to the area that will be filled by the non zero coefficients
 * \param ellIndices pointer to the area that will be filled by the non zero indices
 * \param ellValuesPitch the column-major allocation's pitch of ellValues (in bytes)
 * \param ellIndicesPitch the column-major allocation's pitch of ellIndices (in bytes)
 * \param ellMaxRowSize the greatest row size
 * \param ellBaseIndex the desired base index for the ELL matrix (e.g. 0 for C, 1 for Fortran)
 * \param rowsCount input matrix rows count
 * \param nonZerosCount input matrix non zeros count
 * \param cooRowIndices input matrix row indices pointer 
 * \param cooColsIndices input matrix column indices pointer 
 * \param cooValues input matrix non zeros values pointer
 * \param cooBaseIndex input matrix base index
 */
template<typename ValueType>
void cooToEll(
	ValueType *ellValues,
	int *ellIndices,
	int ellValuesPitch,
	int ellIndicesPitch,
	int ellMaxRowSize,
	int ellBaseIndex,
	int rowsCount,
	int nonZerosCount,
	const int* cooRowIndices,
	const int* cooColsIndices,
	const ValueType* cooValues,
	int cooBaseIndex
	)
{	
	// fill values and indices
	int* currentPos = (int*)malloc(rowsCount*sizeof(int));

	for (int i=0; i<rowsCount; i++)
		currentPos[i] = 0;

	for (int  i=0; i<nonZerosCount; i++)
	{
		int argRow = cooRowIndices[i] - cooBaseIndex;

		void* currentCm = ((char*)&ellValues[argRow]) + currentPos[argRow]*ellValuesPitch;
		void* currentRp = ((char*)&ellIndices[argRow]) + currentPos[argRow]*ellIndicesPitch;

		*((int*)currentRp) = cooColsIndices[i] - cooBaseIndex + ellBaseIndex;
		*((ValueType*)currentCm) = cooValues[i];

		currentPos[argRow]++;

	}
	free(currentPos);
}

/** @}*/
