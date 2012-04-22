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
 

#include "mmread.hpp"
#include "stdio.h"

extern "C" {
#include "mmio.h"
}

bool loadMmProperties(int *rowsCount,
	int *columnsCount,
	int *nonZerosCount,
	bool *isStoredSparse,
	int* matrixStorage,
	int* matrixType,
	FILE *file)
{
	MM_typecode matcode;

	// supports only valid matrices
	if ((mm_read_banner(file, &matcode) != 0) 
		|| (!mm_is_matrix(matcode))
		|| (!mm_is_valid(matcode))) 
		return false;

	if ( mm_read_mtx_crd_size(file, rowsCount, columnsCount, nonZerosCount) != 0 ) 
		return false;

	// is it stored sparse?
	if (mm_is_sparse(matcode))
		*isStoredSparse = true;
	else
		*isStoredSparse = false;

	if (mm_is_integer(matcode))
		*matrixStorage = MATRIX_STORAGE_INTEGER;
	else if (mm_is_real(matcode))
		*matrixStorage = MATRIX_STORAGE_REAL;
	else if (mm_is_complex(matcode))
		*matrixStorage = MATRIX_STORAGE_COMPLEX;
	else if (mm_is_pattern(matcode))
		*matrixStorage = MATRIX_STORAGE_PATTERN;
	
	if (mm_is_general(matcode))
		*matrixType = MATRIX_TYPE_GENERAL;
	else if (mm_is_symmetric(matcode))
		*matrixType = MATRIX_TYPE_SYMMETRIC;
	else if (mm_is_skew(matcode))
		*matrixType = MATRIX_TYPE_SKEW;
	else if (mm_is_hermitian(matcode))
		*matrixType = MATRIX_TYPE_HERMITIAN;

	return true;
}



template<typename T>
void loadMmMatrixToCooReal(
	T* values, 
	int* rowIndices, 
	int* columnIndices,
	int nonZerosCount,
	FILE *file)
{
	int i;
	for (i=0; i<nonZerosCount; ++i)
	{
		double value;
		int row;
		int column;

		fscanf(file, "%d %d %lg\n", &row, &column, &value);

		/* adjust from 1-based to 0-based */
		--row;  
		--column;

		values[i] = (T)value;
		rowIndices[i] = row;
		columnIndices[i] = column;
	}
}

void loadMmMatrixToCooInteger(
	int* values, 
	int* rowIndices, 
	int* columnIndices,
	int nonZerosCount,
	FILE *file)
{
	int i;
	for (i=0; i<nonZerosCount; ++i)
	{
		int value;
		int row;
		int column;

		fscanf(file, "%d %d %d\n", &row, &column, &value);

		/* adjust from 1-based to 0-based */
		--row;  
		--column;

		values[i] = value;
		rowIndices[i] = row;
		columnIndices[i] = column;
	}
}

void loadMmMatrixToCooPattern(
	int* rowIndices, 
	int* columnIndices,
	int nonZerosCount,
	FILE *file)
{
	for (int i=0; i<nonZerosCount; ++i)
	{
		int row;
		int column;

		fscanf(file, "%d %d\n", &row, &column);

		/* adjust from 1-based to 0-based */
		--row;  
		--column;

		rowIndices[i] = row;
		columnIndices[i] = column;
	}
}

int loadMmMatrixToCoo(
	float* values, 
	int* rowIndices, 
	int* columnIndices,
	int rowsCount,
	int columnsCount,
	int nonZerosCount,
	bool isStoredSparse,
	int matrixStorage,
	FILE *file)
{
	if (!isStoredSparse)
		return MATRIX_READ_INVALID_INPUT;

	if (matrixStorage != MATRIX_STORAGE_REAL)
		return MATRIX_READ_UNSUPPORTED;

	loadMmMatrixToCooReal(values, rowIndices, columnIndices, nonZerosCount, file);
	
	return MATRIX_READ_SUCCESS;
}

int loadMmMatrixToCoo(
	double* values, 
	int* rowIndices, 
	int* columnIndices,
	int rowsCount,
	int columnsCount,
	int nonZerosCount,
	bool isStoredSparse,
	int matrixStorage,
	FILE *file)
{
	if (!isStoredSparse)
		return MATRIX_READ_INVALID_INPUT;

	if (matrixStorage != MATRIX_STORAGE_REAL)
		return MATRIX_READ_UNSUPPORTED;

	loadMmMatrixToCooReal(values,rowIndices, columnIndices, nonZerosCount, file);

	return MATRIX_READ_SUCCESS;
}

int loadMmMatrixToCoo(
	int* values, 
	int* rowIndices, 
	int* columnIndices,
	int rowsCount,
	int columnsCount,
	int nonZerosCount,
	bool isStoredSparse,
	int matrixStorage,
	FILE *file)
{
	if (!isStoredSparse) return MATRIX_READ_INVALID_INPUT;
	if (matrixStorage != MATRIX_STORAGE_INTEGER) return MATRIX_READ_UNSUPPORTED;

	loadMmMatrixToCooInteger(values,rowIndices, columnIndices, nonZerosCount, file);
	return MATRIX_READ_SUCCESS;
}

int loadMmMatrixToCoo(
	int* rowIndices, 
	int* columnIndices,
	int rowsCount,
	int columnsCount,
	int nonZerosCount,
	bool isStoredSparse,
	int matrixStorage,
	FILE *file)
{
	if (!isStoredSparse) return MATRIX_READ_INVALID_INPUT;
	if (matrixStorage != MATRIX_STORAGE_PATTERN) return MATRIX_READ_UNSUPPORTED;

	loadMmMatrixToCooPattern(rowIndices, columnIndices, nonZerosCount, file);
	return MATRIX_READ_SUCCESS;
}


int loadMmVectorToDenseVector(
	float* values,
	int vectorSize,
	int matrixStorage,
	FILE *file)
{
	if (matrixStorage != MATRIX_STORAGE_REAL)
		return MATRIX_READ_INVALID_INPUT;

	for (int i=0; i<vectorSize; i++)
	{
		double temp;
		fscanf(file, "%lg\n", &temp);
		values[i] = (float)temp;
	}
	return MATRIX_READ_SUCCESS;
}

int loadMmVectorToDenseVector(
	double* values,
	int vectorSize,
	int matrixStorage,
	FILE *file)
{
	if (matrixStorage != MATRIX_STORAGE_REAL)
		return MATRIX_READ_INVALID_INPUT;

	for (int i=0; i<vectorSize; i++)
	{
		double temp;
		fscanf(file, "%lg\n", &temp);
		values[i] = temp;
	}
	return MATRIX_READ_SUCCESS;
}

int loadMmVectorToDenseVector(
	int* values,
	int vectorSize,
	int matrixStorage,
	FILE *file)
{
	if (matrixStorage != MATRIX_STORAGE_INTEGER)
		return MATRIX_READ_INVALID_INPUT;

	for (int i=0; i<vectorSize; i++)
	{
		int temp;
		fscanf(file, "%i\n", &temp);
		values[i] = temp;
	}
	return MATRIX_READ_SUCCESS;
}