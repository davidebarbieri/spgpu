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
 
 
#include "stdio.h"

#define MATRIX_READ_SUCCESS			0
#define MATRIX_READ_UNSUPPORTED		1
#define MATRIX_READ_INVALID_INPUT	2

/*! Matrix entries are integers */
#define MATRIX_STORAGE_INTEGER		0
/*! Matrix entries are real numbers */
#define MATRIX_STORAGE_REAL		1
/*! Matrix entries are complex numbers */
#define MATRIX_STORAGE_COMPLEX		2
/*! Matrix entries are just coordinates */
#define MATRIX_STORAGE_PATTERN		3

/*! Matrix is general */
#define MATRIX_TYPE_GENERAL		0
/*! Matrix is symmetric */
#define MATRIX_TYPE_SYMMETRIC	1
/*! Matrix is skew-symmetric */
#define MATRIX_TYPE_SKEW		2
/*! Matrix is hermitian */
#define MATRIX_TYPE_HERMITIAN	3

/*! Read the matrix/vector properties from a file in matrix market format. 
 * Use these values to allocate the needed memory to store the matrix.
   * \param rowsCount outputs the rows count
   * \param columnsCount outputs the columns count
   * \param nonZerosCount outputs the non zeros count
   * \param isStoredSparse is the matrix stored in a sparse or dense way?
   * \param matrixStorage it will outputs MATRIX_STORAGE_INTEGER, MATRIX_STORAGE_REAL, MATRIX_STORAGE_COMPLEX or MATRIX_STORAGE_PATTERN
   * \param matrixType it will outputs MATRIX_TYPE_GENERAL, MATRIX_TYPE_SYMMETRIC, MATRIX_TYPE_SKEW or MATRIX_TYPE_HERMITIAN
   * \param file file to read
   * \return Whether the file reading was valid.*/
bool loadMmProperties(int *rowsCount,
	int *columnsCount,
	int *nonZerosCount,
	bool *isStoredSparse,
	int  *matrixStorage,
	int *matrixType,
	FILE *file);

int loadMmMatrixToCoo(
	float* values, 
	int* rowIndices, 
	int* columnIndices,
	int rowsCount,
	int columnsCount,
	int nonZerosCount,
	bool isStoredSparse,
	int matrixStorage,
	FILE *file);

int loadMmMatrixToCoo(
	double* values, 
	int* rowIndices, 
	int* columnIndices,
	int rowsCount,
	int columnsCount,
	int nonZerosCount,
	bool isStoredSparse,
	int matrixStorage,
	FILE *file);

int loadMmMatrixToCoo(
	int* values, 
	int* rowIndices, 
	int* columnIndices,
	int rowsCount,
	int columnsCount,
	int nonZerosCount,
	bool isStoredSparse,
	int matrixStorage,
	FILE *file);

int loadMmMatrixToCoo(
	int* rowIndices, 
	int* columnIndices,
	int rowsCount,
	int columnsCount,
	int nonZerosCount,
	bool isStoredSparse,
	int matrixStorage,
	FILE *file);


int loadMmVectorToDenseVector(
	float* values,
	int vectorSize,
	int matrixStorage,
	FILE *file);

int loadMmVectorToDenseVector(
	double* values,
	int vectorSize,
	int matrixStorage,
	FILE *file);

int loadMmVectorToDenseVector(
	int* values,
	int vectorSize,
	int matrixStorage,
	FILE *file);