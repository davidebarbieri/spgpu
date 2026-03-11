#pragma once

/*
 * spGPU - Sparse matrices on GPU library.
 * Copyright (C) 2010-2012 Davide Barbieri - University of Rome Tor Vergata
 * SPDX-License-Identifier: BSD-3-Clause
 */
 
 
template<typename T>
void getUnfoldedMmSymmetricSize(int *unfoldedNonZerosCount, T* value, int* rows, int* cols, int nonZerosCount)
{
	for (int i=0; i<nonZerosCount; ++i)
	{
		int r = rows[i];
		int c = cols[i];
		
		T v = value[i];
		if (v != 0)
		{
			if (r == c)
				*unfoldedNonZerosCount += 1; 
			else
				*unfoldedNonZerosCount += 2;
		}
	}
}


template<typename T>	
void unfoldMmSymmetricReal(int* unfoldedRows, int* unfoldedCols, T* unfoldedValues, int *rows, int *cols, T* values, int nonZerosCount)
{
	int nnz = 0;
	for (int i=0; i<nonZerosCount; ++i)
	{
		int r = rows[i];
		int c = cols[i];
		T v = values[i];
		
		if (v != 0)
		{
			if (r == c)
			{
				unfoldedRows[nnz] = r;
				unfoldedCols[nnz] = c;
				unfoldedValues[nnz] = v;
				++nnz;
			}
			else
			{
				unfoldedRows[nnz] = r;
				unfoldedCols[nnz] = c;
				unfoldedValues[nnz] = v;
				++nnz;
				unfoldedRows[nnz] = c;
				unfoldedCols[nnz] = r;
				unfoldedValues[nnz] = v;
				++nnz;
			}
		}
	}
}
