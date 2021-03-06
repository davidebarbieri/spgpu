#pragma once

/*
 * spGPU - Sparse matrices on GPU library.
 * 
 * Copyright (C) 2010 - 2014
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
