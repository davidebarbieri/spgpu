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
 
inline void computeHellAllocSize(
	int* allocationHeight,
	int hackSize,
	int rowsCount,
	const int *ellRowLengths
	)
{
	int totalLen = 0;
	for (int i=0; i<rowsCount/hackSize; ++i)
	{
		int maxLen = 0;
		for (int j=0; j<hackSize; ++j)
		{
			int row = i*hackSize + j;
			int curLen = ellRowLengths[row];
			if (curLen > maxLen)
				maxLen = curLen;
		}
		totalLen += maxLen;
	}

	int remainings = rowsCount % hackSize;
	int done = (rowsCount/hackSize)*hackSize;
	int maxLen = 0;
	for (int i=0; i<remainings; ++i)
	{
		int row = done + i;
		int curLen = ellRowLengths[row];
		if (curLen > maxLen)
			maxLen = curLen;
	}
	
	*allocationHeight = totalLen + maxLen;
}

template<typename ValueType>
void ellToHell(
	ValueType *hellValues,
	int *hellIndices,
	int* hackOffsets,
	int hackSize,

	const ValueType *ellValues,
	const int *ellIndices,
	int ellValuesPitch,
	int ellIndicesPitch,
	int *ellRowLengths,
	int rowsCount
	)
{
	int hacks = (rowsCount + hackSize - 1)/hackSize;
	
	ValueType* currValPos = hellValues;
	int* currIndPos = hellIndices;

	int hackOffset = 0;
	for (int i=0; i<hacks; ++i)
	{
		int maxLen = 0;
		hackOffsets[i] = hackOffset;

		for (int j=0; j<hackSize; ++j)
		{
			int row = i*hackSize + j;
			if (row >= rowsCount)
				break;

			int rowLen = ellRowLengths[row];

			if (rowLen > maxLen)
				maxLen = rowLen;

			for (int k=0; k<rowLen; ++k)
			{
				currValPos[j + k*hackSize] = *((ValueType*) (((char*)ellValues) + k*ellValuesPitch) + row);
				currIndPos[j + k*hackSize] = *((int*) (((char*)ellIndices) + k*ellIndicesPitch) + row);
			}
		}

		hackOffset += hackSize*maxLen;
		currValPos += hackSize*maxLen;
		currIndPos += hackSize*maxLen;
	}
}

/** @}*/
