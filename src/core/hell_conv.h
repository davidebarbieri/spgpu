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

#include <string.h>
 
#ifdef __cplusplus
extern "C" {
#endif

void computeHellAllocSize(
	int* allocationHeight,
	int hackSize,
	int rowsCount,
	const int *ellRowLengths
	);

void ellToHell(
	void *hellValues,
	int *hellIndices,
	int* hackOffsets,
	int hackSize,

	const void *ellValues,
	const int *ellIndices,
	int ellValuesPitch,
	int ellIndicesPitch,
	int *ellRowLengths,
	int rowsCount,
	spgpuType_t valuesType
	);

#ifdef __cplusplus
}
#endif


/** @}*/
