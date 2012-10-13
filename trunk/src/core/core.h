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
 
#include "driver_types.h"
 
/** \addtogroup coreFun Core Routines
 *  @{
 */
 
#ifdef __cplusplus
extern "C" {
#endif

/// __host pointers reference host allocations (it's just a placeholder)
#define __host
/// __device pointers reference device allocations (it's just a placeholder)
#define __device

typedef int spgpuStatus_t;

#define SPGPU_SUCCESS 		0
#define SPGPU_UNSUPPORTED 	1
#define SPGPU_UNSPECIFIED	2
#define SPGPU_OUTOFMEMORY	3

#define SPGPU_TYPE_INT		0
#define SPGPU_TYPE_FLOAT	1
#define SPGPU_TYPE_DOUBLE	2

typedef int spgpuType_t;

/// this struct should be modified only internally by spgpu
typedef struct spgpuHandleStruct {
	/// the current stream used by every calls on spgpu routines (passing this handle)
	cudaStream_t currentStream;
	/// the default stream, created during the handle creation.
	cudaStream_t defaultStream;
	/// the device associated to this handle
	int device;
	/// the warp size for this device
	int warpSize;
	/// the max threads per block count for this device
	int maxThreadsPerBlock;
} SpgpuHandleStruct;

typedef const SpgpuHandleStruct* spgpuHandle_t;

/**
* \fn spgpuStatus_t spgpuCreate(spgpuHandle_t* pHandle, int device)
* Create a spgpu context for a GPU device. Every call to spgpu routines using this
* handle will execute on the same GPU. This is re-entrant, so it will work if used by multiple host threads.
* \param pHandle outputs the handle
* \param device id of the device to be used by this context
*/
spgpuStatus_t spgpuCreate(spgpuHandle_t* pHandle, int device);

/**
* \fn void spgpuDestroy(spgpuHandle_t pHandle)
* Destroy the spgpu context for pHandle.
* \param pHandle the handle previously created with spgpuCreate().
*/
void spgpuDestroy(spgpuHandle_t pHandle);

/**
* \fn void spgpuStreamCreate(spgpuHandle_t pHandle, cudaStream_t* stream)
* Create a cuda stream according to the device of the spgpu handle.
* \param stream outputs the new stream
* \param pHandle the handle used to obtain the device id for the stream
*/
void spgpuStreamCreate(spgpuHandle_t pHandle, cudaStream_t* stream);

/**
* \fn void spgpuStreamDestroy(cudaStream_t stream)
* Destroy a stream, previously created with spgpuStreamCreate().
* \param stream the stream to destroy
*/
void spgpuStreamDestroy(cudaStream_t stream);

/**
* \fn void spgpuSetStream(spgpuHandle_t pHandle, cudaStream_t stream)
* Change the current stream for the handle pHandle.
* \param pHandle the handle to configure.
* \param stream the stream to use for next spgpu routines call. Use 0 to reset to the default stream.
*/
void spgpuSetStream(spgpuHandle_t pHandle, cudaStream_t stream);

/**
* \fn size_t spgpuSizeOf(spgpuType_t typeCode)
* Returns the size, in bytes, of the type denoted by typeCode (e.g. 4 for SPGPU_TYPE_FLOAT, 8 for SPGPU_TYPE_DOUBLE).
* \param typeCode outputs the handle
*/
size_t spgpuSizeOf(spgpuType_t typeCode);

/*
typedef struct {
spgpuMatrix

spgpuMatrixType_t MatrixType;
spgpuFillMode_t FillMode;
spgpuDiagType_t DiagType;
int baseIndex;
} spgpuMatrixDesc_t
*/

#ifdef __cplusplus
}
#endif

/** @}*/

