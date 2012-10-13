#include "ell.h"
#include "ell_conv.h"
#include "stdlib.h"

void getEllAllocAlignment(
	int* ellValuesAlignment,
	int* ellIndicesAlignment)
{
	// Compute ellValues and ellIndices pitch (in bytes)
	*ellValuesAlignment = ELL_PITCH_ALIGN_BYTE;
	*ellIndicesAlignment = ELL_PITCH_ALIGN_BYTE;
}

void computeEllRowLenghts(
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
	int i;
	for (i=0; i<rowsCount; i++) 
		ellRowLengths[i] = 0;

	for (i=0; i<nonZerosCount; i++)
		++ellRowLengths[cooRowIndices[i] - cooBaseIndex];

	for (i=0; i<rowsCount; i++)
	{
		int currCount = ellRowLengths[i];
		if (currCount > maxRowSize) 
			maxRowSize = currCount;
	}

	*ellMaxRowSize = maxRowSize;
}

void computeEllAllocPitch(
	int* ellValuesPitch,
	int* ellIndicesPitch,
	int rowsCount,
	spgpuType_t ellValuesType)
{
	// Compute ellValues and ellIndices pitch (in bytes)
	*ellValuesPitch = ((rowsCount*spgpuSizeOf(ellValuesType) + ELL_PITCH_ALIGN_BYTE - 1)/ELL_PITCH_ALIGN_BYTE)*ELL_PITCH_ALIGN_BYTE;
	*ellIndicesPitch = ((rowsCount*sizeof(int) + ELL_PITCH_ALIGN_BYTE - 1)/ELL_PITCH_ALIGN_BYTE)*ELL_PITCH_ALIGN_BYTE;
}

void cooToEll(
	void *ellValues,
	int *ellIndices,
	int ellValuesPitch,
	int ellIndicesPitch,
	int ellMaxRowSize,
	int ellBaseIndex,
	int rowsCount,
	int nonZerosCount,
	const int* cooRowIndices,
	const int* cooColsIndices,
	const void* cooValues,
	int cooBaseIndex,
	spgpuType_t valuesType
	)
{	

	size_t elementSize = spgpuSizeOf(valuesType);
		
	// fill values and indices
	int* currentPos = (int*)malloc(rowsCount*sizeof(int));
	int i;
	
	for (i=0; i<rowsCount; i++)
		currentPos[i] = 0;

	for (i=0; i<nonZerosCount; i++)
	{
		int argRow = cooRowIndices[i] - cooBaseIndex;

		void* currentCm = ((char*)ellValues + argRow*elementSize) + currentPos[argRow]*ellValuesPitch;
		void* currentRp = ((char*)&ellIndices[argRow]) + currentPos[argRow]*ellIndicesPitch;

		*((int*)currentRp) = cooColsIndices[i] - cooBaseIndex + ellBaseIndex;
		
		memcpy(currentCm, (char*)cooValues + i*elementSize, elementSize);

		currentPos[argRow]++;

	}
	free(currentPos);
}
