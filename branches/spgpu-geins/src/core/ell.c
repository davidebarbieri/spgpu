#include "ell.h"
#include "ell_conv.h"
#include "stdlib.h"

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

int computeEllAllocPitch(int rowsCount)
{
	// returns a pitch good for indices and values
	return ((rowsCount + 31)/32)*32;
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

		void* currentCm = ((char*)ellValues + argRow*elementSize) + currentPos[argRow]*ellValuesPitch*elementSize;
		void* currentRp = ((char*)&ellIndices[argRow]) + currentPos[argRow]*ellIndicesPitch*sizeof(int);

		*((int*)currentRp) = cooColsIndices[i] - cooBaseIndex + ellBaseIndex;
		
		memcpy(currentCm, (char*)cooValues + i*elementSize, elementSize);

		currentPos[argRow]++;

	}
	free(currentPos);
}

void ellToOell(
	int *rIdx,
	void *dstEllValues,
	int *dstEllIndices,
	int *dstRs,
	const void *srcEllValues,
	const int *srcEllIndices,
	const int *srcRs,
	int ellValuesPitch,
	int ellIndicesPitch,
	int rowsCount,
	spgpuType_t valuesType
	)
{
	size_t elementSize = spgpuSizeOf(valuesType);

	int i,j;
	for (i=0; i<rowsCount; ++i)
	{
		rIdx[i] = i;
		dstRs[i] = srcRs[i];
	}
	// simple bubble sort..
	for(i=1;i<rowsCount;i++) 
	{ 
		for(j=0;j<rowsCount-i;j++) 
		{ 
			if(dstRs[j] > dstRs[j+1]) 
			{ 
				int temp=dstRs[j]; 
				dstRs[j]=dstRs[j+1]; 
				dstRs[j+1]=temp; 

				temp=rIdx[j]; 
				rIdx[j]=rIdx[j+1]; 
				rIdx[j+1]=temp; 
			} 
		} 
	} 
	for(i=0;i<rowsCount;i++) 
	{ 
		//Copy a row
		int srcId = rIdx[i];
		int srcLen = srcRs[srcId];

		int k;
		for (k=0; k<srcLen; ++k)
		{
			void* dstCm = ((char*)dstEllValues + i*elementSize) + k*ellValuesPitch*elementSize;
			void* srcCm = ((char*)srcEllValues + srcId*elementSize) + k*ellValuesPitch*elementSize;
			memcpy(dstCm, srcCm, elementSize);

			dstEllIndices[i + k*ellIndicesPitch] = srcEllIndices[srcId + k*ellIndicesPitch];
		}
	}
}