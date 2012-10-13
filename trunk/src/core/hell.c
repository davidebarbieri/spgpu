#include "hell.h"
#include "hell_conv.h"

void computeHellAllocSize(
	int* allocationHeight,
	int hackSize,
	int rowsCount,
	const int *ellRowLengths
	)
{
	int totalLen = 0;
	int i;
	for (i=0; i<rowsCount/hackSize; ++i)
	{
		int maxLen = 0;
		int j;
		for (j=0; j<hackSize; ++j)
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
	
	for (i=0; i<remainings; ++i)
	{
		int row = done + i;
		int curLen = ellRowLengths[row];
		if (curLen > maxLen)
			maxLen = curLen;
	}
	
	*allocationHeight = totalLen + maxLen;
}

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
	)
{

	size_t elementSize = spgpuSizeOf(valuesType);
	
	int hacks = (rowsCount + hackSize - 1)/hackSize;
	
	char* currValPos = (char*)hellValues;
	int* currIndPos = hellIndices;

	int hackOffset = 0;
	int i;
	for (i=0; i<hacks; ++i)
	{
		int maxLen = 0;
		hackOffsets[i] = hackOffset;

		int j;
		for (j=0; j<hackSize; ++j)
		{
			int row = i*hackSize + j;
			if (row >= rowsCount)
				break;

			int rowLen = ellRowLengths[row];

			if (rowLen > maxLen)
				maxLen = rowLen;

			int k;
			for (k=0; k<rowLen; ++k)
			{
				memcpy(currValPos + (j + k*hackSize)*elementSize,
				 (((char*)ellValues + k*ellValuesPitch) + row*elementSize),
				 elementSize);
				currIndPos[j + k*hackSize] = *((int*) (((char*)ellIndices) + k*ellIndicesPitch) + row);
			}
		}

		hackOffset += hackSize*maxLen;
		currValPos = currValPos + hackSize*maxLen*elementSize;
		currIndPos += hackSize*maxLen;
	}
}
