#include "hdia_conv.h"
#include "stdlib.h"
#include "string.h"

int getHdiaHacksCount(int hackSize, int rowsCount)
{
	return (rowsCount + hackSize - 1)/hackSize;
}

void computeHdiaHackOffsets(
	int *allocationHeight,
	int *hackOffsets,
	int hackSize,
	const void* diaValues,
	int diaValuesPitch,	
	int diagonals,
	int rowsCount,
	spgpuType_t valuesType
	)
{
	int i,r,s, hack;
	int hackCount = getHdiaHacksCount(hackSize, rowsCount);
	
	size_t elementSize = spgpuSizeOf(valuesType);
	
	int hackHeight = 0;
		
	hackOffsets[0] = 0;
	for (hack=0; hack<hackCount; ++hack)
	{
		for (i=0; i<diagonals; ++i)
		{
			int found = 0;
			for (r=0; r<hackSize; ++r)
			{
				int row = hack*hackSize + r;
				
				if (row >= rowsCount)
					break;
			
				const void* val = diaValues + elementSize*(row + i*diaValuesPitch);
				
				for (s=0; s<elementSize; ++s)
					if (*(char*)(val+s) != 0)
					{
						found = 1;
						goto hackTest1;
					}
			}
hackTest1:			
			if (found != 0)
				++hackHeight;
		}	
		hackOffsets[hack+1] = hackHeight;
	}
	
	*allocationHeight = hackOffsets[hackCount];	
}






void diaToHdia(
	void *hdiaValues,
	int *hdiaOffsets,
	const int *hackOffsets,
	int hackSize,
	const void* diaValues,
	const int* diaOffsets,
	int diaValuesPitch,	
	int diagonals,
	int rowsCount,
	spgpuType_t valuesType
	)
{
	int i,r,s;
	int hack;
	int hackCount = getHdiaHacksCount(hackSize, rowsCount);
	
	// Compute offsets
	int hackOffsetsSize = hackCount + 1;
	
	size_t elementSize = spgpuSizeOf(valuesType);
	
	for (hack=0; hack<hackCount; ++hack)
	{
		int posOffset = hackOffsets[hack];
		
		int hackHeight = 0;
		for (i=0; i<diagonals; ++i)
		{
			int found = 0;
			for (r=0; r<hackSize; ++r)
			{
				int row = hack*hackSize + r;
				
				if (row >= rowsCount)
					break;
			
				const void* val = diaValues + elementSize*(row + i*diaValuesPitch);
				
				for (s=0; s<elementSize; ++s)
					if (*(char*)(val+s) != 0)
					{
						found = 1;
						goto hackTest2;
					}
			}
hackTest2:			
			if (found != 0)
			{
				// use hdiaOffsets to temporarely store i, instead of diaOffsets[i]
				hdiaOffsets[posOffset + hackHeight++] = i;

			}
		}
	}
	
	// Copy values
	for (hack=0; hack<hackCount; ++hack)
	{
		// get diagonal offset
		int posOffset = hackOffsets[hack];
		int hackDiags = hackOffsets[hack+1] - posOffset;
		
		for (i=0; i<hackDiags; ++i)
		{
			int diagPosInsideDia = hdiaOffsets[posOffset + i];
			int diagOffset;
			
			// reupdate hdiOffsets with the correct value
			hdiaOffsets[posOffset + i] = diagOffset = diaOffsets[diagPosInsideDia];
			
			for (r=0; r<hackSize; ++r)
			{
				int row = hack*hackSize + r;
				
				if (row >= rowsCount)
					break;
			
				void* dest = hdiaValues + elementSize*((posOffset + i)*hackSize + r);
				const void* src = diaValues + elementSize*(row + diagPosInsideDia*diaValuesPitch);
			
				memcpy(dest, src, elementSize);
			}
		}
	}
}







void computeHdiaHackOffsetsFromCoo(
	int *allocationHeight,
	int *hackOffsets,
	int hackSize,
	int rowsCount,
	int columnsCount,
	int nonZerosCount,
	const int* cooRowIndices,
	const int* cooColsIndices
	)
{	
	int i,h;
	
	int hackCount = getHdiaHacksCount(hackSize, rowsCount);

	// Use hackOffsets to deduce hack's heights

	int* diagIdsToPos = (int*)malloc((hackSize + columnsCount - 1)*sizeof(int));
	
	hackOffsets[0] = 0;
	for (h=0; h<hackCount; ++h)
	{
		for (i=0; i<(hackSize + columnsCount - 1); ++i)
			diagIdsToPos[i] = -1;
			
		int diagonalsCount = 0;
	
		for (i=0; i<nonZerosCount; ++i)
		{
			int rowIdx = cooRowIndices[i];
			
			// It is inside current hack
			if (rowIdx/hackSize == h)
			{
				int colIdx = cooColsIndices[i];
				int diagId = colIdx - (rowIdx % hackSize);
				int diagPos = hackSize - 1 + diagId;
		
				if (diagIdsToPos[diagPos] < 0)
				{
					diagIdsToPos[diagPos] = 1;
					++diagonalsCount;
				}
			}
		}		
				
		hackOffsets[h+1] = hackOffsets[h] + diagonalsCount;
	} 
	
	*allocationHeight = hackOffsets[hackCount];	
	
	free(diagIdsToPos);
}
	
	




void cooToHdia(
	void *hdiaValues,
	int *hdiaOffsets,
	const int *hackOffsets,
	int hackSize,
	int rowsCount,
	int columnsCount,
	int nonZerosCount,
	const int* cooRowIndices,
	const int* cooColsIndices,
	const void* cooValues,
	spgpuType_t valuesType
	)
{	
	int i,h;
	
	int hackCount = getHdiaHacksCount(hackSize, rowsCount);

	int* hackDiagIdsToPos = (int*)malloc((hackSize + columnsCount - 1)*sizeof(int));
	
	for (h=0; h<hackCount; ++h)
	{
		int diagonalsCount = 0;
	
		for (i=0; i<(hackSize + columnsCount - 1); ++i)
			hackDiagIdsToPos[i] = -rowsCount;

		for (i=0; i<nonZerosCount; ++i)
		{
			int rowIdx = cooRowIndices[i];
			// It is inside current hack
			if (rowIdx/hackSize == h)
			{
				int colIdx = cooColsIndices[i];
				int globalDiagId = colIdx - rowIdx;
				int diagId = colIdx - (rowIdx % hackSize);
				int diagPos = hackSize - 1 + diagId;
		
				if (hackDiagIdsToPos[diagPos] <= -rowsCount)
				{
					hackDiagIdsToPos[diagPos] = globalDiagId;
				}
			}
		}	
	
		// Reorder diags
		for (i=0; i<(hackSize + columnsCount - 1); ++i)
		{
			int globalDiagId = hackDiagIdsToPos[i];
			if (globalDiagId > -rowsCount)
			{
			
				int diagPosInsideOffsets;
				int diagId = i - hackSize + 1;
				hackDiagIdsToPos[i] = diagPosInsideOffsets = diagonalsCount++;
				hdiaOffsets[diagPosInsideOffsets] = globalDiagId;
			}
		}

	
		hdiaOffsets += diagonalsCount;
		
		for (i=0; i<nonZerosCount; ++i)
		{
			int rowIdx = cooRowIndices[i];
			// It is inside current hack
			if (rowIdx/hackSize == h)
			{
				int colIdx = cooColsIndices[i];
				int diagId = colIdx - (rowIdx % hackSize);
		
				int diagPosInsideOffsets = hackDiagIdsToPos[hackSize - 1 + diagId];
		
				size_t elementSize = spgpuSizeOf(valuesType);
		
				void* valAddr = hdiaValues + 
					elementSize*((rowIdx % hackSize) 
						+ hackSize* (hackOffsets[h] + diagPosInsideOffsets));
		
				memcpy(valAddr, (const char*)cooValues + i*elementSize, elementSize);
			}
		}
	
	}
	
	free(hackDiagIdsToPos);
}

