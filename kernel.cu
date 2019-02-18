#include "kernel.h"
#include "structs.h"
#include <math.h>
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>
#include <cooperative_groups.h>

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#include "params.h"

//namespace cg = cooperative_groups;
using namespace cooperative_groups;




__device__ void print(unsigned int tid, unsigned int value)
{
	if(0 == tid)
	{
		printf("threadIdx.x 0, value = %d\n", value);
	}
}



/******************************************************************************/



__global__ void sortByWorkLoad(
		DTYPE* database,
		DTYPE* epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		struct gridCellLookup * gridCellLookupArr,
		DTYPE* minArr,
		unsigned int * nCells,
		unsigned int * nNonEmptyCells,
		unsigned int * gridCellNDMask,
		unsigned int * gridCellNDMaskOffsets,
		schedulingCell * sortedCells,
		DTYPE* sortedSet)
{

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(*nNonEmptyCells <= tid)
	{
		return;
	}

	int cell = gridCellLookupArr[tid].idx;
	int nbNeighborPoints = 0;
	int tmpId = indexLookupArr[ index[cell].indexmin ];

	DTYPE point[NUMINDEXEDDIM];

	unsigned int nDCellIDs[NUMINDEXEDDIM];

	unsigned int rangeFilteredCellIdsMin[NUMINDEXEDDIM];
	unsigned int rangeFilteredCellIdsMax[NUMINDEXEDDIM];

	for(int n = 0; n < NUMINDEXEDDIM; n++)
	{
		point[n] = database[tmpId * NUMINDEXEDDIM + n];
		nDCellIDs[n] = (point[n] - minArr[n]) / (*epsilon);
		unsigned int nDMinCellIDs = max(0, nDCellIDs[n] - 1);;
		unsigned int nDMaxCellIDs = min(nCells[n] - 1, nDCellIDs[n] + 1);

		bool foundMin = 0;
		bool foundMax = 0;

		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[ (n * 2) ],
				gridCellNDMask + gridCellNDMaskOffsets[ (n * 2) + 1 ] + 1, nDMinCellIDs)){ //extra +1 here is because we include the upper bound
			foundMin = 1;
		}
		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[ (n * 2) ],
				gridCellNDMask + gridCellNDMaskOffsets[ (n * 2) + 1 ] + 1, nDMaxCellIDs)){ //extra +1 here is because we include the upper bound
			foundMax = 1;
		}

		if (1 == foundMin && 1 == foundMax){
			rangeFilteredCellIdsMin[n] = nDMinCellIDs;
			rangeFilteredCellIdsMax[n] = nDMaxCellIDs;
			//printf("\nmin and max");
		}
		else if (1 == foundMin && 0 == foundMax){
			rangeFilteredCellIdsMin[n] = nDMinCellIDs;
			rangeFilteredCellIdsMax[n] = nDMinCellIDs + 1;
			//printf("\nmin not max");
		}
		else if (0 == foundMin && 1 == foundMax){
			rangeFilteredCellIdsMin[n] = nDMinCellIDs + 1;
			rangeFilteredCellIdsMax[n] = nDMaxCellIDs;
			//printf("\nmax not min");
		}
		else{
			//printf("\nneither");
			rangeFilteredCellIdsMin[n] = nDMinCellIDs + 1;
			rangeFilteredCellIdsMax[n] = nDMinCellIDs + 1;
		}
	}

	unsigned int indexes[NUMINDEXEDDIM];
	unsigned int loopRng[NUMINDEXEDDIM];

	for (loopRng[0] = rangeFilteredCellIdsMin[0]; loopRng[0] <= rangeFilteredCellIdsMax[0]; loopRng[0]++)
		for (loopRng[1] = rangeFilteredCellIdsMin[1]; loopRng[1] <= rangeFilteredCellIdsMax[1]; loopRng[1]++)
		#include "kernelloops.h"
		{
			for (int x = 0; x < NUMINDEXEDDIM; x++){
				indexes[x] = loopRng[x];
			}

			uint64_t cellID = getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);
			struct gridCellLookup tmp;
			tmp.gridLinearID = cellID;
			if (thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
			{
				struct gridCellLookup * resultBinSearch = thrust::lower_bound(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
				unsigned int GridIndex = resultBinSearch->idx;
				nbNeighborPoints += index[GridIndex].indexmax - index[GridIndex].indexmin + 1;
			}

		}

	sortedCells[tid].nbPoints = nbNeighborPoints;
	sortedCells[tid].cellId = cell;

}



/******************************************************************************/



__global__ void sortByWorkLoadLidUnicomp(
		DTYPE* database,
		DTYPE* epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		struct gridCellLookup * gridCellLookupArr,
		DTYPE* minArr,
		unsigned int * nCells,
		unsigned int * nNonEmptyCells,
		unsigned int * gridCellNDMask,
		unsigned int * gridCellNDMaskOffsets,
		schedulingCell * sortedCells,
		DTYPE* sortedSet)
{

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(*nNonEmptyCells <= tid)
	{
		return;
	}

	int cell = gridCellLookupArr[tid].idx;
	int nbNeighborPoints = 0;
	int tmpId = indexLookupArr[ index[cell].indexmin ];

	DTYPE point[NUMINDEXEDDIM];

	unsigned int nDCellIDs[NUMINDEXEDDIM];

	unsigned int rangeFilteredCellIdsMin[NUMINDEXEDDIM];
	unsigned int rangeFilteredCellIdsMax[NUMINDEXEDDIM];

	for(int n = 0; n < NUMINDEXEDDIM; n++)
	{
		point[n] = database[tmpId * NUMINDEXEDDIM + n];
		nDCellIDs[n] = (point[n] - minArr[n]) / (*epsilon);
		unsigned int nDMinCellIDs = max(0, nDCellIDs[n] - 1);;
		unsigned int nDMaxCellIDs = min(nCells[n] - 1, nDCellIDs[n] + 1);

		bool foundMin = 0;
		bool foundMax = 0;

		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[ (n * 2) ],
				gridCellNDMask + gridCellNDMaskOffsets[ (n * 2) + 1 ] + 1, nDMinCellIDs)){ //extra +1 here is because we include the upper bound
			foundMin = 1;
		}
		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[ (n * 2) ],
				gridCellNDMask + gridCellNDMaskOffsets[ (n * 2) + 1 ] + 1, nDMaxCellIDs)){ //extra +1 here is because we include the upper bound
			foundMax = 1;
		}

		if (1 == foundMin && 1 == foundMax){
			rangeFilteredCellIdsMin[n] = nDMinCellIDs;
			rangeFilteredCellIdsMax[n] = nDMaxCellIDs;
			//printf("\nmin and max");
		}
		else if (1 == foundMin && 0 == foundMax){
			rangeFilteredCellIdsMin[n] = nDMinCellIDs;
			rangeFilteredCellIdsMax[n] = nDMinCellIDs + 1;
			//printf("\nmin not max");
		}
		else if (0 == foundMin && 1 == foundMax){
			rangeFilteredCellIdsMin[n] = nDMinCellIDs + 1;
			rangeFilteredCellIdsMax[n] = nDMaxCellIDs;
			//printf("\nmax not min");
		}
		else{
			//printf("\nneither");
			rangeFilteredCellIdsMin[n] = nDMinCellIDs + 1;
			rangeFilteredCellIdsMax[n] = nDMinCellIDs + 1;
		}
	}

	unsigned int indexes[NUMINDEXEDDIM];
	unsigned int loopRng[NUMINDEXEDDIM];

	for (int x = 0; x < NUMINDEXEDDIM; x++){
		indexes[x] = nDCellIDs[x];
	}

	uint64_t originCellID = getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);

	for (loopRng[0] = rangeFilteredCellIdsMin[0]; loopRng[0] <= rangeFilteredCellIdsMax[0]; loopRng[0]++)
		for (loopRng[1] = rangeFilteredCellIdsMin[1]; loopRng[1] <= rangeFilteredCellIdsMax[1]; loopRng[1]++)
		#include "kernelloops.h"
		{
			for (int x = 0; x < NUMINDEXEDDIM; x++){
				indexes[x] = loopRng[x];
			}

			uint64_t cellID = getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);
			if(originCellID <= cellID)
			{
				struct gridCellLookup tmp;
				tmp.gridLinearID = cellID;
				if (thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
				{
					struct gridCellLookup * resultBinSearch = thrust::lower_bound(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
					unsigned int GridIndex = resultBinSearch->idx;
					nbNeighborPoints += index[GridIndex].indexmax - index[GridIndex].indexmin + 1;
				}
			}

		}

	sortedCells[tid].nbPoints = nbNeighborPoints;
	sortedCells[tid].cellId = cell;

}



/******************************************************************************/



__device__ uint64_t getLinearID_nDimensionsGPU(
		unsigned int * indexes,
		unsigned int * dimLen,
		unsigned int nDimensions)
{
    uint64_t offset = 0;
	uint64_t multiplier = 1;

	for (int i = 0; i < nDimensions; i++)
	{
		offset += (uint64_t) indexes[i] * multiplier;
		multiplier *= dimLen[i];
	}

	return offset;
}



/******************************************************************************/



__device__ unsigned int binary_search(
		const unsigned int * threadArray,
		unsigned int begin,
		unsigned int end,
		const unsigned int value)
{
	unsigned int mid = (begin + end) / 2;
	if(threadArray[mid] <= value && value < threadArray[mid + 1])
	{
		return mid;
	}else{
		if(threadArray[mid] < value)
		{
			return binary_search(threadArray, mid + 1, end, value);
		}else{
			return binary_search(threadArray, begin, mid - 1, value);
		}
	}


	/*
	while(begin <= end)
	{
		unsigned int mid = (begin + end) / 2;
		if(threadArray[mid] <= value && value < threadArray[mid + 1])
		{
			(*tPerPoint) = threadArray[mid + 1] - threadArray[mid];
			return mid;
		}else{
			if(threadArray[mid] < value)
			{
				begin = mid + 1;
			}else{
				end = mid - 1;
			}
		}
	}
	(*tPerPoint) = 1;
	return end;
	*/
}



/******************************************************************************/



__forceinline__ __device__ void evalPoint(
		unsigned int* indexLookupArr,
		int k,
		DTYPE* database,
		DTYPE* epsilon,
		DTYPE* point,
		unsigned int* cnt,
		int* pointIDKey,
		int* pointInDistVal,
		int pointIdx,
		bool differentCell)
{
	DTYPE runningTotalDist = 0;
	unsigned int dataIdx = indexLookupArr[k];

	for(int l = 0; l < GPUNUMDIM; l++){
		runningTotalDist += ( database[dataIdx * GPUNUMDIM + l] - point[l])
				* (database[dataIdx * GPUNUMDIM + l] - point[l] );
	}

	if(sqrt(runningTotalDist) <= (*epsilon)){
	//if(runningTotalDist <= ((*epsilon) * (*epsilon))){
		unsigned int idx = atomicAdd(cnt, int(1));
		// printf("tid = %d, tidx = %d, idx = %d\n", blockIdx.x * BLOCKSIZE + threadIdx.x, threadIdx.x, idx);
		pointIDKey[idx] = pointIdx; // --> HERE
		pointInDistVal[idx] = dataIdx;

		if(differentCell) {
			unsigned int idx = atomicAdd(cnt, int(1));
			pointIDKey[idx] = pointIdx;
			pointInDistVal[idx] = dataIdx;
		}
	}
}



/******************************************************************************/



__device__ void evaluateCell(
		unsigned int* nCells,
		unsigned int* indexes,
		struct gridCellLookup * gridCellLookupArr,
		unsigned int* nNonEmptyCells,
		DTYPE* database, DTYPE* epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		DTYPE* point, unsigned int* cnt,
		int* pointIDKey,
		int* pointInDistVal,
		int pointIdx,
		bool differentCell,
		unsigned int* nDCellIDs)
{
	//compare the linear ID with the gridCellLookupArr to determine if the cell is non-empty: this can happen because one point says
	//a cell in a particular dimension is non-empty, but that's because it was related to a different point (not adjacent to the query point)
	uint64_t calcLinearID = getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);

	struct gridCellLookup tmp;
	tmp.gridLinearID = calcLinearID;
	//find if the cell is non-empty
	if(thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
	{
		//compute the neighbors for the adjacent non-empty cell
		struct gridCellLookup * resultBinSearch = thrust::lower_bound(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
		unsigned int GridIndex = resultBinSearch->idx;

		for(int k = index[GridIndex].indexmin; k <= index[GridIndex].indexmax; k++){
			evalPoint(indexLookupArr, k, database, epsilon, point, cnt, pointIDKey, pointInDistVal, pointIdx, differentCell);
		}
	}
}



/******************************************************************************/



__forceinline__ __device__ void evalPointUnicompOrigin(
		unsigned int* indexLookupArr,
		int k,
		DTYPE* database,
		DTYPE* epsilon,
		DTYPE* point,
		unsigned int* cnt,
		int* pointIDKey,
		int* pointInDistVal,
		int pointIdx)
{
	DTYPE runningTotalDist = 0;
	unsigned int dataIdx = indexLookupArr[k];

	for (int l = 0; l < GPUNUMDIM; l++)
	{
		runningTotalDist += (database[dataIdx * GPUNUMDIM + l] - point[l]) * (database[dataIdx * GPUNUMDIM + l] - point[l]);
	}

	if (sqrt(runningTotalDist) <= (*epsilon)){
	//if(runningTotalDist <= ((*epsilon) * (*epsilon))){
		unsigned int idx = atomicAdd(cnt, int(1));
		//printf("\n\nLOL CA VA TROP LOIN (%d)\n\n", idx);
		// assert(idx < 2000000);
		pointIDKey[idx] = pointIdx; // --> HERE
		pointInDistVal[idx] = dataIdx;
	}
}



/******************************************************************************/



__device__ void evaluateCellUnicompOrigin(
		unsigned int* nCells,
		unsigned int* indexes,
		struct gridCellLookup * gridCellLookupArr,
		unsigned int* nNonEmptyCells,
		DTYPE* database, DTYPE* epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		DTYPE* point, unsigned int* cnt,
		int* pointIDKey,
		int* pointInDistVal,
		int pointIdx,
		unsigned int* nDCellIDs,
		unsigned int nbThreads,
		unsigned int numThread)
{
	//compare the linear ID with the gridCellLookupArr to determine if the cell is non-empty: this can happen because one point says
	//a cell in a particular dimension is non-empty, but that's because it was related to a different point (not adjacent to the query point)
	uint64_t calcLinearID = getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);

	struct gridCellLookup tmp;
	tmp.gridLinearID = calcLinearID;
	//find if the cell is non-empty
	if (thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
	{
		//compute the neighbors for the adjacent non-empty cell
		struct gridCellLookup * resultBinSearch = thrust::lower_bound(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
		unsigned int GridIndex = resultBinSearch->idx;

		int begin = index[GridIndex].indexmin;
		int end = index[GridIndex].indexmax;
		int nbElem = end - begin + 1;
		if(numThread < nbElem)
		{
			int size = nbElem / nbThreads;
			int oneMore = nbElem - (size * nbThreads);
			if(nbElem == (size * nbThreads))
			{
				begin += size * numThread;
				end = begin + size - 1;
			}else{
				begin += numThread * size + ((numThread < oneMore)?numThread:oneMore);
				end = begin + size - 1 + (numThread < oneMore);
			}

			for(int k = begin; k <= end; k++)
			{
				evalPointUnicompOrigin(indexLookupArr, k, database, epsilon, point, cnt, pointIDKey, pointInDistVal, pointIdx);
			}
		}
	}
}



/******************************************************************************/



__forceinline__ __device__ void evalPointUnicompAdjacent(
		unsigned int* indexLookupArr,
		int k,
		DTYPE* database,
		DTYPE* epsilon,
		DTYPE* point,
		unsigned int* cnt,
		int* pointIDKey,
		int* pointInDistVal,
		int pointIdx)
{
	DTYPE runningTotalDist = 0;
	unsigned int dataIdx = indexLookupArr[k];

	for (int l = 0; l < GPUNUMDIM; l++)
	{
		runningTotalDist += (database[dataIdx * GPUNUMDIM + l] - point[l]) * (database[dataIdx * GPUNUMDIM + l] - point[l]);
	}

	if (sqrt(runningTotalDist) <= (*epsilon)){
	//if(runningTotalDist <= ((*epsilon) * (*epsilon))){
		unsigned int idx = atomicAdd(cnt, int(2));
		pointIDKey[idx] = pointIdx;
		pointInDistVal[idx] = dataIdx;
		pointIDKey[idx + 1] = dataIdx;
		pointInDistVal[idx + 1] = pointIdx;
	}
}



/******************************************************************************/



__device__ void evaluateCellUnicompAdjacent(
		unsigned int* nCells,
		unsigned int* indexes,
		struct gridCellLookup * gridCellLookupArr,
		unsigned int* nNonEmptyCells,
		DTYPE* database, DTYPE* epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		DTYPE* point, unsigned int* cnt,
		int* pointIDKey,
		int* pointInDistVal,
		int pointIdx,
		unsigned int* nDCellIDs,
		unsigned int nbThreads,
		unsigned int numThread)
{
	//compare the linear ID with the gridCellLookupArr to determine if the cell is non-empty: this can happen because one point says
	//a cell in a particular dimension is non-empty, but that's because it was related to a different point (not adjacent to the query point)
	uint64_t calcLinearID = getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);

	struct gridCellLookup tmp;
	tmp.gridLinearID = calcLinearID;
	//find if the cell is non-empty
	if (thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
	{
		//compute the neighbors for the adjacent non-empty cell
		struct gridCellLookup * resultBinSearch = thrust::lower_bound(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
		unsigned int GridIndex = resultBinSearch->idx;

		int begin = index[GridIndex].indexmin;
		int end = index[GridIndex].indexmax;
		int nbElem = end - begin + 1;
		if(numThread < nbElem)
		{
			int size = nbElem / nbThreads;
			int oneMore = nbElem - (size * nbThreads);
			if(nbElem == (size * nbThreads))
			{
				begin += size * numThread;
				end = begin + size - 1;
			}else{
				begin += numThread * size + ((numThread < oneMore)?numThread:oneMore);
				end = begin + size - 1 + (numThread < oneMore);
			}

			for(int k = begin; k <= end; k++)
			{
				evalPointUnicompAdjacent(indexLookupArr, k, database, epsilon, point, cnt, pointIDKey, pointInDistVal, pointIdx);
			}
		}
	}
}





/******************************************************************************/





__global__ void kernelNDGridIndexBatchEstimatorOLD(
		unsigned int *debug1,
		unsigned int *debug2,
		unsigned int *N,
		unsigned int * sampleOffset,
		DTYPE* database,
		DTYPE* epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		struct gridCellLookup * gridCellLookupArr,
		DTYPE* minArr,
		unsigned int * nCells,
		unsigned int * cnt,
		unsigned int * nNonEmptyCells,
		unsigned int * gridCellNDMask,
		unsigned int * gridCellNDMaskOffsets)
{

	unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE);

	if (tid>=*N){
		return;
	}

	unsigned int pointID = tid  * (*sampleOffset) * (GPUNUMDIM);

	//make a local copy of the point
	DTYPE point[GPUNUMDIM];
	for (int i = 0; i < GPUNUMDIM; i++){
		point[i] = database[pointID + i];
	}

	//calculate the coords of the Cell for the point
	//and the min/max ranges in each dimension
	unsigned int nDCellIDs[NUMINDEXEDDIM];
	unsigned int nDMinCellIDs[NUMINDEXEDDIM];
	unsigned int nDMaxCellIDs[NUMINDEXEDDIM];

	for (int i = 0; i < NUMINDEXEDDIM; i++)
	{
		nDCellIDs[i] = (point[i] - minArr[i]) / (*epsilon);
		nDMinCellIDs[i] = max(0, nDCellIDs[i] - 1); //boundary conditions (don't go beyond cell 0)
		nDMaxCellIDs[i] = min(nCells[i] - 1, nDCellIDs[i] + 1); //boundary conditions (don't go beyond the maximum number of cells)

	}

	///////////////////////////
	//Take the intersection of the ranges for each dimension between
	//the point and the filtered set of cells in each dimension
	//Ranges in a given dimension that have points in them that are non-empty in a dimension will be tested
	///////////////////////////

	unsigned int rangeFilteredCellIdsMin[NUMINDEXEDDIM];
	unsigned int rangeFilteredCellIdsMax[NUMINDEXEDDIM];
	//compare the point's range of cell IDs in each dimension to the filter mask
	//only 2 possible values (you always find the middle point in the range), because that's the cell of the point itself
	bool foundMin = 0;
	bool foundMax = 0;

	//we go throgh each dimension and compare the range of the query points min/max cell ids to the filtered ones
	//find out which ones in the range exist based on the min/max
	//then determine the appropriate ranges
	for (int i=0; i<NUMINDEXEDDIM; i++)
	{
		foundMin = 0;
		foundMax = 0;
		//for each dimension
		//OPTIMIZE: WITH BINARY SEARCH LATER
		// for (int dimFilterRng=gridCellNDMaskOffsets[(i*2)]; dimFilterRng<=gridCellNDMaskOffsets[(i*2)+1]; dimFilterRng++){
		// 	if (gridCellNDMask[dimFilterRng]==nDMinCellIDs[i])
		// 		foundMin=1;
		// 	if (gridCellNDMask[dimFilterRng]==nDMaxCellIDs[i])
		// 		foundMax=1;
		// }

		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) ],
				gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) + 1 ] + 1, nDMinCellIDs[i])){ //extra +1 here is because we include the upper bound
			foundMin=1;
		}
		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) ],
				gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) + 1 ] + 1, nDMaxCellIDs[i])){ //extra +1 here is because we include the upper bound
			foundMax=1;
		}

		// cases:
		// found the min and max
		// found the min and not max
		//found the max and not the min
		//you don't find the min or max -- then only check the mid
		//you always find the mid because it's in the cell of the point you're looking for

		//NEED TO OPTIMIZE STILL
		if (foundMin == 1 && foundMax == 1){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs[i];
			rangeFilteredCellIdsMax[i] = nDMaxCellIDs[i];
			//printf("\nmin and max");
		}
		else if (foundMin == 1 && foundMax == 0){
			rangeFilteredCellIdsMin[i]  =nDMinCellIDs[i];
			rangeFilteredCellIdsMax[i] = nDMinCellIDs[i] + 1;
			//printf("\nmin not max");
		}
		else if (foundMin == 0 && foundMax == 1){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs[i] + 1;
			rangeFilteredCellIdsMax[i] = nDMaxCellIDs[i];
			//printf("\nmax not min");
		}
		//dont find either min or max
		//get middle value only
		else{
			//printf("\nneither");
			rangeFilteredCellIdsMin[i] = nDMinCellIDs[i] + 1;
			rangeFilteredCellIdsMax[i] = nDMinCellIDs[i] + 1;
		}
	}

	///////////////////////////////////////
	//End taking intersection
	//////////////////////////////////////

	unsigned int indexes[NUMINDEXEDDIM];
	unsigned int loopRng[NUMINDEXEDDIM];

	for (loopRng[0] = rangeFilteredCellIdsMin[0]; loopRng[0] <= rangeFilteredCellIdsMax[0]; loopRng[0]++)
		for (loopRng[1] = rangeFilteredCellIdsMin[1]; loopRng[1] <= rangeFilteredCellIdsMax[1]; loopRng[1]++)
		#include "kernelloops.h"
		{ //beginning of loop body


			for (int x = 0; x < NUMINDEXEDDIM; x++)
			{
				indexes[x] = loopRng[x];
				// if (tid==0)
				// 	printf("\ndim: %d, indexes: %d",x, indexes[x]);
			}

			uint64_t calcLinearID = getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);
			//compare the linear ID with the gridCellLookupArr to determine if the cell is non-empty: this can happen because one point says
			//a cell in a particular dimension is non-empty, but that's because it was related to a different point (not adjacent to the query point)

			struct gridCellLookup tmp;
			tmp.gridLinearID = calcLinearID;

			if (thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
			{
				//in the GPU implementation we go directly to computing neighbors so that we don't need to
				//store a buffer of the cells to check
				//cellsToCheck->push_back(calcLinearID);

				//HERE WE COMPUTE THE NEIGHBORS FOR THE CELL
				//XXXXXXXXXXXXXXXXXXXXXXXXX

				struct gridCellLookup * resultBinSearch=thrust::lower_bound(thrust::seq, gridCellLookupArr, gridCellLookupArr+(*nNonEmptyCells), gridCellLookup(tmp));
				unsigned int GridIndex = resultBinSearch->idx;

				for (int k = index[GridIndex].indexmin; k <= index[GridIndex].indexmax; k++)
				{
					DTYPE runningTotalDist = 0;
					unsigned int dataIdx = indexLookupArr[k];

					for (int l = 0; l < GPUNUMDIM; l++)
					{
						runningTotalDist += (database[dataIdx * GPUNUMDIM + l]  -point[l])
								* (database[dataIdx * GPUNUMDIM + l] - point[l]);
					}


					if (sqrt(runningTotalDist) <= (*epsilon)){
						unsigned int idx = atomicAdd(cnt, int(1));
						// pointIDKey[idx]=tid;
						// pointInDistVal[idx]=i;
						//neighborTableCPUPrototype[queryPoint].neighbors.push_back(dataIdx);

					}
				}
			}
			//printf("\nLinear id: %d",calcLinearID);
		} //end loop body

}



__device__ int counterEstimator = 0;

__global__ void kernelNDGridIndexWorkQueueBatchEstimatorOLD(
		unsigned int *debug1,
		unsigned int *debug2,
		unsigned int *N,
		unsigned int * sampleOffset,
		DTYPE* database,
		DTYPE* sortedCells,
		DTYPE* epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		struct gridCellLookup * gridCellLookupArr,
		DTYPE* minArr,
		unsigned int * nCells,
		unsigned int * cnt,
		unsigned int * nNonEmptyCells,
		unsigned int * gridCellNDMask,
		unsigned int * gridCellNDMaskOffsets)
{

	unsigned int tid = blockIdx.x * BLOCKSIZE + threadIdx.x;

	if (*N <= tid){
		return;
	}

	//unsigned int pointID = tid  * (*sampleOffset) * (GPUNUMDIM);
	unsigned int pointID = atomicAdd(&counterEstimator, int(1));

	//make a local copy of the point
	DTYPE point[GPUNUMDIM];
	for (int i = 0; i < GPUNUMDIM; i++){
		// point[i] = database[pointID + i];
		point[i] = sortedCells[pointID * GPUNUMDIM + i];
	}

	//calculate the coords of the Cell for the point
	//and the min/max ranges in each dimension
	unsigned int nDCellIDs[NUMINDEXEDDIM];
	unsigned int nDMinCellIDs[NUMINDEXEDDIM];
	unsigned int nDMaxCellIDs[NUMINDEXEDDIM];

	for (int i = 0; i < NUMINDEXEDDIM; i++)
	{
		nDCellIDs[i] = (point[i] - minArr[i]) / (*epsilon);
		nDMinCellIDs[i] = max(0, nDCellIDs[i] - 1); //boundary conditions (don't go beyond cell 0)
		nDMaxCellIDs[i] = min(nCells[i] - 1, nDCellIDs[i] + 1); //boundary conditions (don't go beyond the maximum number of cells)
	}

	///////////////////////////
	//Take the intersection of the ranges for each dimension between
	//the point and the filtered set of cells in each dimension
	//Ranges in a given dimension that have points in them that are non-empty in a dimension will be tested
	///////////////////////////

	unsigned int rangeFilteredCellIdsMin[NUMINDEXEDDIM];
	unsigned int rangeFilteredCellIdsMax[NUMINDEXEDDIM];
	//compare the point's range of cell IDs in each dimension to the filter mask
	//only 2 possible values (you always find the middle point in the range), because that's the cell of the point itself
	bool foundMin = 0;
	bool foundMax = 0;

	//we go throgh each dimension and compare the range of the query points min/max cell ids to the filtered ones
	//find out which ones in the range exist based on the min/max
	//then determine the appropriate ranges
	for (int i=0; i<NUMINDEXEDDIM; i++)
	{
		foundMin = 0;
		foundMax = 0;

		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) ],
				gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) + 1 ] + 1, nDMinCellIDs[i])){ //extra +1 here is because we include the upper bound
			foundMin=1;
		}
		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) ],
				gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) + 1 ] + 1, nDMaxCellIDs[i])){ //extra +1 here is because we include the upper bound
			foundMax=1;
		}

		if (foundMin == 1 && foundMax == 1){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs[i];
			rangeFilteredCellIdsMax[i] = nDMaxCellIDs[i];
		}
		else if (foundMin == 1 && foundMax == 0){
			rangeFilteredCellIdsMin[i]  =nDMinCellIDs[i];
			rangeFilteredCellIdsMax[i] = nDMinCellIDs[i] + 1;
		}
		else if (foundMin == 0 && foundMax == 1){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs[i] + 1;
			rangeFilteredCellIdsMax[i] = nDMaxCellIDs[i];
		}
		else{
			rangeFilteredCellIdsMin[i] = nDMinCellIDs[i] + 1;
			rangeFilteredCellIdsMax[i] = nDMinCellIDs[i] + 1;
		}
	}

	///////////////////////////////////////
	//End taking intersection
	//////////////////////////////////////

	unsigned int indexes[NUMINDEXEDDIM];
	unsigned int loopRng[NUMINDEXEDDIM];

	for (loopRng[0] = rangeFilteredCellIdsMin[0]; loopRng[0] <= rangeFilteredCellIdsMax[0]; loopRng[0]++)
		for (loopRng[1] = rangeFilteredCellIdsMin[1]; loopRng[1] <= rangeFilteredCellIdsMax[1]; loopRng[1]++)
		#include "kernelloops.h"
		{ //beginning of loop body

			for (int x = 0; x < NUMINDEXEDDIM; x++)
			{
				indexes[x] = loopRng[x];
			}

			uint64_t calcLinearID = getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);
			//compare the linear ID with the gridCellLookupArr to determine if the cell is non-empty: this can happen because one point says
			//a cell in a particular dimension is non-empty, but that's because it was related to a different point (not adjacent to the query point)

			struct gridCellLookup tmp;
			tmp.gridLinearID = calcLinearID;

			if (thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
			{
				//in the GPU implementation we go directly to computing neighbors so that we don't need to
				//store a buffer of the cells to check
				//cellsToCheck->push_back(calcLinearID);

				//HERE WE COMPUTE THE NEIGHBORS FOR THE CELL
				//XXXXXXXXXXXXXXXXXXXXXXXXX

				struct gridCellLookup * resultBinSearch = thrust::lower_bound(thrust::seq, gridCellLookupArr, gridCellLookupArr+(*nNonEmptyCells), gridCellLookup(tmp));
				unsigned int GridIndex = resultBinSearch->idx;

				for (int k = index[GridIndex].indexmin; k <= index[GridIndex].indexmax; k++)
				{
					DTYPE runningTotalDist = 0;
					unsigned int dataIdx = indexLookupArr[k];

					for (int l = 0; l < GPUNUMDIM; l++)
					{
						runningTotalDist += (database[dataIdx * GPUNUMDIM + l]  - point[l])
								* (database[dataIdx * GPUNUMDIM + l] - point[l]);
					}


					if (sqrt(runningTotalDist) <= (*epsilon)){
						unsigned int idx = atomicAdd(cnt, int(1));
						// pointIDKey[idx]=tid;
						// pointInDistVal[idx]=i;
						//neighborTableCPUPrototype[queryPoint].neighbors.push_back(dataIdx);

					}
				}
			}
			//printf("\nLinear id: %d",calcLinearID);
		} //end loop body
}





// Global memory kernel - Initial version ("GPU")
__global__ void kernelNDGridIndexGlobal(
		unsigned int *debug1,
		unsigned int *debug2,
		unsigned int *N,
		unsigned int * offset,
		unsigned int *batchNum,
		DTYPE* database,
		DTYPE* sortedCells,
		DTYPE* epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		struct gridCellLookup * gridCellLookupArr,
		DTYPE* minArr,
		unsigned int * nCells,
		unsigned int * cnt,
		unsigned int * nNonEmptyCells,
		unsigned int * gridCellNDMask,
		unsigned int * gridCellNDMaskOffsets,
		int * pointIDKey,
		int * pointInDistVal)
{

	unsigned int tid = (blockIdx.x * BLOCKSIZE + threadIdx.x) / THREADPERPOINT;

	if (*N <= tid){
		return;
	}

	//the point id in the dataset
	unsigned int pointIdx = tid * (*offset) + (*batchNum);
	//The offset into the database, taking into consideration the length of each dimension
	unsigned int pointOffset = tid * (GPUNUMDIM) * (*offset) + (*batchNum) * (GPUNUMDIM);

	//printf("tid %d, working on point %d,\n", tid, pointIdx);

	//make a local copy of the point
	DTYPE point[GPUNUMDIM];
	#if SORT_BY_WORKLOAD
		for (int i = 0; i < GPUNUMDIM; i++){
			point[i] = sortedCells[pointOffset + i];
		}
	#else
		for (int i = 0; i < GPUNUMDIM; i++){
			point[i] = database[pointOffset + i];
		}
	#endif

	//calculate the coords of the Cell for the point
	//and the min/max ranges in each dimension
	unsigned int nDCellIDs[NUMINDEXEDDIM];
	unsigned int nDMinCellIDs[NUMINDEXEDDIM];
	unsigned int nDMaxCellIDs[NUMINDEXEDDIM];
	for (int i = 0; i < NUMINDEXEDDIM; i++){
		nDCellIDs[i] = (point[i] - minArr[i]) / (*epsilon);
		nDMinCellIDs[i] = max(0, nDCellIDs[i] - 1); //boundary conditions (don't go beyond cell 0)
		nDMaxCellIDs[i] = min(nCells[i] - 1, nDCellIDs[i] + 1); //boundary conditions (don't go beyond the maximum number of cells)
	}

	///////////////////////////
	//Take the intersection of the ranges for each dimension between
	//the point and the filtered set of cells in each dimension
	//Ranges in a given dimension that have points in them that are non-empty in a dimension will be tested
	///////////////////////////

	unsigned int rangeFilteredCellIdsMin[NUMINDEXEDDIM];
	unsigned int rangeFilteredCellIdsMax[NUMINDEXEDDIM];
	//compare the point's range of cell IDs in each dimension to the filter mask
	//only 2 possible values (you always find the middle point in the range), because that's the cell of the point itself
	bool foundMin = 0;
	bool foundMax = 0;

	//we go through each dimension and compare the range of the query points min/max cell ids to the filtered ones
	//find out which ones in the range exist based on the min/max
	//then determine the appropriate ranges
	for (int i = 0; i < NUMINDEXEDDIM; i++)
	{
		foundMin = 0;
		foundMax = 0;

		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[(i * 2)],
				gridCellNDMask + gridCellNDMaskOffsets[(i * 2) + 1] + 1, nDMinCellIDs[i])){ //extra +1 here is because we include the upper bound
			foundMin = 1;
		}
		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[(i * 2)],
				gridCellNDMask + gridCellNDMaskOffsets[(i * 2) + 1] + 1, nDMaxCellIDs[i])){ //extra +1 here is because we include the upper bound
			foundMax = 1;
		}

		// cases:
		// found the min and max
		// found the min and not max
		//found the max and not the min
		//you don't find the min or max -- then only check the mid
		//you always find the mid because it's in the cell of the point you're looking for

		//NEED TO OPTIMIZE STILL
		if (1 == foundMin && 1 == foundMax){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs[i];
			rangeFilteredCellIdsMax[i] = nDMaxCellIDs[i];
			//printf("\nmin and max");
		}
		else if (1 == foundMin && 0 == foundMax){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs[i];
			rangeFilteredCellIdsMax[i] = nDMinCellIDs[i] + 1;
			//printf("\nmin not max");
		}
		else if (0 == foundMin && 1 == foundMax){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs[i] + 1;
			rangeFilteredCellIdsMax[i] = nDMaxCellIDs[i];
			//printf("\nmax not min");
		}
		//dont find either min or max
		//get middle value only
		else{
			//printf("\nneither");
			rangeFilteredCellIdsMin[i] = nDMinCellIDs[i] + 1;
			rangeFilteredCellIdsMax[i] = nDMinCellIDs[i] + 1;
		}
	}

	///////////////////////////////////////
	//End taking intersection
	//////////////////////////////////////

	unsigned int indexes[NUMINDEXEDDIM];
	unsigned int loopRng[NUMINDEXEDDIM];

	for (loopRng[0] = rangeFilteredCellIdsMin[0]; loopRng[0] <= rangeFilteredCellIdsMax[0]; loopRng[0]++)
		for (loopRng[1] = rangeFilteredCellIdsMin[1]; loopRng[1] <= rangeFilteredCellIdsMax[1]; loopRng[1]++)
		#include "kernelloops.h"
		{ //beginning of loop body

			for (int x = 0; x < NUMINDEXEDDIM; x++){
				indexes[x] = loopRng[x];
			}

			#if THREADPERPOINT > 1
				evaluateCellUnicompOrigin(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index, indexLookupArr,
					point, cnt, pointIDKey, pointInDistVal, pointIdx, nDCellIDs, THREADPERPOINT, threadIdx.x % THREADPERPOINT);
			#else
				evaluateCell(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon,
						index, indexLookupArr, point, cnt, pointIDKey, pointInDistVal, pointIdx, false, nDCellIDs);
			#endif

		} //end loop body

}





// Global memory kernel - Unicomp version ("Unicomp")
__global__ void kernelNDGridIndexGlobalUnicomp(
		unsigned int *debug1,
		unsigned int *debug2,
		unsigned int *N,
		unsigned int * offset,
		unsigned int *batchNum,
		DTYPE* database,
		DTYPE* sortedCells,
		DTYPE* epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		struct gridCellLookup * gridCellLookupArr,
		DTYPE* minArr,
		unsigned int * nCells,
		unsigned int * cnt,
		unsigned int * nNonEmptyCells,
		unsigned int * gridCellNDMask,
		unsigned int * gridCellNDMaskOffsets,
		int * pointIDKey,
		int * pointInDistVal)
{

	unsigned int tid = (blockIdx.x * BLOCKSIZE + threadIdx.x) / THREADPERPOINT;

	if (*N <= tid){
		return;
	}

	//the point id in the dataset
	unsigned int pointIdx = tid * (*offset) + (*batchNum);
	//The offset into the database, taking into consideration the length of each dimension
	unsigned int pointOffset = tid * (GPUNUMDIM) * (*offset) + (*batchNum) * (GPUNUMDIM);

	//make a local copy of the point
	DTYPE point[GPUNUMDIM];
	#if SORT_BY_WORKLOAD
		for (int i = 0; i < GPUNUMDIM; i++){
			point[i] = sortedCells[pointOffset + i];
		}
	#else
		for (int i = 0; i < GPUNUMDIM; i++){
			point[i] = database[pointOffset + i];
		}
	#endif

	//calculate the coords of the Cell for the point
	//and the min/max ranges in each dimension
	unsigned int nDCellIDs[NUMINDEXEDDIM];
	unsigned int nDMinCellIDs[NUMINDEXEDDIM];
	unsigned int nDMaxCellIDs[NUMINDEXEDDIM];
	for (int i = 0; i < NUMINDEXEDDIM; i++){
		nDCellIDs[i] = (point[i] - minArr[i]) / (*epsilon);
		nDMinCellIDs[i] = max(0, nDCellIDs[i]  - 1); //boundary conditions (don't go beyond cell 0)
		nDMaxCellIDs[i] = min(nCells[i] - 1, nDCellIDs[i] + 1); //boundary conditions (don't go beyond the maximum number of cells)

	}

	///////////////////////////
	//Take the intersection of the ranges for each dimension between
	//the point and the filtered set of cells in each dimension
	//Ranges in a given dimension that have points in them that are non-empty in a dimension will be tested
	///////////////////////////

	unsigned int rangeFilteredCellIdsMin[NUMINDEXEDDIM];
	unsigned int rangeFilteredCellIdsMax[NUMINDEXEDDIM];
	//compare the point's range of cell IDs in each dimension to the filter mask
	//only 2 possible values (you always find the middle point in the range), because that's the cell of the point itself
	bool foundMin = 0;
	bool foundMax = 0;

	//we go through each dimension and compare the range of the query points min/max cell ids to the filtered ones
	//find out which ones in the range exist based on the min/max
	//then determine the appropriate ranges
	for (int i = 0; i < NUMINDEXEDDIM; i++)
	{
		foundMin = 0;
		foundMax = 0;

		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) ],
			gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) + 1 ] + 1, nDMinCellIDs[i])){ //extra +1 here is because we include the upper bound
			foundMin = 1;
		}
		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) ],
			gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) + 1 ] + 1, nDMaxCellIDs[i])){ //extra +1 here is because we include the upper bound
			foundMax = 1;
		}

		// cases:
		// found the min and max
		// found the min and not max
		//found the max and not the min
		//you don't find the min or max -- then only check the mid
		//you always find the mid because it's in the cell of the point you're looking for

		//NEED TO OPTIMIZE STILL
		if (foundMin == 1 && foundMax == 1){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs[i];
			rangeFilteredCellIdsMax[i] = nDMaxCellIDs[i];
			//printf("\nmin and max");
		}
		else if (foundMin == 1 && foundMax == 0){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs[i];
			rangeFilteredCellIdsMax[i] = nDMinCellIDs[i] + 1;
			//printf("\nmin not max");
		}
		else if (foundMin == 0 && foundMax == 1){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs[i] + 1;
			rangeFilteredCellIdsMax[i] = nDMaxCellIDs[i];
			//printf("\nmax not min");
		}
		//dont find either min or max
		//get middle value only
		else{
			//printf("\nneither");
			rangeFilteredCellIdsMin[i] = nDMinCellIDs[i] + 1;
			rangeFilteredCellIdsMax[i] = nDMinCellIDs[i] + 1;
		}
	}

	///////////////////////////////////////
	//End taking intersection
	//////////////////////////////////////

    unsigned int indexes[NUMINDEXEDDIM];
    unsigned int loopRng[NUMINDEXEDDIM];

	for(int i = 0; i < NUMINDEXEDDIM; i++) {
		indexes[i] = nDCellIDs[i];
	}

	#if THREADPERPOINT > 1
		evaluateCellUnicompOrigin(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index, indexLookupArr,
			point, cnt, pointIDKey, pointInDistVal, pointIdx, nDCellIDs, THREADPERPOINT, threadIdx.x % THREADPERPOINT);
		#include "stamploopsV2.h"
	#else
		evaluateCell(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon,
				index, indexLookupArr, point, cnt, pointIDKey, pointInDistVal, pointIdx, false, nDCellIDs);
		#include "stamploops.h"
	#endif

}





// Global memory kernel - B-Unicomp version ("B-Unicomp")
__global__ void kernelNDGridIndexGlobalBUnicomp(
		unsigned int *debug1,
		unsigned int *debug2,
		unsigned int *N,
		unsigned int * offset,
		unsigned int *batchNum,
		DTYPE* database,
		DTYPE* sortedCells,
		DTYPE* epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		struct gridCellLookup * gridCellLookupArr,
		DTYPE* minArr,
		unsigned int * nCells,
		unsigned int * cnt,
		unsigned int * nNonEmptyCells,
		unsigned int * gridCellNDMask,
		unsigned int * gridCellNDMaskOffsets,
		int * pointIDKey,
		int * pointInDistVal)
{

	unsigned int tid = (blockIdx.x * BLOCKSIZE + threadIdx.x) / THREADPERPOINT;

	if (*N <= tid){
		return;
	}

	//the point id in the dataset
	unsigned int pointIdx = tid * (*offset) + (*batchNum);
	//The offset into the database, taking into consideration the length of each dimension
	unsigned int pointOffset = tid * (GPUNUMDIM) * (*offset) + (*batchNum) * (GPUNUMDIM);

	//make a local copy of the point
	DTYPE point[GPUNUMDIM];
	#if SORT_BY_WORKLOAD
		for (int i = 0; i < GPUNUMDIM; i++){
			point[i] = sortedCells[pointOffset + i];
		}
	#else
		for (int i = 0; i < GPUNUMDIM; i++){
			point[i] = database[pointOffset + i];
		}
	#endif

	//calculate the coords of the Cell for the point
	//and the min/max ranges in each dimension
	unsigned int nDCellIDs[NUMINDEXEDDIM];

	unsigned int rangeFilteredCellIdsMin[NUMINDEXEDDIM];
	unsigned int rangeFilteredCellIdsMax[NUMINDEXEDDIM];

	for (int i = 0; i < NUMINDEXEDDIM; i++){
		nDCellIDs[i] = (point[i] - minArr[i]) / (*epsilon);
		unsigned int nDMinCellIDs = max(0, nDCellIDs[i] - 1); //boundary conditions (don't go beyond cell 0)
		unsigned int nDMaxCellIDs = min(nCells[i] - 1, nDCellIDs[i] + 1); //boundary conditions (don't go beyond the maximum number of cells)

		bool foundMin = 0;
		bool foundMax = 0;

		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) ],
			gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) + 1 ] + 1, nDMinCellIDs)){ //extra +1 here is because we include the upper bound
			foundMin = 1;
		}
		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) ],
			gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) + 1 ] + 1, nDMaxCellIDs)){ //extra +1 here is because we include the upper bound
			foundMax = 1;
		}

		if (foundMin == 1 && foundMax == 1){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs;
			rangeFilteredCellIdsMax[i] = nDMaxCellIDs;
			//printf("\nmin and max");
		}
		else if (foundMin == 1 && foundMax == 0){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs;
			rangeFilteredCellIdsMax[i] = nDMinCellIDs + 1;
			//printf("\nmin not max");
		}
		else if (foundMin == 0 && foundMax == 1){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs + 1;
			rangeFilteredCellIdsMax[i] = nDMaxCellIDs;
			//printf("\nmax not min");
		}
		//dont find either min or max
		//get middle value only
		else{
			//printf("\nneither");
			rangeFilteredCellIdsMin[i] = nDMinCellIDs + 1;
			rangeFilteredCellIdsMax[i] = nDMinCellIDs + 1;
		}

	}

    unsigned int indexes[NUMINDEXEDDIM];
    unsigned int loopRng[NUMINDEXEDDIM];

	#if NUMINDEXEDDIM==2
		indexes[0] = nDCellIDs[0];
		indexes[1] = nDCellIDs[1];
		unsigned int colorId = nDCellIDs[0] + nDCellIDs[1];
		evaluateCellUnicompOrigin(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index, indexLookupArr,
			point, cnt, pointIDKey, pointInDistVal, pointIdx, nDCellIDs, THREADPERPOINT, threadIdx.x % THREADPERPOINT);

		for(loopRng[0] = rangeFilteredCellIdsMin[0]; loopRng[0] <= rangeFilteredCellIdsMax[0]; loopRng[0]++)
			for(loopRng[1] = rangeFilteredCellIdsMin[1]; loopRng[1] <= rangeFilteredCellIdsMax[1]; loopRng[1]++)
			{
				if( ( (1 == colorId % 2) && (nDCellIDs[1] <= loopRng[1]) && (nDCellIDs[0] != loopRng[0]) )
					|| ( (0 == colorId % 2) && ((nDCellIDs[1] < loopRng[1]) || (loopRng[1] < nDCellIDs[1] && loopRng[0] == nDCellIDs[0])) ) ) // ( odd => red pattern ) || ( even => green pattern )
				{
					indexes[0] = loopRng[0];
					indexes[1] = loopRng[1];
					evaluateCellUnicompAdjacent(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index, indexLookupArr,
						point, cnt, pointIDKey, pointInDistVal, pointIdx, nDCellIDs, THREADPERPOINT, threadIdx.x % THREADPERPOINT);
				}
			}
	#else
		#if NUMINDEXEDDIM==3
			indexes[0] = nDCellIDs[0];
			indexes[1] = nDCellIDs[1];
			indexes[2] = nDCellIDs[2];
			unsigned int colorId = nDCellIDs[0] + nDCellIDs[1] + nDCellIDs[2];
			evaluateCellUnicompOrigin(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index, indexLookupArr,
				point, cnt, pointIDKey, pointInDistVal, pointIdx, nDCellIDs, THREADPERPOINT, threadIdx.x % THREADPERPOINT);

			for(loopRng[0] = rangeFilteredCellIdsMin[0]; loopRng[0] <= rangeFilteredCellIdsMax[0]; loopRng[0]++)
				for(loopRng[1] = rangeFilteredCellIdsMin[1]; loopRng[1] <= rangeFilteredCellIdsMax[1]; loopRng[1]++)
					for(loopRng[2] = rangeFilteredCellIdsMin[2]; loopRng[2] <= rangeFilteredCellIdsMax[2]; loopRng[2]++)
					{
						if( ( (1 == colorId % 2) && ( (nDCellIDs[0] != loopRng[0] && nDCellIDs[1] <= loopRng[1] && nDCellIDs[2] <= loopRng[2])
								|| (nDCellIDs[0] == loopRng[0] && nDCellIDs[1] <= loopRng[1] && nDCellIDs[2] < loopRng[2])
								|| (nDCellIDs[1] < loopRng[1] && loopRng[2] < nDCellIDs[2]) ) )
							|| ( (0 == colorId % 2) && ( (nDCellIDs[1] < loopRng[1] && nDCellIDs[2] <= loopRng[2])
								|| (nDCellIDs[0] == loopRng[0] && loopRng[1] < nDCellIDs[1] && nDCellIDs[2] <= loopRng[2])
								|| (nDCellIDs[0] != loopRng[0] && nDCellIDs[1] < loopRng[1] && loopRng[2] < nDCellIDs[2])
								|| (nDCellIDs[1] == loopRng[1] && nDCellIDs[2] < loopRng[2]) ) ) )
						{
							indexes[0] = loopRng[0];
							indexes[1] = loopRng[1];
							indexes[2] = loopRng[2];
							evaluateCellUnicompAdjacent(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index, indexLookupArr,
								point, cnt, pointIDKey, pointInDistVal, pointIdx, nDCellIDs, THREADPERPOINT, threadIdx.x % THREADPERPOINT);
						}
					}
		#endif
	#endif

}





// Global memory kernel - Linear ID comparison (Need to find a name : L-Unicomp ? Lin-Unicomp ? LId-Unicomp ?)
__global__ void kernelNDGridIndexGlobalLinearIDUnicomp(
		unsigned int *debug1,
		unsigned int *debug2,
		unsigned int *N,
		unsigned int * offset,
		unsigned int *batchNum,
		DTYPE* database,
		DTYPE* sortedCells,
		DTYPE* epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		struct gridCellLookup * gridCellLookupArr,
		DTYPE* minArr,
		unsigned int * nCells,
		unsigned int * cnt,
		unsigned int * nNonEmptyCells,
		unsigned int * gridCellNDMask,
		unsigned int * gridCellNDMaskOffsets,
		int * pointIDKey,
		int * pointInDistVal)
{

	unsigned int tid = (blockIdx.x * BLOCKSIZE + threadIdx.x) / THREADPERPOINT;

	if (*N <= tid){
		return;
	}

	//the point id in the dataset
	unsigned int pointIdx = tid * (*offset) + (*batchNum);
	//The offset into the database, taking into consideration the length of each dimension
	unsigned int pointOffset = tid * (GPUNUMDIM) * (*offset) + (*batchNum) * (GPUNUMDIM);

	//make a local copy of the point
	DTYPE point[GPUNUMDIM];
	#if SORT_BY_WORKLOAD
		for (int i = 0; i < GPUNUMDIM; i++){
			point[i] = sortedCells[pointOffset + i];
		}
	#else
		for (int i = 0; i < GPUNUMDIM; i++){
			point[i] = database[pointOffset + i];
		}
	#endif

	//calculate the coords of the Cell for the point
	//and the min/max ranges in each dimension
	unsigned int nDCellIDs[NUMINDEXEDDIM];

	unsigned int rangeFilteredCellIdsMin[NUMINDEXEDDIM];
	unsigned int rangeFilteredCellIdsMax[NUMINDEXEDDIM];

	for (int i = 0; i < NUMINDEXEDDIM; i++)
	{
		nDCellIDs[i] = (point[i] - minArr[i]) / (*epsilon);
		unsigned int nDMinCellIDs = max(0, nDCellIDs[i] - 1); //boundary conditions (don't go beyond cell 0)
		unsigned int nDMaxCellIDs = min(nCells[i] - 1, nDCellIDs[i] + 1); //boundary conditions (don't go beyond the maximum number of cells)

		bool foundMin = 0;
		bool foundMax = 0;

		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) ],
				gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) + 1 ] + 1, nDMinCellIDs)){ //extra +1 here is because we include the upper bound
			foundMin = 1;
		}
		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) ],
				gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) + 1 ] + 1, nDMaxCellIDs)){ //extra +1 here is because we include the upper bound
			foundMax = 1;
		}

		// cases:
		// found the min and max
		// found the min and not max
		//found the max and not the min
		//you don't find the min or max -- then only check the mid
		//you always find the mid because it's in the cell of the point you're looking for

		//NEED TO OPTIMIZE STILL
		if (1 == foundMin && 1 == foundMax){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs;
			rangeFilteredCellIdsMax[i] = nDMaxCellIDs;
			//printf("\nmin and max");
		}
		else if (1 == foundMin && 0 == foundMax){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs;
			rangeFilteredCellIdsMax[i] = nDMinCellIDs + 1;
			//printf("\nmin not max");
		}
		else if (0 == foundMin && 1 == foundMax){
			rangeFilteredCellIdsMin[i] =nDMinCellIDs + 1;
			rangeFilteredCellIdsMax[i] = nDMaxCellIDs;
			//printf("\nmax not min");
		}
		//dont find either min or max
		//get middle value only
		else{
			//printf("\nneither");
			rangeFilteredCellIdsMin[i] = nDMinCellIDs + 1;
			rangeFilteredCellIdsMax[i] = nDMinCellIDs + 1;
		}

	}

	///////////////////////////////////////
	//End taking intersection
	//////////////////////////////////////

	unsigned int indexes[NUMINDEXEDDIM];
	unsigned int loopRng[NUMINDEXEDDIM];

	uint64_t cellID = getLinearID_nDimensionsGPU(nDCellIDs, nCells, NUMINDEXEDDIM);
	for(int i = 0; i < NUMINDEXEDDIM; i++) {
		indexes[i] = nDCellIDs[i];
	}
	evaluateCellUnicompOrigin(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index, indexLookupArr,
		point, cnt, pointIDKey, pointInDistVal, pointIdx, nDCellIDs, THREADPERPOINT, threadIdx.x % THREADPERPOINT);

	for (loopRng[0] = rangeFilteredCellIdsMin[0]; loopRng[0] <= rangeFilteredCellIdsMax[0]; loopRng[0]++)
		for (loopRng[1] = rangeFilteredCellIdsMin[1]; loopRng[1] <= rangeFilteredCellIdsMax[1]; loopRng[1]++)
		#include "kernelloops.h"
		{ //beginning of loop body

			for (int x = 0; x < NUMINDEXEDDIM; x++){
				indexes[x] = loopRng[x];
			}

			uint64_t neighborID = getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);
			if(cellID < neighborID)
			{
				evaluateCellUnicompAdjacent(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index, indexLookupArr,
					point, cnt, pointIDKey, pointInDistVal, pointIdx, nDCellIDs, THREADPERPOINT, threadIdx.x % THREADPERPOINT);
			}

		} //end loop body

}





// Global memory kernel - Sorting cells by workload (Need to find a name)
__global__ void kernelNDGridIndexGlobalSortedCells(
		unsigned int *debug1,
		unsigned int *debug2,
		unsigned int *N,
		unsigned int * offset,
		unsigned int *batchNum,
		DTYPE* database,
		DTYPE* sortedCells,
		DTYPE* epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		struct gridCellLookup * gridCellLookupArr,
		DTYPE* minArr,
		unsigned int * nCells,
		unsigned int * cnt,
		unsigned int * nNonEmptyCells,
		unsigned int * gridCellNDMask,
		unsigned int * gridCellNDMaskOffsets,
		int * pointIDKey,
		int * pointInDistVal)
{

	unsigned int tid = (blockIdx.x * BLOCKSIZE + threadIdx.x) / THREADPERPOINT;

	if (*N <= tid){
		return;
	}

	//the point id in the dataset
	unsigned int pointIdx = tid * (*offset) + (*batchNum);
	//The offset into the database, taking into consideration the length of each dimension
	unsigned int pointOffset = tid * (GPUNUMDIM) * (*offset) + (*batchNum) * (GPUNUMDIM);

	//make a local copy of the point
	DTYPE point[GPUNUMDIM];
	for (int i = 0; i < GPUNUMDIM; i++){
		point[i] = sortedCells[pointOffset + i];
	}

	//calculate the coords of the Cell for the point
	//and the min/max ranges in each dimension
	unsigned int nDCellIDs[NUMINDEXEDDIM];

	unsigned int rangeFilteredCellIdsMin[NUMINDEXEDDIM];
	unsigned int rangeFilteredCellIdsMax[NUMINDEXEDDIM];

	for (int i = 0; i < NUMINDEXEDDIM; i++){
		nDCellIDs[i] = (point[i] - minArr[i]) / (*epsilon);
		unsigned int nDMinCellIDs = max(0, nDCellIDs[i] - 1); //boundary conditions (don't go beyond cell 0)
		unsigned int nDMaxCellIDs = min(nCells[i] - 1, nDCellIDs[i] + 1); //boundary conditions (don't go beyond the maximum number of cells)

		bool foundMin = 0;
		bool foundMax = 0;

		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) ],
			gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) + 1 ] + 1, nDMinCellIDs)){ //extra +1 here is because we include the upper bound
			foundMin = 1;
		}
		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) ],
			gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) + 1 ] + 1, nDMaxCellIDs)){ //extra +1 here is because we include the upper bound
			foundMax = 1;
		}

		if (foundMin == 1 && foundMax == 1){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs;
			rangeFilteredCellIdsMax[i] = nDMaxCellIDs;
			//printf("\nmin and max");
		}
		else if (foundMin == 1 && foundMax == 0){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs;
			rangeFilteredCellIdsMax[i] = nDMinCellIDs + 1;
			//printf("\nmin not max");
		}
		else if (foundMin == 0 && foundMax == 1){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs + 1;
			rangeFilteredCellIdsMax[i] = nDMaxCellIDs;
			//printf("\nmax not min");
		}
		//dont find either min or max
		//get middle value only
		else{
			//printf("\nneither");
			rangeFilteredCellIdsMin[i] = nDMinCellIDs + 1;
			rangeFilteredCellIdsMax[i] = nDMinCellIDs + 1;
		}

	}

	///////////////////////////////////////
	//End taking intersection
	//////////////////////////////////////

	unsigned int indexes[NUMINDEXEDDIM];
	unsigned int loopRng[NUMINDEXEDDIM];

	for (loopRng[0] = rangeFilteredCellIdsMin[0]; loopRng[0] <= rangeFilteredCellIdsMax[0]; loopRng[0]++)
		for (loopRng[1] = rangeFilteredCellIdsMin[1]; loopRng[1] <= rangeFilteredCellIdsMax[1]; loopRng[1]++)
		#include "kernelloops.h"
		{ //beginning of loop body

			for (int x = 0; x < NUMINDEXEDDIM; x++){
				indexes[x] = loopRng[x];
			}

			#if THREADPERPOINT > 1
				evaluateCellUnicompOrigin(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index, indexLookupArr,
					point, cnt, pointIDKey, pointInDistVal, pointIdx, nDCellIDs, THREADPERPOINT, threadIdx.x % THREADPERPOINT);
			#else
				evaluateCell(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index,
						indexLookupArr, point, cnt, pointIDKey, pointInDistVal, pointIdx, false, nDCellIDs);
			#endif

		} //end loop body

}





__global__ void kernelNDGridIndexGlobalSortedCellsDynamicThreads(
		unsigned int *debug1,
		unsigned int *debug2,
		unsigned int *N,
		unsigned int * offset,
		unsigned int *batchNum,
		DTYPE* database,
		unsigned int * sortedCells,
		unsigned int * sortedCellsNbThreads,
		unsigned int * sortedCellsNbThreadsBefore,
		unsigned int nbTotalThreads,
		DTYPE* epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		struct gridCellLookup * gridCellLookupArr,
		DTYPE* minArr,
		unsigned int * nCells,
		unsigned int * cnt,
		unsigned int * nNonEmptyCells,
		unsigned int * gridCellNDMask,
		unsigned int * gridCellNDMaskOffsets,
		int * pointIDKey,
		int * pointInDistVal)
{

	unsigned int tid = (blockIdx.x * BLOCKSIZE + threadIdx.x);

	/*if (nbTotalThreads <= tid){
		return;
	}*/

	//the point id in the dataset
	unsigned int pointIdx = tid * (*offset) + (*batchNum);
	//The offset into the database, taking into consideration the length of each dimension
	unsigned int pointOffset = tid * (GPUNUMDIM) * (*offset) + (*batchNum) * (GPUNUMDIM);

	if(nbTotalThreads <= pointOffset){
		return;
	}

	//make a local copy of the point
	DTYPE point[GPUNUMDIM];
	unsigned int dataIdx = indexLookupArr[ sortedCells[pointOffset] ];
	for (int i = 0; i < GPUNUMDIM; i++){
		point[i] = database[ dataIdx * GPUNUMDIM + i ];
	}

	//calculate the coords of the Cell for the point
	//and the min/max ranges in each dimension
	unsigned int nDCellIDs[NUMINDEXEDDIM];

	unsigned int rangeFilteredCellIdsMin[NUMINDEXEDDIM];
	unsigned int rangeFilteredCellIdsMax[NUMINDEXEDDIM];

	for (int i = 0; i < NUMINDEXEDDIM; i++)
	{
		nDCellIDs[i] = (point[i] - minArr[i]) / (*epsilon);
		unsigned int nDMinCellIDs = max(0, nDCellIDs[i] - 1); //boundary conditions (don't go beyond cell 0)
		unsigned int nDMaxCellIDs = min(nCells[i] - 1, nDCellIDs[i] + 1); //boundary conditions (don't go beyond the maximum number of cells)

		bool foundMin = 0;
		bool foundMax = 0;

		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) ],
				gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) + 1 ] + 1, nDMinCellIDs)){ //extra +1 here is because we include the upper bound
			foundMin = 1;
		}
		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) ],
				gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) + 1 ] + 1, nDMaxCellIDs)){ //extra +1 here is because we include the upper bound
			foundMax = 1;
		}

		// cases:
		// found the min and max
		// found the min and not max
		//found the max and not the min
		//you don't find the min or max -- then only check the mid
		//you always find the mid because it's in the cell of the point you're looking for

		//NEED TO OPTIMIZE STILL
		if (1 == foundMin && 1 == foundMax){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs;
			rangeFilteredCellIdsMax[i] = nDMaxCellIDs;
			//printf("\nmin and max");
		}
		else if (1 == foundMin && 0 == foundMax){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs;
			rangeFilteredCellIdsMax[i] = nDMinCellIDs + 1;
			//printf("\nmin not max");
		}
		else if (0 == foundMin && 1 == foundMax){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs + 1;
			rangeFilteredCellIdsMax[i] = nDMaxCellIDs;
			//printf("\nmax not min");
		}
		//dont find either min or max
		//get middle value only
		else{
			//printf("\nneither");
			rangeFilteredCellIdsMin[i] = nDMinCellIDs + 1;
			rangeFilteredCellIdsMax[i] = nDMinCellIDs + 1;
		}

	}

	///////////////////////////////////////
	//End taking intersection
	//////////////////////////////////////

	unsigned int indexes[NUMINDEXEDDIM];
	unsigned int loopRng[NUMINDEXEDDIM];

	uint64_t cellID = getLinearID_nDimensionsGPU(nDCellIDs, nCells, NUMINDEXEDDIM);
	for(int i = 0; i < NUMINDEXEDDIM; i++) {
		indexes[i] = nDCellIDs[i];
	}
	evaluateCellUnicompOrigin(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index, indexLookupArr,
		point, cnt, pointIDKey, pointInDistVal, pointIdx, nDCellIDs, sortedCellsNbThreads[pointOffset], sortedCellsNbThreadsBefore[pointOffset]);


	for (loopRng[0] = rangeFilteredCellIdsMin[0]; loopRng[0] <= rangeFilteredCellIdsMax[0]; loopRng[0]++)
		for (loopRng[1] = rangeFilteredCellIdsMin[1]; loopRng[1] <= rangeFilteredCellIdsMax[1]; loopRng[1]++)
		#include "kernelloops.h"
		{ //beginning of loop body

			for (int x = 0; x < NUMINDEXEDDIM; x++){
				indexes[x] = loopRng[x];
			}

			uint64_t neighborID = getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);
			if(cellID < neighborID)
			{
				evaluateCellUnicompAdjacent(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index, indexLookupArr,
					point, cnt, pointIDKey, pointInDistVal, pointIdx, nDCellIDs, sortedCellsNbThreads[pointOffset], sortedCellsNbThreadsBefore[pointOffset]);
			}

		} //end loop body

}





__global__ void kernelNDGridIndexGlobalSortedCellsDynamicThreadsV2(
		unsigned int *debug1,
		unsigned int *debug2,
		unsigned int *N,
		unsigned int * offset,
		unsigned int *batchNum,
		DTYPE* database,
		DTYPE * sortedSet,
		unsigned int * sortedCellsNbThreads,
		unsigned int * sortedCellsNbThreadsBefore,
		unsigned int nbTotalThreads,
		DTYPE* epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		struct gridCellLookup * gridCellLookupArr,
		DTYPE* minArr,
		unsigned int * nCells,
		unsigned int * cnt,
		unsigned int * nNonEmptyCells,
		unsigned int * gridCellNDMask,
		unsigned int * gridCellNDMaskOffsets,
		int * pointIDKey,
		int * pointInDistVal)
{

	unsigned int tid = (blockIdx.x * BLOCKSIZE + threadIdx.x);

	/*if (nbTotalThreads <= tid){
		return;
	}*/

	//the point id in the dataset
	unsigned int pointIdx = tid * (*offset) + (*batchNum);
	//The offset into the database, taking into consideration the length of each dimension
	unsigned int pointOffset = tid * (GPUNUMDIM) * (*offset) + (*batchNum) * (GPUNUMDIM);

	if(nbTotalThreads <= pointIdx){
		return;
	}

	//make a local copy of the point
	DTYPE point[GPUNUMDIM];
	for (int i = 0; i < GPUNUMDIM; i++){
		point[i] = sortedSet[pointOffset + i];
	}

	//calculate the coords of the Cell for the point
	//and the min/max ranges in each dimension
	unsigned int nDCellIDs[NUMINDEXEDDIM];

	unsigned int rangeFilteredCellIdsMin[NUMINDEXEDDIM];
	unsigned int rangeFilteredCellIdsMax[NUMINDEXEDDIM];

	for (int i = 0; i < NUMINDEXEDDIM; i++)
	{
		nDCellIDs[i] = (point[i] - minArr[i]) / (*epsilon);
		unsigned int nDMinCellIDs = max(0, nDCellIDs[i] - 1); //boundary conditions (don't go beyond cell 0)
		unsigned int nDMaxCellIDs = min(nCells[i] - 1, nDCellIDs[i] + 1); //boundary conditions (don't go beyond the maximum number of cells)

		bool foundMin = 0;
		bool foundMax = 0;

		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) ],
				gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) + 1 ] + 1, nDMinCellIDs)){ //extra +1 here is because we include the upper bound
			foundMin = 1;
		}
		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) ],
				gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) + 1 ] + 1, nDMaxCellIDs)){ //extra +1 here is because we include the upper bound
			foundMax = 1;
		}

		// cases:
		// found the min and max
		// found the min and not max
		//found the max and not the min
		//you don't find the min or max -- then only check the mid
		//you always find the mid because it's in the cell of the point you're looking for

		//NEED TO OPTIMIZE STILL
		if (1 == foundMin && 1 == foundMax){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs;
			rangeFilteredCellIdsMax[i] = nDMaxCellIDs;
			//printf("\nmin and max");
		}
		else if (1 == foundMin && 0 == foundMax){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs;
			rangeFilteredCellIdsMax[i] = nDMinCellIDs + 1;
			//printf("\nmin not max");
		}
		else if (0 == foundMin && 1 == foundMax){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs + 1;
			rangeFilteredCellIdsMax[i] = nDMaxCellIDs;
			//printf("\nmax not min");
		}
		//dont find either min or max
		//get middle value only
		else{
			//printf("\nneither");
			rangeFilteredCellIdsMin[i] = nDMinCellIDs + 1;
			rangeFilteredCellIdsMax[i] = nDMinCellIDs + 1;
		}

	}

	///////////////////////////////////////
	//End taking intersection
	//////////////////////////////////////

	unsigned int indexes[NUMINDEXEDDIM];
	unsigned int loopRng[NUMINDEXEDDIM];

	uint64_t cellID = getLinearID_nDimensionsGPU(nDCellIDs, nCells, NUMINDEXEDDIM);

	for(int i = 0; i < NUMINDEXEDDIM; i++) {
		indexes[i] = nDCellIDs[i];
	}
	evaluateCellUnicompOrigin(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index, indexLookupArr,
		point, cnt, pointIDKey, pointInDistVal, pointIdx, nDCellIDs, sortedCellsNbThreads[pointIdx], sortedCellsNbThreadsBefore[pointIdx]);

	for (loopRng[0] = rangeFilteredCellIdsMin[0]; loopRng[0] <= rangeFilteredCellIdsMax[0]; loopRng[0]++)
		for (loopRng[1] = rangeFilteredCellIdsMin[1]; loopRng[1] <= rangeFilteredCellIdsMax[1]; loopRng[1]++)
		#include "kernelloops.h"
		{ //beginning of loop body

			for (int x = 0; x < NUMINDEXEDDIM; x++){
				indexes[x] = loopRng[x];
			}

			uint64_t neighborID = getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);
			if(cellID < neighborID)
			{
				evaluateCellUnicompAdjacent(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index, indexLookupArr,
					point, cnt, pointIDKey, pointInDistVal, pointIdx, nDCellIDs, sortedCellsNbThreads[pointIdx], sortedCellsNbThreadsBefore[pointIdx]);
			}

		} //end loop body

}





__global__ void kernelNDGridIndexGlobalSortedCellsDynamicThreadsFixed(
	unsigned int *debug1,
	unsigned int *debug2,
	unsigned int *N,
	unsigned int *offset,
	unsigned int *batchNum,
	DTYPE * database,
	DTYPE * sortedCells,
	unsigned int * threadArray,
	unsigned int nbTotalThreads,
	DTYPE* epsilon,
	struct grid * index,
	unsigned int * indexLookupArr,
	struct gridCellLookup * gridCellLookupArr,
	DTYPE* minArr,
	unsigned int * nCells,
	unsigned int * cnt,
	unsigned int * nNonEmptyCells,
	unsigned int * gridCellNDMask,
	unsigned int * gridCellNDMaskOffsets,
	int * pointIDKey,
	int * pointInDistVal)
{

	unsigned int tid = (blockIdx.x * BLOCKSIZE + threadIdx.x);
	unsigned int tPerPoint;

	//unsigned int pointToWork = binary_search(threadArray, 0, (*N), globalId);
	unsigned int pointToWork = threadArray[tid];

	//make a local copy of the point
	DTYPE point[GPUNUMDIM];
	for (int i = 0; i < GPUNUMDIM; i++){
		//point[i] = sortedCells[pointOffset + i];
		point[i] = sortedCells[pointToWork * GPUNUMDIM + i];
	}

	//calculate the coords of the Cell for the point
	//and the min/max ranges in each dimension
	unsigned int nDCellIDs[NUMINDEXEDDIM];

	unsigned int rangeFilteredCellIdsMin[NUMINDEXEDDIM];
	unsigned int rangeFilteredCellIdsMax[NUMINDEXEDDIM];

	for (int i = 0; i < NUMINDEXEDDIM; i++)
	{
		nDCellIDs[i] = (point[i] - minArr[i]) / (*epsilon);
		unsigned int nDMinCellIDs = max(0, nDCellIDs[i] - 1); //boundary conditions (don't go beyond cell 0)
		unsigned int nDMaxCellIDs = min(nCells[i] - 1, nDCellIDs[i] + 1); //boundary conditions (don't go beyond the maximum number of cells)

		bool foundMin = 0;
		bool foundMax = 0;

		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) ],
			gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) + 1 ] + 1, nDMinCellIDs)){ //extra +1 here is because we include the upper bound
			foundMin = 1;
		}
		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) ],
			gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) + 1 ] + 1, nDMaxCellIDs)){ //extra +1 here is because we include the upper bound
			foundMax = 1;
		}

		if (foundMin == 1 && foundMax == 1){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs;
			rangeFilteredCellIdsMax[i] = nDMaxCellIDs;
			//printf("\nmin and max");
		}
		else if (foundMin == 1 && foundMax == 0){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs;
			rangeFilteredCellIdsMax[i] = nDMinCellIDs + 1;
			//printf("\nmin not max");
		}
		else if (foundMin == 0 && foundMax == 1){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs + 1;
			rangeFilteredCellIdsMax[i] = nDMaxCellIDs;
			//printf("\nmax not min");
		}
		//dont find either min or max
		//get middle value only
		else{
			//printf("\nneither");
			rangeFilteredCellIdsMin[i] = nDMinCellIDs + 1;
			rangeFilteredCellIdsMax[i] = nDMinCellIDs + 1;
		}

	}

	///////////////////////////////////////
	//End taking intersection
	//////////////////////////////////////

	unsigned int indexes[NUMINDEXEDDIM];
	unsigned int loopRng[NUMINDEXEDDIM];

	for (loopRng[0] = rangeFilteredCellIdsMin[0]; loopRng[0] <= rangeFilteredCellIdsMax[0]; loopRng[0]++)
		for (loopRng[1] = rangeFilteredCellIdsMin[1]; loopRng[1] <= rangeFilteredCellIdsMax[1]; loopRng[1]++)
		#include "kernelloops.h"
		{ //beginning of loop body

			for (int x = 0; x < NUMINDEXEDDIM; x++){
				indexes[x] = loopRng[x];
			}

			if(1 < tPerPoint)
			{
				evaluateCellUnicompOrigin(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index, indexLookupArr,
						point, cnt, pointIDKey, pointInDistVal, pointToWork, nDCellIDs, tPerPoint, threadIdx.x % tPerPoint);
			}else{
				evaluateCell(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index,
						indexLookupArr, point, cnt, pointIDKey, pointInDistVal, pointToWork, false, nDCellIDs);
			}

		} //end loop body

}





__global__ void kernelNDGridIndexGlobalSortedCellsDynamicThreadsV3(
		unsigned int *debug1,
		unsigned int *debug2,
		unsigned int *N,
		unsigned int * offset,
		unsigned int *batchNum,
		DTYPE* database,
		DTYPE* sortedCells,
		DTYPE* epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		struct gridCellLookup * gridCellLookupArr,
		DTYPE* minArr,
		unsigned int * nCells,
		unsigned int * cnt,
		unsigned int * nNonEmptyCells,
		unsigned int * gridCellNDMask,
		unsigned int * gridCellNDMaskOffsets,
		int * pointIDKey,
		int * pointInDistVal,
		int nbThreadsPerPoint)
{

	unsigned int tid = (blockIdx.x * BLOCKSIZE + threadIdx.x) / nbThreadsPerPoint;

	if (*N <= tid){
		return;
	}

	//the point id in the dataset
	unsigned int pointIdx = tid * (*offset) + (*batchNum);
	//The offset into the database, taking into consideration the length of each dimension
	unsigned int pointOffset = tid * (GPUNUMDIM) * (*offset) + (*batchNum) * (GPUNUMDIM);

	//make a local copy of the point
	DTYPE point[GPUNUMDIM];
	for (int i = 0; i < GPUNUMDIM; i++){
		point[i] = sortedCells[pointOffset + i];
	}

	//calculate the coords of the Cell for the point
	//and the min/max ranges in each dimension
	unsigned int nDCellIDs[NUMINDEXEDDIM];

	unsigned int rangeFilteredCellIdsMin[NUMINDEXEDDIM];
	unsigned int rangeFilteredCellIdsMax[NUMINDEXEDDIM];

	for (int i = 0; i < NUMINDEXEDDIM; i++){
		nDCellIDs[i] = (point[i] - minArr[i]) / (*epsilon);
		unsigned int nDMinCellIDs = max(0, nDCellIDs[i] - 1); //boundary conditions (don't go beyond cell 0)
		unsigned int nDMaxCellIDs = min(nCells[i] - 1, nDCellIDs[i] + 1); //boundary conditions (don't go beyond the maximum number of cells)

		bool foundMin = 0;
		bool foundMax = 0;

		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) ],
			gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) + 1 ] + 1, nDMinCellIDs)){ //extra +1 here is because we include the upper bound
			foundMin = 1;
		}
		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) ],
			gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) + 1 ] + 1, nDMaxCellIDs)){ //extra +1 here is because we include the upper bound
			foundMax = 1;
		}

		if (foundMin == 1 && foundMax == 1){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs;
			rangeFilteredCellIdsMax[i] = nDMaxCellIDs;
			//printf("\nmin and max");
		}
		else if (foundMin == 1 && foundMax == 0){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs;
			rangeFilteredCellIdsMax[i] = nDMinCellIDs + 1;
			//printf("\nmin not max");
		}
		else if (foundMin == 0 && foundMax == 1){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs + 1;
			rangeFilteredCellIdsMax[i] = nDMaxCellIDs;
			//printf("\nmax not min");
		}
		//dont find either min or max
		//get middle value only
		else{
			//printf("\nneither");
			rangeFilteredCellIdsMin[i] = nDMinCellIDs + 1;
			rangeFilteredCellIdsMax[i] = nDMinCellIDs + 1;
		}

	}

	///////////////////////////////////////
	//End taking intersection
	//////////////////////////////////////

	unsigned int indexes[NUMINDEXEDDIM];
	unsigned int loopRng[NUMINDEXEDDIM];

	for (loopRng[0] = rangeFilteredCellIdsMin[0]; loopRng[0] <= rangeFilteredCellIdsMax[0]; loopRng[0]++)
		for (loopRng[1] = rangeFilteredCellIdsMin[1]; loopRng[1] <= rangeFilteredCellIdsMax[1]; loopRng[1]++)
		#include "kernelloops.h"
		{ //beginning of loop body

			for (int x = 0; x < NUMINDEXEDDIM; x++){
				indexes[x] = loopRng[x];
			}

			if(1 < nbThreadsPerPoint)
			{
				evaluateCellUnicompOrigin(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index, indexLookupArr,
					point, cnt, pointIDKey, pointInDistVal, pointIdx, nDCellIDs, nbThreadsPerPoint, threadIdx.x % nbThreadsPerPoint);
			}else{
				evaluateCell(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index,
						indexLookupArr, point, cnt, pointIDKey, pointInDistVal, pointIdx, false, nDCellIDs);
			}

		} //end loop body

}





__device__ int atomicAggInc(int *ctr) {
	auto g = coalesced_threads();
	int warp_res;
	if(g.thread_rank() == 0)
		warp_res = atomicAdd(ctr, g.size());
	return g.shfl(warp_res, 0) + g.thread_rank();
}

__device__ int counter = 0;

__global__ void kernelNDGridIndexGlobalWorkQueue(
		unsigned int * debug1,
		unsigned int * debug2,
		unsigned int * N,
		unsigned int * offset,
		unsigned int * batchNum,
		DTYPE * database,
		DTYPE * sortedCells,
		unsigned int * originPointIndex,
		DTYPE * epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		struct gridCellLookup * gridCellLookupArr,
		DTYPE* minArr,
		unsigned int * nCells,
		unsigned int * cnt,
		unsigned int * nNonEmptyCells,
		unsigned int * gridCellNDMask,
		unsigned int * gridCellNDMaskOffsets,
		int * pointIDKey,
		int * pointInDistVal,
		unsigned int * elementToWork,
		unsigned int nbPoints)
{

	unsigned int tid = (blockIdx.x * BLOCKSIZE + threadIdx.x) / THREADPERPOINT;

	if (*N <= tid){
		return;
	}

	#if THREADPERPOINT == 1
		unsigned int pointId = atomicAdd(&counter, int(1));
	#else
		// __shared__ int pointIdShared;
		// if(0 == threadIdx.x)
		// {
			// pointIdShared = atomicAdd(&counter, int(BLOCKSIZE / THREADPERPOINT));
		// }
		// __syncthreads();
		// unsigned int pointId = pointIdShared + (threadIdx.x / THREADPERPOINT);


		// thread_block_tile<32> tile32 = tiled_partition<32>(this_thread_block());
		// unsigned int pointId;
		// if(0 == tile32.thread_rank())
		// {
			// pointId = atomicAdd(&counter, int(32 / THREADPERPOINT));
		// }
		// pointId = tile32.shfl(pointId, 0) + (tile32.thread_rank() / THREADPERPOINT);


		// coalesced_group active = coalesced_threads();
		// unsigned int pointId;
		// if(0 == active.thread_rank())
		// {
			// pointId = atomicAdd(&counter, int(active.size() / THREADPERPOINT));
		// }
		// pointId = active.shfl(pointId, 0) + (active.thread_rank() / THREADPERPOINT);


		// thread_block_tile<THREADPERPOINT> tile = tiled_partition<THREADPERPOINT>(coalesced_threads());
		auto tile = tiled_partition(coalesced_threads(), THREADPERPOINT);
		unsigned int pointId;
		if(0 == tile.thread_rank())
		{
			pointId = atomicAdd(&counter, int(1));
		}
		pointId = tile.shfl(pointId, 0);
	#endif

	//make a local copy of the point
	DTYPE point[GPUNUMDIM];
	for (int i = 0; i < GPUNUMDIM; i++){
		//point[i] = sortedCells[pointOffset + i];
		point[i] = sortedCells[pointId * GPUNUMDIM + i];
		//point[i] = database[pointId * GPUNUMDIM + i];
	}

	//calculate the coords of the Cell for the point
	//and the min/max ranges in each dimension
	unsigned int nDCellIDs[NUMINDEXEDDIM];
	unsigned int rangeFilteredCellIdsMin[NUMINDEXEDDIM];
	unsigned int rangeFilteredCellIdsMax[NUMINDEXEDDIM];

	for (int i = 0; i < NUMINDEXEDDIM; i++){
		nDCellIDs[i] = (point[i] - minArr[i]) / (*epsilon);
		unsigned int nDMinCellIDs = max(0, nDCellIDs[i] - 1); //boundary conditions (don't go beyond cell 0)
		unsigned int nDMaxCellIDs = min(nCells[i] - 1, nDCellIDs[i] + 1); //boundary conditions (don't go beyond the maximum number of cells)

		bool foundMin = 0;
		bool foundMax = 0;

		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) ],
			gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) + 1 ] + 1, nDMinCellIDs)){ //extra +1 here is because we include the upper bound
			foundMin = 1;
		}
		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) ],
			gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) + 1 ] + 1, nDMaxCellIDs)){ //extra +1 here is because we include the upper bound
			foundMax = 1;
		}

		if (foundMin == 1 && foundMax == 1){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs;
			rangeFilteredCellIdsMax[i] = nDMaxCellIDs;
			//printf("\nmin and max");
		}
		else if (foundMin == 1 && foundMax == 0){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs;
			rangeFilteredCellIdsMax[i] = nDMinCellIDs + 1;
			//printf("\nmin not max");
		}
		else if (foundMin == 0 && foundMax == 1){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs + 1;
			rangeFilteredCellIdsMax[i] = nDMaxCellIDs;
			//printf("\nmax not min");
		}
		//dont find either min or max
		//get middle value only
		else{
			//printf("\nneither");
			rangeFilteredCellIdsMin[i] = nDMinCellIDs + 1;
			rangeFilteredCellIdsMax[i] = nDMinCellIDs + 1;
		}

	}

	///////////////////////////////////////
	//End taking intersection
	//////////////////////////////////////

	unsigned int indexes[NUMINDEXEDDIM];
	unsigned int loopRng[NUMINDEXEDDIM];

	for (loopRng[0] = rangeFilteredCellIdsMin[0]; loopRng[0] <= rangeFilteredCellIdsMax[0]; loopRng[0]++)
		for (loopRng[1] = rangeFilteredCellIdsMin[1]; loopRng[1] <= rangeFilteredCellIdsMax[1]; loopRng[1]++)
		#include "kernelloops.h"
		{ //beginning of loop body

			for (int x = 0; x < NUMINDEXEDDIM; x++){
				indexes[x] = loopRng[x];
			}

			#if THREADPERPOINT > 1
				evaluateCellUnicompOrigin(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index, indexLookupArr,
					point, cnt, pointIDKey, pointInDistVal, originPointIndex[pointId], nDCellIDs, THREADPERPOINT, tile.thread_rank());
			#else
				evaluateCell(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index,
						indexLookupArr, point, cnt, pointIDKey, pointInDistVal, originPointIndex[pointId], false, nDCellIDs);
			#endif

		} //end loop body

}





__global__ void kernelNDGridIndexGlobalWorkQueueLidUnicomp(
		unsigned int * debug1,
		unsigned int * debug2,
		unsigned int * N,
		unsigned int * offset,
		unsigned int * batchNum,
		DTYPE * database,
		DTYPE * sortedCells,
		DTYPE * epsilon,
		struct grid * index,
		unsigned int * indexLookupArr,
		struct gridCellLookup * gridCellLookupArr,
		DTYPE* minArr,
		unsigned int * nCells,
		unsigned int * cnt,
		unsigned int * nNonEmptyCells,
		unsigned int * gridCellNDMask,
		unsigned int * gridCellNDMaskOffsets,
		int * pointIDKey,
		int * pointInDistVal,
		unsigned int * elementToWork,
		unsigned int nbPoints)
{

	unsigned int tid = (blockIdx.x * BLOCKSIZE + threadIdx.x) / THREADPERPOINT;

	if (*N <= tid){
		return;
	}

	#if THREADPERPOINT == 1
		unsigned int pointId = atomicAdd(&counter, int(1));
	#else
		// __shared__ int pointIdShared;
		// if(0 == threadIdx.x)
		// {
			// pointIdShared = atomicAdd(&counter, int(BLOCKSIZE / THREADPERPOINT));
		// }
		// __syncthreads();
		// unsigned int pointId = pointIdShared + (threadIdx.x / THREADPERPOINT);


		// thread_block_tile<32> tile32 = tiled_partition<32>(this_thread_block());
		// unsigned int pointId;
		// if(0 == tile32.thread_rank())
		// {
			// pointId = atomicAdd(&counter, int(32 / THREADPERPOINT));
		// }
		// pointId = tile32.shfl(pointId, 0) + (tile32.thread_rank() / THREADPERPOINT);


		// coalesced_group active = coalesced_threads();
		// unsigned int pointId;
		// if(0 == active.thread_rank())
		// {
			// pointId = atomicAdd(&counter, int(active.size() / THREADPERPOINT));
		// }
		// pointId = active.shfl(pointId, 0) + (active.thread_rank() / THREADPERPOINT);


		// thread_block_tile<THREADPERPOINT> tile = tiled_partition<THREADPERPOINT>(coalesced_threads());
		auto tile = tiled_partition(coalesced_threads(), THREADPERPOINT);
		unsigned int pointId;
		if(0 == tile.thread_rank())
		{
			pointId = atomicAdd(&counter, int(1));
		}
		pointId = tile.shfl(pointId, 0);
	#endif

	//make a local copy of the point
	DTYPE point[GPUNUMDIM];
	for (int i = 0; i < GPUNUMDIM; i++){
		//point[i] = sortedCells[pointOffset + i];
		point[i] = sortedCells[pointId * GPUNUMDIM + i];
		//point[i] = database[pointId * GPUNUMDIM + i];
	}

	//calculate the coords of the Cell for the point
	//and the min/max ranges in each dimension
	unsigned int nDCellIDs[NUMINDEXEDDIM];
	unsigned int rangeFilteredCellIdsMin[NUMINDEXEDDIM];
	unsigned int rangeFilteredCellIdsMax[NUMINDEXEDDIM];

	for (int i = 0; i < NUMINDEXEDDIM; i++){
		nDCellIDs[i] = (point[i] - minArr[i]) / (*epsilon);
		unsigned int nDMinCellIDs = max(0, nDCellIDs[i] - 1); //boundary conditions (don't go beyond cell 0)
		unsigned int nDMaxCellIDs = min(nCells[i] - 1, nDCellIDs[i] + 1); //boundary conditions (don't go beyond the maximum number of cells)

		bool foundMin = 0;
		bool foundMax = 0;

		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) ],
			gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) + 1 ] + 1, nDMinCellIDs)){ //extra +1 here is because we include the upper bound
			foundMin = 1;
		}
		if(thrust::binary_search(thrust::seq, gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) ],
			gridCellNDMask + gridCellNDMaskOffsets[ (i * 2) + 1 ] + 1, nDMaxCellIDs)){ //extra +1 here is because we include the upper bound
			foundMax = 1;
		}

		if (foundMin == 1 && foundMax == 1){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs;
			rangeFilteredCellIdsMax[i] = nDMaxCellIDs;
			//printf("\nmin and max");
		}
		else if (foundMin == 1 && foundMax == 0){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs;
			rangeFilteredCellIdsMax[i] = nDMinCellIDs + 1;
			//printf("\nmin not max");
		}
		else if (foundMin == 0 && foundMax == 1){
			rangeFilteredCellIdsMin[i] = nDMinCellIDs + 1;
			rangeFilteredCellIdsMax[i] = nDMaxCellIDs;
			//printf("\nmax not min");
		}
		//dont find either min or max
		//get middle value only
		else{
			//printf("\nneither");
			rangeFilteredCellIdsMin[i] = nDMinCellIDs + 1;
			rangeFilteredCellIdsMax[i] = nDMinCellIDs + 1;
		}

	}

	///////////////////////////////////////
	//End taking intersection
	//////////////////////////////////////

	unsigned int indexes[NUMINDEXEDDIM];
	unsigned int loopRng[NUMINDEXEDDIM];

	uint64_t cellID = getLinearID_nDimensionsGPU(nDCellIDs, nCells, NUMINDEXEDDIM);
	for(int i = 0; i < NUMINDEXEDDIM; i++) {
		indexes[i] = nDCellIDs[i];
	}
	#if THREADPERPOINT > 1
		evaluateCellUnicompOrigin(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index, indexLookupArr,
				point, cnt, pointIDKey, pointInDistVal, pointId, nDCellIDs, THREADPERPOINT, tile.thread_rank());
	#else
		evaluateCellUnicompOrigin(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index, indexLookupArr,
				point, cnt, pointIDKey, pointInDistVal, pointId, nDCellIDs, 1, 0);
	#endif

	for (loopRng[0] = rangeFilteredCellIdsMin[0]; loopRng[0] <= rangeFilteredCellIdsMax[0]; loopRng[0]++)
		for (loopRng[1] = rangeFilteredCellIdsMin[1]; loopRng[1] <= rangeFilteredCellIdsMax[1]; loopRng[1]++)
		#include "kernelloops.h"
		{ //beginning of loop body

			for (int x = 0; x < NUMINDEXEDDIM; x++){
				indexes[x] = loopRng[x];
			}

			uint64_t neighborID = getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);
			if(cellID < neighborID)
			{
				#if THREADPERPOINT > 1
					evaluateCellUnicompAdjacent(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index, indexLookupArr,
							point, cnt, pointIDKey, pointInDistVal, pointId, nDCellIDs, THREADPERPOINT, tile.thread_rank());
				#else
					evaluateCellUnicompAdjacent(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index, indexLookupArr,
							point, cnt, pointIDKey, pointInDistVal, pointId, nDCellIDs, 1, 0);
				#endif
			}

		} //end loop body

}
