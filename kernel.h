#include "structs.h"
#include "params.h"


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
		DTYPE* sortedSet);

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
		DTYPE* sortedSet);


__device__ uint64_t getLinearID_nDimensionsGPU(
		unsigned int * indexes,
		unsigned int * dimLen,
		unsigned int nDimensions);


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
		bool differentCell);


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
		unsigned int* nDCellIDs);


__forceinline__ __device__ void evalPointUnicompOrigin(
		unsigned int* indexLookupArr,
		int k,
		DTYPE* database,
		DTYPE* epsilon,
		DTYPE* point,
		unsigned int* cnt,
		int* pointIDKey,
		int* pointInDistVal,
		int pointIdx);


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
		unsigned int numThread);


__forceinline__ __device__ void evalPointUnicompAdjacent(
		unsigned int* indexLookupArr,
		int k,
		DTYPE* database,
		DTYPE* epsilon,
		DTYPE* point,
		unsigned int* cnt,
		int* pointIDKey,
		int* pointInDistVal,
		int pointIdx);


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
		unsigned int numThread);


/*############################################################################*/


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
		unsigned int * gridCellNDMaskOffsets);


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
		unsigned int * gridCellNDMaskOffsets);


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
		int * pointInDistVal);


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
		int * pointInDistVal);


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
		int * pointInDistVal);


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
		int * pointInDistVal);


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
		int * pointInDistVal);


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
		int * pointInDistVal);


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
		int * pointInDistVal);


__global__ void kernelNDGridIndexGlobalSortedCellsDynamicThreadsFixed(
		unsigned int * debug1,
		unsigned int * debug2,
		unsigned int * N,
		unsigned int * offset,
		unsigned int * batchNum,
		DTYPE* database,
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
		int * pointInDistVal);


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
		int nbThreadsPerPoint);


__global__ void kernelNDGridIndexGlobalWorkQueue(
		unsigned int *debug1,
		unsigned int *debug2,
		unsigned int *N,
		unsigned int * offset,
		unsigned int *batchNum,
		DTYPE* database,
		DTYPE* sortedCells,
		unsigned int * originPointIndex,
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
		unsigned int * elementToWork,
		unsigned int nbPoints);


__global__ void kernelNDGridIndexGlobalWorkQueueLidUnicomp(
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
		unsigned int * elementToWork,
		unsigned int nbPoints);
