#include "structs.h"
#include "params.h"




void schedulePointsByWorkDynamicThreadsPerPoint(
        int searchMode,
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
        std::vector<unsigned int> * sortedSetDynamicThreads,
        std::vector<unsigned int> * sortedCellsNbThreads,
        std::vector<unsigned int> * sortedCellsNbThreadsBefore);

void schedulePointsByWork(
        int searchMode,
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
        DTYPE* sortedSet);

void distanceTableNDGridBatches(int searchMode, std::vector<std::vector<DTYPE> > * NDdataPoints, DTYPE* epsilon, struct grid * index,
	struct gridCellLookup * gridCellLookupArr, unsigned int * nNonEmptyCells, DTYPE* minArr, unsigned int * nCells,
	unsigned int * indexLookupArr, struct neighborTableLookup * neighborTable, std::vector<struct neighborDataPtrs> * pointersToNeighbors,
	uint64_t * totalNeighbors, unsigned int * gridCellNDMask, unsigned int * gridCellNDMaskOffsets, unsigned int * nNDMaskElems);

unsigned long long callGPUBatchEst(unsigned int * DBSIZE, DTYPE* dev_database, DTYPE* dev_epsilon, struct grid * dev_grid,
	unsigned int * dev_indexLookupArr, struct gridCellLookup * dev_gridCellLookupArr, DTYPE* dev_minArr,
	unsigned int * dev_nCells, unsigned int * dev_nNonEmptyCells, unsigned int * dev_gridCellNDMask,
	unsigned int * dev_gridCellNDMaskOffsets, unsigned int * dev_nNDMaskElems, unsigned int * retNumBatches, unsigned int * retGPUBufferSize);

unsigned long long callGPUBatchEstWorkQueue(unsigned int * DBSIZE, DTYPE* dev_database, DTYPE* dev_sortedCells, DTYPE* dev_epsilon,
    struct grid * dev_grid, unsigned int * dev_indexLookupArr, struct gridCellLookup * dev_gridCellLookupArr, DTYPE* dev_minArr,
	unsigned int * dev_nCells, unsigned int * dev_nNonEmptyCells, unsigned int * dev_gridCellNDMask,
	unsigned int * dev_gridCellNDMaskOffsets, unsigned int * dev_nNDMaskElems, unsigned int * retNumBatches, unsigned int * retGPUBufferSize);

void constructNeighborTableKeyValueWithPtrs(int * pointIDKey, int * pointInDistValue, struct neighborTableLookup * neighborTable, int * pointersToNeighbors, unsigned int * cnt);

void warmUpGPU();
