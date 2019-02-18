

//precompute direct neighbors with the GPU:
#include <cuda_runtime.h>
#include <cuda.h>
#include "structs.h"
#include <stdio.h>
#include "kernel.h"
#include <math.h>
#include "GPU.h"
#include <algorithm>
#include "omp.h"
#include <queue>
#include <unistd.h>

//thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h> //for streams for thrust (added with Thrust v1.8)


//for warming up GPU:
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>


//elements for the result set
//FOR A SINGLE KERNEL INVOCATION
//NOT FOR THE BATCHED ONE
#define BUFFERELEM 300000000 //400000000-original (when removing the data from the device before putting it back for the sort)

//FOR THE BATCHED EXECUTION:
//#define BATCHTOTALELEM 1200000000 //THE TOTAL SIZE ALLOCATED ON THE HOST
//THE NUMBER OF BATCHES AND THE SIZE OF THE BUFFER FOR EACH KERNEL EXECUTION ARE NOT RELATED TO THE TOTAL NUMBER
//OF ELEMENTS (ABOVE).
// #define NUMBATCHES 20
// #define BATCHBUFFERELEM 100000000 //THE SMALLER SIZE ALLOCATED ON THE DEVICE FOR EACH KERNEL EXECUTION

#define GPUSTREAMS 1 //number of concurrent gpu streams

using namespace std;

//sort ascending
bool compareByPointValue(const key_val_sort &a, const key_val_sort &b)
{
    return a.value_at_dim < b.value_at_dim;
}



uint64_t getLinearID_nDimensions2(unsigned int * indexes, unsigned int * dimLen, unsigned int nDimensions) {
    uint64_t index = 0;
	uint64_t multiplier = 1;
	for (int i = 0; i<nDimensions; i++){
  	     index += (uint64_t)indexes[i] * multiplier;
  	      multiplier *= dimLen[i];
	}

	return index;
}






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
        std::vector<unsigned int> * sortedCellsNbThreadsBefore)
{
    struct schedulingCell* sortedCells = (struct schedulingCell*)malloc(sizeof(struct schedulingCell) * (*nNonEmptyCells));

    for(int i = 0; i < (*nNonEmptyCells); i++)
    {
        int cell = gridCellLookupArr[i].idx;
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

            if(binary_search(gridCellNDMask + gridCellNDMaskOffsets[ (n * 2) ],
    				gridCellNDMask + gridCellNDMaskOffsets[ (n * 2) + 1 ] + 1, nDMinCellIDs)){ //extra +1 here is because we include the upper bound
    			foundMin = 1;
    		}
    		if(binary_search(gridCellNDMask + gridCellNDMaskOffsets[ (n * 2) ],
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
        uint64_t originCellID = getLinearID_nDimensions2(indexes, nCells, NUMINDEXEDDIM);
        for (loopRng[0] = rangeFilteredCellIdsMin[0]; loopRng[0] <= rangeFilteredCellIdsMax[0]; loopRng[0]++)
            for (loopRng[1] = rangeFilteredCellIdsMin[1]; loopRng[1] <= rangeFilteredCellIdsMax[1]; loopRng[1]++)
            #include "kernelloops.h"
            {
                for (int x = 0; x < NUMINDEXEDDIM; x++){
                    indexes[x] = loopRng[x];
                }

                uint64_t cellID = getLinearID_nDimensions2(indexes, nCells, NUMINDEXEDDIM);
                if(originCellID <= cellID)
                {
                    struct gridCellLookup tmp;
                    tmp.gridLinearID = cellID;

                    if (binary_search(gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
                    {
                        struct gridCellLookup * resultBinSearch = lower_bound(gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
                        unsigned int GridIndex = resultBinSearch->idx;
                        nbNeighborPoints += index[GridIndex].indexmax - index[GridIndex].indexmin + 1;
                        sortedCells[i].cellId = GridIndex;
                    }
                }
            }

            sortedCells[i].nbPoints = nbNeighborPoints;
            //sortedCells[i].cellId = cell;

    }

    sort(sortedCells, sortedCells + (*nNonEmptyCells),
            [](const schedulingCell& a, const schedulingCell& b){ return a.nbPoints > b.nbPoints; });

    int nbPointsPast = 0;
    for(int i = 0; i < (*nNonEmptyCells); i++)
    {
        int cellId = sortedCells[i].cellId;
        int nbPoints = index[cellId].indexmax - index[cellId].indexmin + 1;
        int nbNeighbor = sortedCells[i].nbPoints;
        //for(int n = 0; n < nbPoints; n++)
        for(int n = index[cellId].indexmin; n <= index[cellId].indexmax; n++)
        {
            int nbThreads = (nbNeighbor / 50) + 1;
            for(int t = 0; t < nbThreads; t++)
            {
                //sortedSetDynamicThreads->push_back(index[cellId].indexmin + n);
                sortedSetDynamicThreads->push_back(n);
                sortedCellsNbThreads->push_back(nbThreads);
                sortedCellsNbThreadsBefore->push_back(t);
            }
        }
    }
    /*
    for(int i = 0; i < (*nNonEmptyCells); i++)
    {
        int cellId = sortedCells[i].cellId;
        int nbNeighbor = index[cellId].indexmax - index[cellId].indexmin + 1;
        for(int j = 0; j < nbNeighbor; j++)
        {
            int tmpId = indexLookupArr[ index[cellId].indexmin + j ];
            //TODO better adjust this value
            int nbThreads = (nbNeighbor / 15000) + 1;
            sortedCellsNbThreads->push_back(nbThreads);
            for(int n = 0; n < nbThreads; n++)
            {
                sortedSetDynamicThreads->push_back(tmpId * NUMINDEXEDDIM);
            }
        }
    }
    */
}



void schedulePointsByWorkDynamicThreadsPerPointV2(
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
        std::vector<DTYPE> * sortedSetDynamicThreads,
        std::vector<unsigned int> * sortedCellsNbThreads,
        std::vector<unsigned int> * sortedCellsNbThreadsBefore)
{
    struct schedulingCell* sortedCells = (struct schedulingCell*)malloc(sizeof(struct schedulingCell) * (*nNonEmptyCells));

    for(int i = 0; i < (*nNonEmptyCells); i++)
    {
        int cell = gridCellLookupArr[i].idx;
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

            if(binary_search(gridCellNDMask + gridCellNDMaskOffsets[ (n * 2) ],
    				gridCellNDMask + gridCellNDMaskOffsets[ (n * 2) + 1 ] + 1, nDMinCellIDs)){ //extra +1 here is because we include the upper bound
    			foundMin = 1;
    		}
    		if(binary_search(gridCellNDMask + gridCellNDMaskOffsets[ (n * 2) ],
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

        uint64_t originCellID = getLinearID_nDimensions2(indexes, nCells, NUMINDEXEDDIM);

        for (loopRng[0] = rangeFilteredCellIdsMin[0]; loopRng[0] <= rangeFilteredCellIdsMax[0]; loopRng[0]++)
            for (loopRng[1] = rangeFilteredCellIdsMin[1]; loopRng[1] <= rangeFilteredCellIdsMax[1]; loopRng[1]++)
            #include "kernelloops.h"
            {
                for (int x = 0; x < NUMINDEXEDDIM; x++){
                    indexes[x] = loopRng[x];
                }

                uint64_t cellID = getLinearID_nDimensions2(indexes, nCells, NUMINDEXEDDIM);
                if(originCellID <= cellID)
                {
                    struct gridCellLookup tmp;
                    tmp.gridLinearID = cellID;

                    if (binary_search(gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
                    {
                        struct gridCellLookup * resultBinSearch = lower_bound(gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
                        unsigned int GridIndex = resultBinSearch->idx;
                        nbNeighborPoints += index[GridIndex].indexmax - index[GridIndex].indexmin + 1;
                    }
                }
            }

            sortedCells[i].nbPoints = nbNeighborPoints;
            sortedCells[i].cellId = cell;

    }

    sort(sortedCells, sortedCells + (*nNonEmptyCells),
            [](const schedulingCell& a, const schedulingCell& b){ return a.nbPoints > b.nbPoints; });

    for(int i = 0; i < (*nNonEmptyCells); i++)
    {
        int cellId = sortedCells[i].cellId;
        int nbPoints = index[cellId].indexmax - index[cellId].indexmin + 1;
        int nbTotalNeighbor = sortedCells[i].nbPoints;
        int overhead = (nbTotalNeighbor / 50) + 1;
        for(int j = 0; j < nbPoints; j++)
        {
            int tmpId = indexLookupArr[ index[cellId].indexmin + j ];
            for(int t = 0; t < overhead; t++)
            {
                for(int n = 0; n < NUMINDEXEDDIM; n++)
                {
                    sortedSetDynamicThreads->push_back(database[tmpId * NUMINDEXEDDIM + n]);
                }
                sortedCellsNbThreads->push_back(overhead);
                sortedCellsNbThreadsBefore->push_back(t);
            }
        }




        /*
        int cellId = sortedCells[i].cellId;
        int nbPoints = index[cellId].indexmax - index[cellId].indexmin + 1;
        int nbNeighbor = sortedCells[i].nbPoints;
        //for(int n = 0; n < nbPoints; n++)
        for(int n = index[cellId].indexmin; n <= index[cellId].indexmax; n++)
        {
            int nbThreads = (nbNeighbor / 50) + 1;
            for(int t = 0; t < nbThreads; t++)
            {
                //sortedSetDynamicThreads->push_back(index[cellId].indexmin + n);
                sortedSetDynamicThreads->push_back(n);
                sortedCellsNbThreads->push_back(nbThreads);
                sortedCellsNbThreadsBefore->push_back(t);
            }
        }
        */
    }
}





std::vector<DTYPE> schedulePointsByWorkDynamicThreadsPerPointFixed(
        int searchMode,
        unsigned int N,
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
        //std::vector<DTYPE> * sortedSetDynamicThreads,
        unsigned int * totalNbThreads,
        unsigned int DBSIZE,
        unsigned int numBatches,
        std::vector<unsigned int>** tmpVectors)
{
    struct schedulingCell* sortedCells = (struct schedulingCell*)malloc(sizeof(struct schedulingCell) * (*nNonEmptyCells));

    for(int i = 0; i < (*nNonEmptyCells); i++)
    {
        int cell = gridCellLookupArr[i].idx;
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

            if(binary_search(gridCellNDMask + gridCellNDMaskOffsets[ (n * 2) ],
    				gridCellNDMask + gridCellNDMaskOffsets[ (n * 2) + 1 ] + 1, nDMinCellIDs)){ //extra +1 here is because we include the upper bound
    			foundMin = 1;
    		}
    		if(binary_search(gridCellNDMask + gridCellNDMaskOffsets[ (n * 2) ],
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

        //uint64_t originCellID = getLinearID_nDimensions2(indexes, nCells, NUMINDEXEDDIM);

        for (loopRng[0] = rangeFilteredCellIdsMin[0]; loopRng[0] <= rangeFilteredCellIdsMax[0]; loopRng[0]++)
            for (loopRng[1] = rangeFilteredCellIdsMin[1]; loopRng[1] <= rangeFilteredCellIdsMax[1]; loopRng[1]++)
            #include "kernelloops.h"
            {
                for (int x = 0; x < NUMINDEXEDDIM; x++){
                    indexes[x] = loopRng[x];
                }

                uint64_t cellID = getLinearID_nDimensions2(indexes, nCells, NUMINDEXEDDIM);
                //if(originCellID <= cellID)
                //{
                    struct gridCellLookup tmp;
                    tmp.gridLinearID = cellID;

                    if (binary_search(gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
                    {
                        struct gridCellLookup * resultBinSearch = lower_bound(gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
                        unsigned int GridIndex = resultBinSearch->idx;
                        nbNeighborPoints += index[GridIndex].indexmax - index[GridIndex].indexmin + 1;
                    }
                //}
            }

        sortedCells[i].nbPoints = nbNeighborPoints;
        sortedCells[i].cellId = cell;

    }

    sort(sortedCells, sortedCells + (*nNonEmptyCells),
            [](const schedulingCell& a, const schedulingCell& b){ return a.nbPoints > b.nbPoints; });

    int minNeighbor = INT_MAX;
    int maxNeighbor = 0;

    for(int i = 0; i < (*nNonEmptyCells); i++)
    {
        int cellId = sortedCells[i].cellId;
        //int nbPoints = index[cellId].indexmax - index[cellId].indexmin + 1;
        int nbNeighbor = sortedCells[i].nbPoints;
        minNeighbor = min(nbNeighbor, minNeighbor);
        maxNeighbor = max(nbNeighbor, maxNeighbor);
    }

    int range = (maxNeighbor - minNeighbor) / 3;
    int third3 = minNeighbor + range;
    int third1 = maxNeighbor - range;

    printf("\nMin neighbor = %d, max = %d, range = %d, third1 = %d, third3 = %d\n",
            minNeighbor, maxNeighbor, range, third1, third3);

    unsigned int currentPoint = 0;
    (*totalNbThreads) = 0;

    std::vector<DTYPE> sortedSetDynamicThreads;

    //tmpVectors = new std::vector<unsigned int>[numBatches];

    unsigned int batchSize = DBSIZE / numBatches;
	unsigned int batchesThatHaveOneMore = DBSIZE - (batchSize * numBatches); //batch number 0- < this value have one more

    unsigned int counter = 0;
    unsigned int counterInBatch = 0;
    unsigned int batchCounter = 0;
    unsigned int overhead = 0;

    for(int i = 0; i < (*nNonEmptyCells); i++)
    {
        int cellId = sortedCells[i].cellId;
        int nbPoints = index[cellId].indexmax - index[cellId].indexmin + 1;
        int nbTotalNeighbor = sortedCells[i].nbPoints;

        for(int j = 0; j < nbPoints; j++)
        {
            int tmpId = indexLookupArr[ index[cellId].indexmin + j ];
            for(int n = 0; n < NUMINDEXEDDIM; n++)
            {
                sortedSetDynamicThreads.push_back(database[tmpId * NUMINDEXEDDIM + n]);
            }


            if(nbTotalNeighbor < third3)
            {
                overhead = NB_THREAD_THIRD3;
            }else{
                if(third3 <= nbTotalNeighbor && nbTotalNeighbor < third1)
                {
                    overhead = NB_THREAD_THIRD2;
                }else{
                    if(third1 <= nbTotalNeighbor)
                    {
                        overhead = NB_THREAD_THIRD1;
                    }
                }
            }

            for(int j = 0; j < overhead; j++)
            {
                unsigned int localSize = batchSize;
                if(batchCounter < batchesThatHaveOneMore)
                {
                    localSize++;
                }
                if(counterInBatch == localSize)
                {
                    batchCounter++;
                    counterInBatch = 0;
                }
                (*tmpVectors[batchCounter]).push_back(tmpId);
                counter++;
                counterInBatch++;
                (*totalNbThreads)++;
            }
        }
    }

    return sortedSetDynamicThreads;
}







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
        DTYPE* sortedSet)
{
    struct schedulingCell* sortedCells = (struct schedulingCell*)malloc(sizeof(struct schedulingCell) * (*nNonEmptyCells));

    #pragma omp parallel for num_threads(GPUSTREAMS)
    for(int i = 0; i < (*nNonEmptyCells); i++)
    {
        int cell = gridCellLookupArr[i].idx;
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

            if(binary_search(gridCellNDMask + gridCellNDMaskOffsets[ (n * 2) ],
    				gridCellNDMask + gridCellNDMaskOffsets[ (n * 2) + 1 ] + 1, nDMinCellIDs)){ //extra +1 here is because we include the upper bound
    			foundMin = 1;
    		}
    		if(binary_search(gridCellNDMask + gridCellNDMaskOffsets[ (n * 2) ],
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


        if(4 == searchMode){ // Sorting cells by workload using the Unicomp pattern

            for(int i = 0; i < NUMINDEXEDDIM; i++) {
                indexes[i] = nDCellIDs[i];
            }

            uint64_t cellID = getLinearID_nDimensions2(indexes, nCells, NUMINDEXEDDIM);
            struct gridCellLookup tmp;
            tmp.gridLinearID = cellID;

            if (binary_search(gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
            {
                struct gridCellLookup * resultBinSearch = lower_bound(gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
                unsigned int GridIndex = resultBinSearch->idx;
                nbNeighborPoints += index[GridIndex].indexmax - index[GridIndex].indexmin + 1;
            }
            #include "stamploopsSortingCells.h"

        }else if(5 == searchMode){ // Sorting cells by workload using the B-Unicomp pattern

            #if NUMINDEXEDDIM==2
                indexes[0] = nDCellIDs[0];
                indexes[1] = nDCellIDs[1];

                unsigned int colorId = nDCellIDs[0] + nDCellIDs[1];

                uint64_t originCellID = getLinearID_nDimensions2(indexes, nCells, NUMINDEXEDDIM);
                struct gridCellLookup tmp;
                tmp.gridLinearID = originCellID;

                if (binary_search(gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
                {
                    struct gridCellLookup * resultBinSearch = lower_bound(gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
                    unsigned int GridIndex = resultBinSearch->idx;
                    nbNeighborPoints += index[GridIndex].indexmax - index[GridIndex].indexmin + 1;
                }

                for(loopRng[0] = rangeFilteredCellIdsMin[0]; loopRng[0] <= rangeFilteredCellIdsMax[0]; loopRng[0]++)
                    for(loopRng[1] = rangeFilteredCellIdsMin[1]; loopRng[1] <= rangeFilteredCellIdsMax[1]; loopRng[1]++)
                    {
                        if( ( (1 == colorId % 2) && (nDCellIDs[1] <= loopRng[1]) && (nDCellIDs[0] != loopRng[0]) )
                            || ( (0 == colorId % 2) && ((nDCellIDs[1] < loopRng[1]) || (loopRng[1] < nDCellIDs[1] && loopRng[0] == nDCellIDs[0])) ) ) // ( odd => red pattern ) || ( even => green pattern )
                        {
                            indexes[0] = loopRng[0];
                            indexes[1] = loopRng[1];

                            uint64_t cellID = getLinearID_nDimensions2(indexes, nCells, NUMINDEXEDDIM);
                            struct gridCellLookup tmp;
                            tmp.gridLinearID = cellID;

                            if (binary_search(gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
                            {
                                struct gridCellLookup * resultBinSearch = lower_bound(gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
                                unsigned int GridIndex = resultBinSearch->idx;
                                nbNeighborPoints += index[GridIndex].indexmax - index[GridIndex].indexmin + 1;
                            }
                        }
                    }
            #else
                #if NUMINDEXEDDIM==3
                    indexes[0] = nDCellIDs[0];
                    indexes[1] = nDCellIDs[1];
                    indexes[2] = nDCellIDs[2];

                    unsigned int colorId = nDCellIDs[0] + nDCellIDs[1] + nDCellIDs[2];

                    uint64_t originCellID = getLinearID_nDimensions2(indexes, nCells, NUMINDEXEDDIM);
                    struct gridCellLookup tmp;
                    tmp.gridLinearID = originCellID;

                    if (binary_search(gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
                    {
                        struct gridCellLookup * resultBinSearch = lower_bound(gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
                        unsigned int GridIndex = resultBinSearch->idx;
                        nbNeighborPoints += index[GridIndex].indexmax - index[GridIndex].indexmin + 1;
                    }

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

                                    uint64_t cellID = getLinearID_nDimensions2(indexes, nCells, NUMINDEXEDDIM);
                                    struct gridCellLookup tmp;
                                    tmp.gridLinearID = cellID;

                                    if (binary_search(gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
                                    {
                                        struct gridCellLookup * resultBinSearch = lower_bound(gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
                                        unsigned int GridIndex = resultBinSearch->idx;
                                        nbNeighborPoints += index[GridIndex].indexmax - index[GridIndex].indexmin + 1;
                                    }
                                }
                            }
                #endif
            #endif

        }else if(6 == searchMode || 10 == searchMode){ // Sorting cells by workload using the Linear ID Unicomp

            for (int x = 0; x < NUMINDEXEDDIM; x++){
                indexes[x] = nDCellIDs[x];
            }
            uint64_t originCellID = getLinearID_nDimensions2(indexes, nCells, NUMINDEXEDDIM);
            for (loopRng[0] = rangeFilteredCellIdsMin[0]; loopRng[0] <= rangeFilteredCellIdsMax[0]; loopRng[0]++)
                for (loopRng[1] = rangeFilteredCellIdsMin[1]; loopRng[1] <= rangeFilteredCellIdsMax[1]; loopRng[1]++)
                #include "kernelloops.h"
                {
                    for (int x = 0; x < NUMINDEXEDDIM; x++){
                        indexes[x] = loopRng[x];
                    }

                    uint64_t cellID = getLinearID_nDimensions2(indexes, nCells, NUMINDEXEDDIM);
                    if(originCellID <= cellID)
                    {
                        struct gridCellLookup tmp;
                        tmp.gridLinearID = cellID;

                        if (binary_search(gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
                        {
                            struct gridCellLookup * resultBinSearch = lower_bound(gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
                            unsigned int GridIndex = resultBinSearch->idx;
                            nbNeighborPoints += index[GridIndex].indexmax - index[GridIndex].indexmin + 1;
                        }
                    }
                }

        }else{ // Sorting cells by workload using no computational pattern

            for (loopRng[0] = rangeFilteredCellIdsMin[0]; loopRng[0] <= rangeFilteredCellIdsMax[0]; loopRng[0]++)
                for (loopRng[1] = rangeFilteredCellIdsMin[1]; loopRng[1] <= rangeFilteredCellIdsMax[1]; loopRng[1]++)
                #include "kernelloops.h"
                {
                    for (int x = 0; x < NUMINDEXEDDIM; x++){
                        indexes[x] = loopRng[x];
                    }

                    uint64_t cellID = getLinearID_nDimensions2(indexes, nCells, NUMINDEXEDDIM);
                    struct gridCellLookup tmp;
                    tmp.gridLinearID = cellID;
                    if (binary_search(gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
                    {
                        struct gridCellLookup * resultBinSearch = lower_bound(gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
                        unsigned int GridIndex = resultBinSearch->idx;
                        nbNeighborPoints += index[GridIndex].indexmax - index[GridIndex].indexmin + 1;
                    }

                }

        }

        sortedCells[i].nbPoints = nbNeighborPoints;
        sortedCells[i].cellId = cell;

    }

    sort(sortedCells, sortedCells + (*nNonEmptyCells),
            [](const schedulingCell& a, const schedulingCell& b){ return a.nbPoints > b.nbPoints; });

    int prec = 0;
    for(int i = 0; i < (*nNonEmptyCells); i++)
    {
        int cellId = sortedCells[i].cellId;
        int nbNeighbor = index[cellId].indexmax - index[cellId].indexmin + 1;
        for(int j = 0; j < nbNeighbor; j++)
        {
            int tmpId = indexLookupArr[ index[cellId].indexmin + j ];
            //cout << tmpId << endl;
            for(int n = 0; n < NUMINDEXEDDIM; n++)
            {
                sortedSet[(prec + j) * NUMINDEXEDDIM + n] = database[tmpId * NUMINDEXEDDIM + n];
            }
        }
        prec += nbNeighbor;
    }
}




unsigned long long callGPUBatchEst(unsigned int * DBSIZE, DTYPE* dev_database, DTYPE* dev_epsilon, struct grid * dev_grid,
	unsigned int * dev_indexLookupArr, struct gridCellLookup * dev_gridCellLookupArr, DTYPE* dev_minArr,
	unsigned int * dev_nCells, unsigned int * dev_nNonEmptyCells, unsigned int * dev_gridCellNDMask,
	unsigned int * dev_gridCellNDMaskOffsets, unsigned int * retNumBatches, unsigned int * retGPUBufferSize)
{
	//CUDA error code:
	cudaError_t errCode;

	printf("\n\n***********************************\nEstimating Batches:");
	cout << "\n** BATCH ESTIMATOR: Sometimes the GPU will error on a previous execution and you won't know. \nLast error start of function: " << cudaGetLastError();

    //////////////////////////////////////////////////////////
	//ESTIMATE THE BUFFER SIZE AND NUMBER OF BATCHES ETC BY COUNTING THE NUMBER OF RESULTS
	//TAKE A SAMPLE OF THE DATA POINTS, NOT ALL OF THEM
	//Use sampleRate for this
	/////////////////////////////////////////////////////////

	// printf("\nDon't estimate: calculate the entire thing (for testing)");
	//Parameters for the batch size estimation.
	double sampleRate = 0.01; //sample 1% of the points in the dataset sampleRate=0.01.
						//Sample the entire dataset(no sampling) sampleRate=1
	int offsetRate = 1.0 / sampleRate;
	printf("\nOffset: %d", offsetRate);

	/////////////////
	//N GPU threads
	////////////////

	unsigned int * dev_N_batchEst;
	dev_N_batchEst = (unsigned int*)malloc(sizeof(unsigned int));

	unsigned int * N_batchEst;
	N_batchEst = (unsigned int*)malloc(sizeof(unsigned int));
	*N_batchEst = *DBSIZE * sampleRate;

	//allocate on the device
	errCode = cudaMalloc((void**)&dev_N_batchEst, sizeof(unsigned int));
	if(errCode != cudaSuccess)
    {
    	cout << "\nError: dev_N_batchEst Got error with code " << errCode << endl;
	}

	//copy N to device
	//N IS THE NUMBER OF THREADS
	errCode = cudaMemcpy( dev_N_batchEst, N_batchEst, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess)
    {
	    cout << "\nError: N batchEST Got error with code " << errCode << endl;
	}

	/////////////
	//count the result set size
	////////////

	unsigned int * dev_cnt_batchEst;
	dev_cnt_batchEst = (unsigned int*)malloc(sizeof(unsigned int));

	unsigned int * cnt_batchEst;
	cnt_batchEst = (unsigned int*)malloc(sizeof(unsigned int));
	*cnt_batchEst = 0;

	//allocate on the device
	errCode = cudaMalloc((void**)&dev_cnt_batchEst, sizeof(unsigned int));
	if(errCode != cudaSuccess)
    {
    	cout << "\nError: dev_cnt_batchEst Got error with code " << errCode << endl;
	}

	//copy cnt to device
	errCode = cudaMemcpy( dev_cnt_batchEst, cnt_batchEst, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess)
    {
    	cout << "\nError: dev_cnt_batchEst Got error with code " << errCode << endl;
	}

	//////////////////
	//SAMPLE OFFSET - TO SAMPLE THE DATA TO ESTIMATE THE TOTAL NUMBER OF KEY VALUE PAIRS
	/////////////////

	//offset into the database when batching the results
	unsigned int * sampleOffset;
	sampleOffset = (unsigned int*)malloc(sizeof(unsigned int));
	*sampleOffset = offsetRate;

	unsigned int * dev_sampleOffset;
	dev_sampleOffset = (unsigned int*)malloc(sizeof(unsigned int));

	//allocate on the device
	errCode = cudaMalloc((void**)&dev_sampleOffset, sizeof(unsigned int));
	if(errCode != cudaSuccess)
    {
    	cout << "\nError: sample offset Got error with code " << errCode << endl;
	}

	//copy offset to device
	errCode = cudaMemcpy( dev_sampleOffset, sampleOffset, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess)
    {
    	cout << "\nError: dev_sampleOffset Got error with code " << errCode << endl;
	}

	////////////////////////////////////
	//TWO DEBUG VALUES SENT TO THE GPU FOR GOOD MEASURE
	////////////////////////////////////

	//debug values
	unsigned int * dev_debug1;
	dev_debug1 = (unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug1 = 0;

	unsigned int * dev_debug2;
	dev_debug2 = (unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug2 = 0;

	unsigned int * debug1;
	debug1 = (unsigned int *)malloc(sizeof(unsigned int ));
	*debug1 = 0;

	unsigned int * debug2;
	debug2 = (unsigned int *)malloc(sizeof(unsigned int ));
	*debug2 = 0;

	//allocate on the device
	errCode = cudaMalloc( (unsigned int **)&dev_debug1, sizeof(unsigned int ) );
	if(errCode != cudaSuccess)
    {
    	cout << "\nError: debug1 alloc -- error with code " << errCode << endl;
	}
	errCode = cudaMalloc( (unsigned int **)&dev_debug2, sizeof(unsigned int ) );
	if(errCode != cudaSuccess)
    {
    	cout << "\nError: debug2 alloc -- error with code " << errCode << endl;
	}

	//copy debug to device
	errCode = cudaMemcpy( dev_debug1, debug1, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess)
    {
    	cout << "\nError: dev_debug1 copy to device -- error with code " << errCode << endl;
	}
	errCode = cudaMemcpy( dev_debug2, debug2, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess)
    {
    	cout << "\nError: dev_debug2 copy to device -- with code " << errCode << endl;
	}

	////////////////////////////////////
	//END TWO DEBUG VALUES SENT TO THE GPU FOR GOOD MEASURE
	////////////////////////////////////

	const int TOTALBLOCKSBATCHEST = ceil((1.0 * (*DBSIZE) * sampleRate) / (1.0 * BLOCKSIZE));
	printf("\ntotal blocks: %d", TOTALBLOCKSBATCHEST);

	// __global__ void kernelNDGridIndexBatchEstimator(unsigned int *debug1, unsigned int *debug2, unsigned int *N,
	// unsigned int * sampleOffset, double * database, double *epsilon, struct grid * index, unsigned int * indexLookupArr,
	// struct gridCellLookup * gridCellLookupArr, double * minArr, unsigned int * nCells, unsigned int * cnt,
	// unsigned int * nNonEmptyCells)

	kernelNDGridIndexBatchEstimatorOLD<<< TOTALBLOCKSBATCHEST, BLOCKSIZE>>>(dev_debug1, dev_debug2, dev_N_batchEst,
		dev_sampleOffset, dev_database, dev_epsilon, dev_grid, dev_indexLookupArr,
		dev_gridCellLookupArr, dev_minArr, dev_nCells, dev_cnt_batchEst, dev_nNonEmptyCells, dev_gridCellNDMask,
		dev_gridCellNDMaskOffsets);

	cout << "\n** ERROR FROM KERNEL LAUNCH OF BATCH ESTIMATOR: " << cudaGetLastError();
	// find the size of the number of results

	errCode = cudaMemcpy( cnt_batchEst, dev_cnt_batchEst, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	if(errCode != cudaSuccess)
    {
	    cout << "\nError: getting cnt for batch estimate from GPU Got error with code " << errCode << endl;
	}else{
		printf("\nGPU: result set size for estimating the number of batches (sampled): %u", *cnt_batchEst);
	}

	uint64_t estimatedNeighbors = (uint64_t)*cnt_batchEst * (uint64_t)offsetRate;
	printf("\nFrom gpu cnt: %d, offset rate: %d", *cnt_batchEst, offsetRate);
	//initial

	unsigned int GPUBufferSize = 40000000; //size in HPBDC paper (low-D)
	// unsigned int GPUBufferSize = 50000000;
    // unsigned int GPUBufferSize = 100000000;

	double alpha = 0.05; //overestimation factor

	uint64_t estimatedTotalSizeWithAlpha = estimatedNeighbors * (1.0 + alpha * 1.0);
	printf("\nEstimated total result set size: %lu", estimatedNeighbors);
	printf("\nEstimated total result set size (with Alpha %f): %lu", alpha, estimatedTotalSizeWithAlpha);

	if (estimatedNeighbors < (GPUBufferSize*GPUSTREAMS))
	{
		printf("\nSmall buffer size, increasing alpha to: %f", alpha * 3.0);
		GPUBufferSize = estimatedNeighbors * (1.0 + (alpha * 2.0)) / (GPUSTREAMS);		//we do 2*alpha for small datasets because the
																		//sampling will be worse for small datasets
																		//but we fix the 3 streams still (thats why divide by 3).
	}

	unsigned int numBatches = ceil(((1.0 + alpha) * estimatedNeighbors * 1.0) / ((uint64_t)GPUBufferSize * 1.0));
	printf("\nNumber of batches: %d, buffer size: %d", numBatches, GPUBufferSize);

	*retNumBatches = numBatches;
	*retGPUBufferSize = 1.5 * GPUBufferSize;

	printf("\nEnd Batch Estimator\n***********************************\n");

	cudaFree(dev_cnt_batchEst);
	cudaFree(dev_N_batchEst);
	cudaFree(dev_sampleOffset);

    return estimatedTotalSizeWithAlpha;

}






unsigned long long callGPUBatchEstWorkQueue(unsigned int * DBSIZE, DTYPE* dev_database, DTYPE * dev_sortedCells, DTYPE* dev_epsilon,
    struct grid * dev_grid, unsigned int * dev_indexLookupArr, struct gridCellLookup * dev_gridCellLookupArr, DTYPE* dev_minArr,
	unsigned int * dev_nCells, unsigned int * dev_nNonEmptyCells, unsigned int * dev_gridCellNDMask,
	unsigned int * dev_gridCellNDMaskOffsets, unsigned int * retNumBatches, unsigned int * retGPUBufferSize)
{
	//CUDA error code:
	cudaError_t errCode;

	printf("\n\n***********************************\nEstimating Batches:");
	cout << "\n** BATCH ESTIMATOR: Sometimes the GPU will error on a previous execution and you won't know. \nLast error start of function: " << cudaGetLastError();

    //////////////////////////////////////////////////////////
	//ESTIMATE THE BUFFER SIZE AND NUMBER OF BATCHES ETC BY COUNTING THE NUMBER OF RESULTS
	//TAKE A SAMPLE OF THE DATA POINTS, NOT ALL OF THEM
	//Use sampleRate for this
	/////////////////////////////////////////////////////////

	// printf("\nDon't estimate: calculate the entire thing (for testing)");
	//Parameters for the batch size estimation.
	double sampleRate = 0.01; //sample 1% of the points in the dataset sampleRate=0.01.
						//Sample the entire dataset(no sampling) sampleRate=1
	int offsetRate = 1.0 / sampleRate;
	printf("\nOffset: %d", offsetRate);

	/////////////////
	//N GPU threads
	////////////////

	unsigned int * dev_N_batchEst;
	dev_N_batchEst = (unsigned int*)malloc(sizeof(unsigned int));

	unsigned int * N_batchEst;
	N_batchEst = (unsigned int*)malloc(sizeof(unsigned int));
	*N_batchEst = *DBSIZE * sampleRate;

	//allocate on the device
	errCode = cudaMalloc((void**)&dev_N_batchEst, sizeof(unsigned int));
	if(errCode != cudaSuccess)
    {
    	cout << "\nError: dev_N_batchEst Got error with code " << errCode << endl;
        cout << cudaGetErrorName(errCode) << ": " << cudaGetErrorString(errCode) << endl;
	}

	//copy N to device
	//N IS THE NUMBER OF THREADS
	errCode = cudaMemcpy( dev_N_batchEst, N_batchEst, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess)
    {
	    cout << "\nError: N batchEST Got error with code " << errCode << endl;
	}

	/////////////
	//count the result set size
	////////////

	unsigned int * dev_cnt_batchEst;
	dev_cnt_batchEst = (unsigned int*)malloc(sizeof(unsigned int));

	unsigned int * cnt_batchEst;
	cnt_batchEst = (unsigned int*)malloc(sizeof(unsigned int));
	*cnt_batchEst = 0;

	//allocate on the device
	errCode = cudaMalloc((void**)&dev_cnt_batchEst, sizeof(unsigned int));
	if(errCode != cudaSuccess)
    {
    	cout << "\nError: dev_cnt_batchEst Got error with code " << errCode << endl;
	}

	//copy cnt to device
	errCode = cudaMemcpy( dev_cnt_batchEst, cnt_batchEst, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess)
    {
    	cout << "\nError: dev_cnt_batchEst Got error with code " << errCode << endl;
	}

	//////////////////
	//SAMPLE OFFSET - TO SAMPLE THE DATA TO ESTIMATE THE TOTAL NUMBER OF KEY VALUE PAIRS
	/////////////////

	//offset into the database when batching the results
	unsigned int * sampleOffset;
	sampleOffset = (unsigned int*)malloc(sizeof(unsigned int));
	*sampleOffset = offsetRate;

	unsigned int * dev_sampleOffset;
	dev_sampleOffset = (unsigned int*)malloc(sizeof(unsigned int));

	//allocate on the device
	errCode = cudaMalloc((void**)&dev_sampleOffset, sizeof(unsigned int));
	if(errCode != cudaSuccess)
    {
    	cout << "\nError: sample offset Got error with code " << errCode << endl;
	}

	//copy offset to device
	errCode = cudaMemcpy( dev_sampleOffset, sampleOffset, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess)
    {
    	cout << "\nError: dev_sampleOffset Got error with code " << errCode << endl;
	}

	////////////////////////////////////
	//TWO DEBUG VALUES SENT TO THE GPU FOR GOOD MEASURE
	////////////////////////////////////

	//debug values
	unsigned int * dev_debug1;
	dev_debug1 = (unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug1 = 0;

	unsigned int * dev_debug2;
	dev_debug2 = (unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug2 = 0;

	unsigned int * debug1;
	debug1 = (unsigned int *)malloc(sizeof(unsigned int ));
	*debug1 = 0;

	unsigned int * debug2;
	debug2 = (unsigned int *)malloc(sizeof(unsigned int ));
	*debug2 = 0;

	//allocate on the device
	errCode = cudaMalloc( (unsigned int **)&dev_debug1, sizeof(unsigned int ) );
	if(errCode != cudaSuccess)
    {
    	cout << "\nError: debug1 alloc -- error with code " << errCode << endl;
	}
	errCode = cudaMalloc( (unsigned int **)&dev_debug2, sizeof(unsigned int ) );
	if(errCode != cudaSuccess)
    {
    	cout << "\nError: debug2 alloc -- error with code " << errCode << endl;
	}

	//copy debug to device
	errCode = cudaMemcpy( dev_debug1, debug1, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess)
    {
    	cout << "\nError: dev_debug1 copy to device -- error with code " << errCode << endl;
	}
	errCode = cudaMemcpy( dev_debug2, debug2, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess)
    {
    	cout << "\nError: dev_debug2 copy to device -- with code " << errCode << endl;
	}

	////////////////////////////////////
	//END TWO DEBUG VALUES SENT TO THE GPU FOR GOOD MEASURE
	////////////////////////////////////

	const int TOTALBLOCKSBATCHEST = ceil((1.0 * (*DBSIZE) * sampleRate) / (1.0 * BLOCKSIZE));
	printf("\ntotal blocks: %d", TOTALBLOCKSBATCHEST);

	// __global__ void kernelNDGridIndexBatchEstimator(unsigned int *debug1, unsigned int *debug2, unsigned int *N,
	// unsigned int * sampleOffset, double * database, double *epsilon, struct grid * index, unsigned int * indexLookupArr,
	// struct gridCellLookup * gridCellLookupArr, double * minArr, unsigned int * nCells, unsigned int * cnt,
	// unsigned int * nNonEmptyCells)

	kernelNDGridIndexWorkQueueBatchEstimatorOLD<<< TOTALBLOCKSBATCHEST, BLOCKSIZE>>>(dev_debug1, dev_debug2, dev_N_batchEst,
		dev_sampleOffset, dev_database, dev_sortedCells, dev_epsilon, dev_grid, dev_indexLookupArr,
		dev_gridCellLookupArr, dev_minArr, dev_nCells, dev_cnt_batchEst, dev_nNonEmptyCells, dev_gridCellNDMask,
		dev_gridCellNDMaskOffsets);

	cout << "\n** ERROR FROM KERNEL LAUNCH OF BATCH ESTIMATOR: " << cudaGetLastError();
	// find the size of the number of results

	errCode = cudaMemcpy( cnt_batchEst, dev_cnt_batchEst, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	if(errCode != cudaSuccess)
    {
	    cout << "\nError: getting cnt for batch estimate from GPU Got error with code " << errCode << endl;
	}else{
		printf("\nGPU: result set size for estimating the number of batches (sampled): %u", *cnt_batchEst);
	}

	uint64_t estimatedNeighbors = (uint64_t)*cnt_batchEst * (uint64_t)offsetRate;
	printf("\nFrom gpu cnt: %d, offset rate: %d", *cnt_batchEst, offsetRate);
	//initial

	unsigned int GPUBufferSize = 40000000; //size in HPBDC paper (low-D)
	// unsigned int GPUBufferSize = 50000000;
    // unsigned int GPUBufferSize = 100000000;

	double alpha = 0.05; //overestimation factor

	uint64_t estimatedTotalSizeWithAlpha = estimatedNeighbors * (1.0 + alpha * 1.0);
	printf("\nEstimated total result set size: %lu", estimatedNeighbors);
	printf("\nEstimated total result set size (with Alpha %f): %lu", alpha, estimatedTotalSizeWithAlpha);

	if (estimatedNeighbors < (GPUBufferSize*GPUSTREAMS))
	{
		printf("\nSmall buffer size, increasing alpha to: %f", alpha * 3.0);
		GPUBufferSize = estimatedNeighbors * (1.0 + (alpha * 2.0)) / (GPUSTREAMS);		//we do 2*alpha for small datasets because the
																		//sampling will be worse for small datasets
																		//but we fix the 3 streams still (thats why divide by 3).
	}

	unsigned int numBatches = ceil(((1.0 + alpha) * estimatedNeighbors * 1.0) / ((uint64_t)GPUBufferSize * 1.0));
	printf("\nNumber of batches: %d, buffer size: %d", numBatches, GPUBufferSize);

	*retNumBatches = numBatches;
	*retGPUBufferSize = 1.5 * GPUBufferSize;

	printf("\nEnd Batch Estimator\n***********************************\n");

	cudaFree(dev_cnt_batchEst);
	cudaFree(dev_N_batchEst);
	cudaFree(dev_sampleOffset);

    return estimatedTotalSizeWithAlpha;

}










//modified from: makeDistanceTableGPUGridIndexBatchesAlternateTest

void distanceTableNDGridBatches(
		int searchMode,
		std::vector<std::vector<DTYPE> > * NDdataPoints,
		DTYPE* epsilon,
		struct grid * index,
		struct gridCellLookup * gridCellLookupArr,
		unsigned int * nNonEmptyCells,
		DTYPE* minArr,
		unsigned int * nCells,
		unsigned int * indexLookupArr,
		struct neighborTableLookup * neighborTable,
		std::vector<struct neighborDataPtrs> * pointersToNeighbors,
		uint64_t * totalNeighbors,
		unsigned int * gridCellNDMask,
		unsigned int * gridCellNDMaskOffsets,
		unsigned int * nNDMaskElems)
{

	double tKernelResultsStart = omp_get_wtime();

	//CUDA error code:
	cudaError_t errCode;

	cout<<"\n** Sometimes the GPU will error on a previous execution and you won't know. \nLast error start of function: "<<cudaGetLastError();

	///////////////////////////////////
	//COPY THE DATABASE TO THE GPU
	///////////////////////////////////

	unsigned int * DBSIZE;
	DBSIZE = (unsigned int*)malloc(sizeof(unsigned int));
	*DBSIZE = NDdataPoints->size();

	printf("\nIn main GPU method: DBSIZE is: %u",*DBSIZE);
    cout.flush();

	DTYPE* database = (DTYPE*)malloc(sizeof(DTYPE) * (*DBSIZE) * (GPUNUMDIM));
	DTYPE* dev_database = (DTYPE*)malloc(sizeof(DTYPE) * (*DBSIZE) * (GPUNUMDIM));

	//allocate memory on device:
	errCode = cudaMalloc( (void**)&dev_database, sizeof(DTYPE) * (GPUNUMDIM) * (*DBSIZE));
	if(errCode != cudaSuccess) {
		cout << "\nError: database alloc -- error with code " << errCode << endl;
        cout.flush();
	}

	//copy the database from the ND vector to the array:
	for (int i = 0; i < (*DBSIZE); i++){
		std::copy((*NDdataPoints)[i].begin(), (*NDdataPoints)[i].end(), database + (i * (GPUNUMDIM)));
	}

	//copy database to the device
	errCode = cudaMemcpy(dev_database, database, sizeof(DTYPE) * (GPUNUMDIM) * (*DBSIZE), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
		cout << "\nError: database2 Got error with code " << errCode << endl;
	}

	///////////////////////////////////
	//END COPY THE DATABASE TO THE GPU
	///////////////////////////////////

	///////////////////////////////////
	//COPY THE INDEX TO THE GPU
	///////////////////////////////////

	struct grid * dev_grid;
	dev_grid = (struct grid*)malloc(sizeof(struct grid) * (*nNonEmptyCells));

	//allocate memory on device:
	errCode = cudaMalloc( (void**)&dev_grid, sizeof(struct grid) * (*nNonEmptyCells));
	if(errCode != cudaSuccess) {
		cout << "\nError: grid index -- error with code " << errCode << endl;
        cout.flush();
	}

	//copy grid index to the device:
	errCode = cudaMemcpy(dev_grid, index, sizeof(struct grid) * (*nNonEmptyCells), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
		cout << "\nError: grid index copy to device -- error with code " << errCode << endl;
	}

	printf("\nSize of index sent to GPU (MiB): %f", (DTYPE)sizeof(struct grid) * (*nNonEmptyCells) / (1024.0 * 1024.0));

	///////////////////////////////////
	//END COPY THE INDEX TO THE GPU
	///////////////////////////////////

	unsigned int * dev_indexLookupArr;
	dev_indexLookupArr = (unsigned int*)malloc(sizeof(unsigned int) * (*DBSIZE));

	//allocate memory on device:
	errCode = cudaMalloc( (void**)&dev_indexLookupArr, sizeof(unsigned int) * (*DBSIZE));
	if(errCode != cudaSuccess) {
		cout << "\nError: lookup array allocation -- error with code " << errCode << endl;
        cout.flush();
	}

	//copy lookup array to the device:
	errCode = cudaMemcpy(dev_indexLookupArr, indexLookupArr, sizeof(unsigned int) * (*DBSIZE), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
		cout << "\nError: copy lookup array to device -- error with code " << errCode << endl;
	}

	///////////////////////////////////
	//END COPY THE LOOKUP ARRAY TO THE DATA ELEMS TO THE GPU
	///////////////////////////////////



	///////////////////////////////////
	//COPY THE GRID CELL LOOKUP ARRAY
	///////////////////////////////////

	struct gridCellLookup * dev_gridCellLookupArr;
	dev_gridCellLookupArr = (struct gridCellLookup*)malloc(sizeof(struct gridCellLookup) * (*nNonEmptyCells));

	//allocate memory on device:
	errCode = cudaMalloc( (void**)&dev_gridCellLookupArr, sizeof(struct gridCellLookup) * (*nNonEmptyCells));
	if(errCode != cudaSuccess) {
		cout << "\nError: copy grid cell lookup array allocation -- error with code " << errCode << endl;
        cout.flush();
	}

	//copy lookup array to the device:
	errCode = cudaMemcpy(dev_gridCellLookupArr, gridCellLookupArr, sizeof(struct gridCellLookup) * (*nNonEmptyCells), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
		cout << "\nError: copy grid cell lookup array to device -- error with code " << errCode << endl;
	}

	///////////////////////////////////
	//END COPY THE GRID CELL LOOKUP ARRAY
	///////////////////////////////////


	///////////////////////////////////
	//COPY GRID DIMENSIONS TO THE GPU
	//THIS INCLUDES THE NUMBER OF CELLS IN EACH DIMENSION,
	//AND THE STARTING POINT OF THE GRID IN THE DIMENSIONS
	///////////////////////////////////

	//minimum boundary of the grid:
	DTYPE* dev_minArr;
	dev_minArr = (DTYPE*)malloc(sizeof(DTYPE) * (NUMINDEXEDDIM));

	//Allocate on the device
	errCode = cudaMalloc((void**)&dev_minArr, sizeof(DTYPE) * (NUMINDEXEDDIM));
	if(errCode != cudaSuccess) {
		cout << "\nError: Alloc minArr -- error with code " << errCode << endl;
	}

	errCode = cudaMemcpy( dev_minArr, minArr, sizeof(DTYPE) * (NUMINDEXEDDIM), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
		cout << "\nError: Copy minArr to device -- error with code " << errCode << endl;
	}

	//number of cells in each dimension
	unsigned int * dev_nCells;
	dev_nCells = (unsigned int*)malloc(sizeof(unsigned int) * (NUMINDEXEDDIM));

	//Allocate on the device
	errCode = cudaMalloc((void**)&dev_nCells, sizeof(unsigned int) * (NUMINDEXEDDIM));
	if(errCode != cudaSuccess) {
		cout << "\nError: Alloc nCells -- error with code " << errCode << endl;
	}

	errCode = cudaMemcpy( dev_nCells, nCells, sizeof(unsigned int) * (NUMINDEXEDDIM), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
		cout << "\nError: Copy nCells to device -- error with code " << errCode << endl;
	}

	///////////////////////////////////
	//END COPY GRID DIMENSIONS TO THE GPU
	///////////////////////////////////



	///////////////////////////////////
	//COUNT VALUES -- RESULT SET SIZE FOR EACH KERNEL INVOCATION
	///////////////////////////////////

	//total size of the result set as it's batched
	//this isnt sent to the GPU
	unsigned int * totalResultSetCnt;
	totalResultSetCnt = (unsigned int*)malloc(sizeof(unsigned int));
	*totalResultSetCnt = 0;

	//count values - for an individual kernel launch
	//need different count values for each stream
	unsigned int * cnt;
	cnt = (unsigned int*)malloc(sizeof(unsigned int) * GPUSTREAMS);
	*cnt = 0;

	unsigned int * dev_cnt;
	dev_cnt = (unsigned int*)malloc(sizeof(unsigned int) * GPUSTREAMS);
	*dev_cnt = 0;

	//allocate on the device
	errCode = cudaMalloc((void**)&dev_cnt, sizeof(unsigned int) * GPUSTREAMS);
	if(errCode != cudaSuccess) {
		cout << "\nError: Alloc cnt -- error with code " << errCode << endl;
	}

	///////////////////////////////////
	//END COUNT VALUES -- RESULT SET SIZE FOR EACH KERNEL INVOCATION
	///////////////////////////////////



	///////////////////////////////////
	//EPSILON
	///////////////////////////////////
	DTYPE* dev_epsilon;
	dev_epsilon = (DTYPE*)malloc(sizeof(DTYPE));

	//Allocate on the device
	errCode = cudaMalloc((void**)&dev_epsilon, sizeof(DTYPE));
	if(errCode != cudaSuccess) {
		cout << "\nError: Alloc epsilon -- error with code " << errCode << endl;
	}

	//copy to device
	errCode = cudaMemcpy( dev_epsilon, epsilon, sizeof(DTYPE), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
		cout << "\nError: epsilon copy to device -- error with code " << errCode << endl;
	}

	///////////////////////////////////
	//END EPSILON
	///////////////////////////////////


	///////////////////////////////////
	//NUMBER OF NON-EMPTY CELLS
	///////////////////////////////////
	unsigned int * dev_nNonEmptyCells;
	dev_nNonEmptyCells = (unsigned int*)malloc(sizeof( unsigned int ));

	//Allocate on the device
	errCode = cudaMalloc((void**)&dev_nNonEmptyCells, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
		cout << "\nError: Alloc nNonEmptyCells -- error with code " << errCode << endl;
	}

	//copy to device
	errCode = cudaMemcpy( dev_nNonEmptyCells, nNonEmptyCells, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
		cout << "\nError: nNonEmptyCells copy to device -- error with code " << errCode << endl;
	}

	///////////////////////////////////
	//NUMBER OF NON-EMPTY CELLS
	///////////////////////////////////



	//////////////////////////////////
	//ND MASK -- The array, the offsets, and the size of the array
	//////////////////////////////////
	//unsigned int * gridCellNDMask, unsigned int * gridCellNDMaskOffsets, unsigned int * nNDMaskElems

	//NDMASK
	unsigned int * dev_gridCellNDMask;
	dev_gridCellNDMask = (unsigned int*)malloc(sizeof(unsigned int) * (*nNDMaskElems));

	//Allocate on the device
	errCode = cudaMalloc((void**)&dev_gridCellNDMask, sizeof(unsigned int) * (*nNDMaskElems));
	if(errCode != cudaSuccess) {
		cout << "\nError: Alloc gridCellNDMask -- error with code " << errCode << endl;
	}

	errCode = cudaMemcpy( dev_gridCellNDMask, gridCellNDMask, sizeof(unsigned int)*(*nNDMaskElems), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
		cout << "\nError: Copy gridCellNDMask to device -- error with code " << errCode << endl;
	}

	//NDMASKOFFSETS
	unsigned int * dev_gridCellNDMaskOffsets;
	dev_gridCellNDMaskOffsets = (unsigned int*)malloc(sizeof(unsigned int) * (2 * NUMINDEXEDDIM));

	//Allocate on the device
	errCode = cudaMalloc((void**)&dev_gridCellNDMaskOffsets, sizeof(unsigned int) * (2 * NUMINDEXEDDIM));
	if(errCode != cudaSuccess) {
		cout << "\nError: Alloc gridCellNDMaskOffsets -- error with code " << errCode << endl;
	}

	errCode = cudaMemcpy( dev_gridCellNDMaskOffsets, gridCellNDMaskOffsets, sizeof(unsigned int) * (2 * NUMINDEXEDDIM), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
		cout << "\nError: Copy gridCellNDMaskOffsets to device -- error with code " << errCode << endl;
	}

	//////////////////////////////////
	//End ND MASK -- The array, the offsets, and the size of the array
	//////////////////////////////////





	////////////////////////////////////
	//NUMBER OF THREADS PER GPU STREAM
	////////////////////////////////////

	//THE NUMBER OF THREADS THAT ARE LAUNCHED IN A SINGLE KERNEL INVOCATION
	//CAN BE FEWER THAN THE NUMBER OF ELEMENTS IN THE DATABASE IF MORE THAN 1 BATCH
	unsigned int * N;
	N = (unsigned int*)malloc(sizeof(unsigned int) * GPUSTREAMS);

	unsigned int * dev_N;
	dev_N = (unsigned int*)malloc(sizeof(unsigned int) * GPUSTREAMS);

	//allocate on the device
	errCode = cudaMalloc((void**)&dev_N, sizeof(unsigned int) * GPUSTREAMS);
	if(errCode != cudaSuccess) {
		cout << "\nError: Alloc dev_N -- error with code " << errCode << endl;
	}

	////////////////////////////////////
	//NUMBER OF THREADS PER GPU STREAM
	////////////////////////////////////





	////////////////////////////////////
	//OFFSET INTO THE DATABASE FOR BATCHING THE RESULTS
	//BATCH NUMBER
	////////////////////////////////////
	unsigned int * batchOffset;
	batchOffset = (unsigned int*)malloc(sizeof(unsigned int) * GPUSTREAMS);

	unsigned int * dev_offset;
	dev_offset = (unsigned int*)malloc(sizeof(unsigned int) * GPUSTREAMS);

	//allocate on the device
	errCode = cudaMalloc((void**)&dev_offset, sizeof(unsigned int) * GPUSTREAMS);
	if(errCode != cudaSuccess) {
		cout << "\nError: Alloc offset -- error with code " << errCode << endl;
	}

	//Batch number to calculate the point to process (in conjunction with the offset)
	//offset into the database when batching the results
	unsigned int * batchNumber;
	batchNumber = (unsigned int*)malloc(sizeof(unsigned int) * GPUSTREAMS);

	unsigned int * dev_batchNumber;
	dev_batchNumber = (unsigned int*)malloc(sizeof(unsigned int) * GPUSTREAMS);

	//allocate on the device
	errCode = cudaMalloc((void**)&dev_batchNumber, sizeof(unsigned int) * GPUSTREAMS);
	if(errCode != cudaSuccess) {
		cout << "\nError: Alloc batch number -- error with code " << errCode << endl;
	}

	////////////////////////////////////
	//END OFFSET INTO THE DATABASE FOR BATCHING THE RESULTS
	//BATCH NUMBER
	////////////////////////////////////





    #if SORT_BY_WORKLOAD

        DTYPE* sortedCells = (DTYPE*)malloc(sizeof(DTYPE) * (*DBSIZE) * (GPUNUMDIM));
        DTYPE* dev_sortedCells = (DTYPE*)malloc(sizeof(DTYPE) * (*DBSIZE) * (GPUNUMDIM));

        //allocate memory on device:
        errCode = cudaMalloc( (void**)&dev_sortedCells, sizeof(DTYPE) * (GPUNUMDIM) * (*DBSIZE));
        if(errCode != cudaSuccess) {
            cout << "\nError: sortedSet alloc -- error with code " << errCode << endl;
            cout.flush();
        }

        struct schedulingCell* sortedCellsTmp = (struct schedulingCell*)malloc(sizeof(struct schedulingCell) * (*nNonEmptyCells));
        struct schedulingCell* dev_sortedCellsTmp = (struct schedulingCell*)malloc(sizeof(struct schedulingCell) * (*nNonEmptyCells));
        errCode = cudaMalloc( (void**)&dev_sortedCellsTmp, sizeof(struct schedulingCell) * (*nNonEmptyCells));
        if(errCode != cudaSuccess) {
            cout << "\nError: sortedSet alloc -- error with code " << errCode << endl;
            cout.flush();
        }

        double tStartSortingCells = omp_get_wtime();

        // schedulePointsByWork(searchMode, database, epsilon, index, indexLookupArr, gridCellLookupArr,
                // minArr, nCells, nNonEmptyCells, gridCellNDMask, gridCellNDMaskOffsets, sortedCells);
        int nbBlock = ((*nNonEmptyCells) / 1024) + 1;
        if(6 == searchMode || 10 == searchMode)
        {
            sortByWorkLoadLidUnicomp<<<nbBlock, 1024>>>(dev_database, dev_epsilon, dev_grid, dev_indexLookupArr, dev_gridCellLookupArr,
                    dev_minArr, dev_nCells, dev_nNonEmptyCells, dev_gridCellNDMask, dev_gridCellNDMaskOffsets, dev_sortedCellsTmp, dev_sortedCells);
        }else{
            sortByWorkLoad<<<nbBlock, 1024>>>(dev_database, dev_epsilon, dev_grid, dev_indexLookupArr, dev_gridCellLookupArr,
                    dev_minArr, dev_nCells, dev_nNonEmptyCells, dev_gridCellNDMask, dev_gridCellNDMaskOffsets, dev_sortedCellsTmp, dev_sortedCells);
        }
        cudaDeviceSynchronize();

        cudaMemcpy(sortedCellsTmp, dev_sortedCellsTmp, sizeof(struct schedulingCell) * (*nNonEmptyCells), cudaMemcpyDeviceToHost);

        sort(sortedCellsTmp, sortedCellsTmp + (*nNonEmptyCells),
                [](const schedulingCell& a, const schedulingCell& b){ return a.nbPoints > b.nbPoints; });

        unsigned int * originPointIndex = (unsigned int *)malloc((*DBSIZE) * sizeof(unsigned int));
        unsigned int * dev_originPointIndex;
        cudaMalloc( (void**)&dev_originPointIndex, (*DBSIZE) * sizeof(unsigned int));

        int prec = 0;
        for(int i = 0; i < (*nNonEmptyCells); i++)
        {
            int cellId = sortedCellsTmp[i].cellId;
            int nbNeighbor = index[cellId].indexmax - index[cellId].indexmin + 1;
            for(int j = 0; j < nbNeighbor; j++)
            {
                int tmpId = indexLookupArr[ index[cellId].indexmin + j ];
                //cout << tmpId << endl;
                for(int n = 0; n < NUMINDEXEDDIM; n++)
                {
                    sortedCells[(prec + j) * NUMINDEXEDDIM + n] = database[tmpId * NUMINDEXEDDIM + n];
                }
                originPointIndex[tmpId] = tmpId;
            }
            prec += nbNeighbor;
        }

        cudaMemcpy(dev_originPointIndex, originPointIndex, (*DBSIZE) * sizeof(unsigned int), cudaMemcpyHostToDevice);

        double tEndSortingCells = omp_get_wtime();

        printf("\nTime sorting cells by workload: %f\n", tEndSortingCells - tStartSortingCells);

        errCode = cudaMemcpy(dev_sortedCells, sortedCells, (*DBSIZE) * sizeof(DTYPE) * GPUNUMDIM, cudaMemcpyHostToDevice);
        if(errCode != cudaSuccess) {
            cout << "\nError: sortedSet copy -- error with code " << errCode << endl;
            cout.flush();
        }

    #endif





    /////////////////////////////////////////////////////////
	//BEGIN BATCH ESTIMATOR
	/////////////////////////////////////////////////////////

	unsigned long long estimatedNeighbors = 0;
	unsigned int numBatches = 0;
	unsigned int GPUBufferSize = 0;

	double tstartbatchest = omp_get_wtime();
    if(9 == searchMode || 10 == searchMode)
    {
        #if SORT_BY_WORKLOAD
        estimatedNeighbors = callGPUBatchEstWorkQueue(DBSIZE, dev_database, dev_sortedCells, dev_epsilon, dev_grid, dev_indexLookupArr, dev_gridCellLookupArr,
                dev_minArr, dev_nCells, dev_nNonEmptyCells, dev_gridCellNDMask, dev_gridCellNDMaskOffsets, &numBatches, &GPUBufferSize);
        #endif
    }else{
	    estimatedNeighbors = callGPUBatchEst(DBSIZE, dev_database, dev_epsilon, dev_grid, dev_indexLookupArr, dev_gridCellLookupArr,
                dev_minArr, dev_nCells, dev_nNonEmptyCells, dev_gridCellNDMask, dev_gridCellNDMaskOffsets, &numBatches, &GPUBufferSize);
    }
	double tendbatchest = omp_get_wtime();
	printf("\nTime to estimate batches: %f", tendbatchest - tstartbatchest);
	printf("\nIn Calling fn: Estimated neighbors: %llu, num. batches: %d, GPU Buffer size: %d", estimatedNeighbors, numBatches, GPUBufferSize);

	/////////////////////////////////////////////////////////
	//END BATCH ESTIMATOR
	/////////////////////////////////////////////////////////





	////////////////////////////////////
	//TWO DEBUG VALUES SENT TO THE GPU FOR GOOD MEASURE
	////////////////////////////////////

	//debug values
	unsigned int * dev_debug1;
	dev_debug1 = (unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug1 = 0;

	unsigned int * dev_debug2;
	dev_debug2 = (unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug2 = 0;

	unsigned int * debug1;
	debug1 = (unsigned int *)malloc(sizeof(unsigned int ));
	*debug1 = 0;

	unsigned int * debug2;
	debug2 = (unsigned int *)malloc(sizeof(unsigned int ));
	*debug2 = 0;

	//allocate on the device
	errCode = cudaMalloc( (unsigned int **)&dev_debug1, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
		cout << "\nError: debug1 alloc -- error with code " << errCode << endl;
	}

	errCode = cudaMalloc( (unsigned int **)&dev_debug2, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
		cout << "\nError: debug2 alloc -- error with code " << errCode << endl;
	}

	//copy debug to device
	errCode = cudaMemcpy( dev_debug1, debug1, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
		cout << "\nError: dev_debug1 copy to device -- error with code " << errCode << endl;
	}

	errCode = cudaMemcpy( dev_debug2, debug2, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
		cout << "\nError: dev_debug2 copy to device -- with code " << errCode << endl;
	}

	////////////////////////////////////
	//END TWO DEBUG VALUES SENT TO THE GPU FOR GOOD MEASURE
	////////////////////////////////////



	///////////////////
	//ALLOCATE POINTERS TO INTEGER ARRAYS FOR THE VALUES FOR THE NEIGHBORTABLES
	///////////////////

	//THE NUMBER OF POINTERS IS EQUAL TO THE NUMBER OF BATCHES
	for (int i = 0; i < numBatches; i++)
    {
		int *ptr;
		struct neighborDataPtrs tmpStruct;
		tmpStruct.dataPtr = ptr;
		tmpStruct.sizeOfDataArr = 0;

		pointersToNeighbors->push_back(tmpStruct);
	}

	///////////////////
	//END ALLOCATE POINTERS TO INTEGER ARRAYS FOR THE VALUES FOR THE NEIGHBORTABLES
	///////////////////



	///////////////////////////////////
	//ALLOCATE MEMORY FOR THE RESULT SET USING THE BATCH ESTIMATOR
	///////////////////////////////////

	//NEED BUFFERS ON THE GPU AND THE HOST FOR THE NUMBER OF CONCURRENT STREAMS
	//GPU BUFFER ON THE DEVICE
	//BUFFER ON THE HOST WITH PINNED MEMORY FOR FAST MEMCPY
	//BUFFER ON THE HOST TO DUMP THE RESULTS OF BATCHES SO THAT GPU THREADS CAN CONTINUE
	//EXECUTING STREAMS ON THE HOST

	//GPU MEMORY ALLOCATION: key/value pairs

	int * dev_pointIDKey[GPUSTREAMS]; //key
	int * dev_pointInDistValue[GPUSTREAMS]; //value
    // GPUBufferSize = 100000000;
    // GPUBufferSize = 150000000;
	for (int i = 0; i < GPUSTREAMS; i++)
	{
		errCode = cudaMalloc((void **)&dev_pointIDKey[i], 2 * sizeof(int) * GPUBufferSize);
		if(errCode != cudaSuccess) {
			cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
		}

		errCode = cudaMalloc((void **)&dev_pointInDistValue[i], 2 * sizeof(int) * GPUBufferSize);
		if(errCode != cudaSuccess) {
			cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
		}

	}
    // printf("\nAllocating pointIDKey and pointInDistValue on the GPU, size = %d", 2 * sizeof(int) * GPUBufferSize);

	//HOST RESULT ALLOCATION FOR THE GPU TO COPY THE DATA INTO A PINNED MEMORY ALLOCATION
	//ON THE HOST
	//pinned result set memory for the host
	//the number of elements are recorded for that batch in resultElemCountPerBatch
	//NEED PINNED MEMORY ALSO BECAUSE YOU NEED IT TO USE STREAMS IN THRUST FOR THE MEMCOPY OF THE SORTED RESULTS
	//can't do async copies without pinned memory

	//PINNED MEMORY TO COPY FROM THE GPU
	int * pointIDKey[GPUSTREAMS]; //key
	int * pointInDistValue[GPUSTREAMS]; //value

	double tstartpinnedresults = omp_get_wtime();

    #pragma omp parallel for num_threads(GPUSTREAMS)
	for (int i = 0; i < GPUSTREAMS; i++)
	{
		cudaMallocHost((void **) &pointIDKey[i], 2 * sizeof(int) * GPUBufferSize);
		cudaMallocHost((void **) &pointInDistValue[i], 2 * sizeof(int) * GPUBufferSize);
	}

	double tendpinnedresults = omp_get_wtime();
	printf("\nTime to allocate pinned memory for results: %f", tendpinnedresults - tstartpinnedresults);

	// cudaMalloc((void **) &pointIDKey, sizeof(int)*GPUBufferSize*NUMBATCHES);
	// cudaMalloc((void **) &pointInDistValue, sizeof(int)*GPUBufferSize*NUMBATCHES);

	printf("\nmemory requested for results ON GPU (GiB): %f",
            (double)(sizeof(int) * 2 * GPUBufferSize * GPUSTREAMS) / (1024 * 1024 * 1024));
	printf("\nmemory requested for results in MAIN MEMORY (GiB): %f",
            (double)(sizeof(int) * 2 * GPUBufferSize * GPUSTREAMS) / (1024 * 1024 * 1024));

	///////////////////////////////////
	//END ALLOCATE MEMORY FOR THE RESULT SET
	///////////////////////////////////




    unsigned int * elementToWork = 0;
    unsigned int * dev_elementToWork = 0;
    cudaMalloc(&dev_elementToWork, sizeof(unsigned int));
    cudaMemcpy(dev_elementToWork, elementToWork, sizeof(unsigned int), cudaMemcpyHostToDevice);

    /*
    unsigned int * elementToWork;
	elementToWork = (unsigned int*)malloc(sizeof(unsigned int));
	*elementToWork = 0;

	unsigned int * dev_elementToWork;
	dev_elementToWork = (unsigned int*)malloc(sizeof(unsigned int));
	*dev_elementToWork = 0;

	//allocate on the device
	errCode = cudaMalloc((void**)&dev_elementToWork, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
		cout << "\nError: Alloc elementToWork -- error with code " << errCode << endl;
	}
    */






	/////////////////////////////////
	//SET OPENMP ENVIRONMENT VARIABLES
	////////////////////////////////

	omp_set_num_threads(GPUSTREAMS);

	/////////////////////////////////
	//END SET OPENMP ENVIRONMENT VARIABLES
	////////////////////////////////



	/////////////////////////////////
	//CREATE STREAMS
	////////////////////////////////

	cudaStream_t stream[GPUSTREAMS];

	for (int i = 0; i < GPUSTREAMS; i++){
		cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
	}

	/////////////////////////////////
	//END CREATE STREAMS
	////////////////////////////////

    // DTYPE * sortedCellsDynamicThreads;
    // DTYPE * dev_sortedCellsDynamicThreads;
    //unsigned int * sortedCellsNbThreads;
    //unsigned int * dev_sortedCellsNbThreads;
    //unsigned int * sortedCellsNbThreadsBefore;
    //unsigned int * dev_sortedCellsNbThreadsBefore;
    //unsigned int threadOverhead;
    // unsigned int totalNbThreads;
    // unsigned int nbPointsThird1;
    // unsigned int nbPointsThird2;

    //unsigned int * threadArray;
    //unsigned int * dev_threadArray;

    // std::vector<unsigned int>* tmpVectors;
    // unsigned int ** threadArray;
    // unsigned int ** dev_threadArray;

    //DTYPE * sortedCellsDynamicThreadsDouble;
    //DTYPE * dev_sortedCellsDynamicThreadsDouble;

    /*
    if(8 == searchMode)
    {
        //std::vector<DTYPE> sortedCellsDynamicThreadsVector((*DBSIZE) * GPUNUMDIM);

        totalNbThreads = 0;
        nbPointsThird1 = 0;
        nbPointsThird2 = 0;

        tmpVectors = new std::vector<unsigned int>[numBatches];

        double tStartSortingCells = omp_get_wtime();

        //threadArray = (unsigned int *)malloc(sizeof(unsigned int) * (*N));
        //dev_threadArray = (unsigned int *)malloc(sizeof(unsigned int) * (*N));

        std::vector<DTYPE> sortedCellsDynamicThreadsVector = schedulePointsByWorkDynamicThreadsPerPointFixed(searchMode, *N, database,
                epsilon, index, indexLookupArr, gridCellLookupArr, minArr, nCells, nNonEmptyCells, gridCellNDMask,
                gridCellNDMaskOffsets, &totalNbThreads, (*DBSIZE), numBatches, &tmpVectors);

        double tEndSortingCells = omp_get_wtime();

        cout << "0-0" << tmpVectors[0][0] << endl;
        cout << "0-1" << tmpVectors[0][1] << endl;
        cout << "0-2" << tmpVectors[0][2] << endl;
        cout << "0-3" << tmpVectors[0][3] << endl;
        cout << "0-4" << tmpVectors[0][4] << endl;


        printf("\nTime sorting cells by workload (dynamic threads per point): %f", tEndSortingCells - tStartSortingCells);

        int size = sortedCellsDynamicThreadsVector.size();
        //totalNbThreads = size;

        sortedCellsDynamicThreads = (DTYPE*)malloc(sizeof(DTYPE) * size);
        dev_sortedCellsDynamicThreads = (DTYPE*)malloc(sizeof(DTYPE) * size);

        printf("\nnbPointsThird1 = %d, nbPointsThird2 = %d, totalNbThreads = %d\n",
                nbPointsThird1, nbPointsThird2, totalNbThreads);

        for(int i = 0; i < size; i++)
        {
            sortedCellsDynamicThreads[i] = sortedCellsDynamicThreadsVector[i];
        }

        //allocate memory on device:
    	errCode = cudaMalloc( (void**)&dev_sortedCellsDynamicThreads, sizeof(DTYPE) * size);
    	if(errCode != cudaSuccess) {
    		cout << "\nError: sortedSet alloc -- error with code " << errCode << endl;
            cout.flush();
    	}
        errCode = cudaMemcpy(dev_sortedCellsDynamicThreads, sortedCellsDynamicThreads, sizeof(DTYPE) * size, cudaMemcpyHostToDevice);
        if(errCode != cudaSuccess) {
    		cout << "\nError: sortedSet copy -- error with code " << errCode << endl;
            cout.flush();
    	}

        //delete sortedCellsDynamicThreadsVector;
        sortedCellsDynamicThreadsVector.clear();

        threadArray = new unsigned int*[numBatches];
        dev_threadArray = new unsigned int*[numBatches];
        for(int i = 0; i < numBatches; i++)
        {
            int size = tmpVectors[i].size();
            threadArray[i] = new unsigned int[size];
            dev_threadArray[i] = new unsigned int[size];
            for(int j = 0; j < size; j++)
            {
                threadArray[i][j] = tmpVectors[i][j];
            }

            errCode = cudaMalloc( (void**)&dev_threadArray[i], sizeof(unsigned int) * size);
        	if(errCode != cudaSuccess) {
        		cout << "\nError: threadArray alloc -- error with code " << errCode << endl;
                cout.flush();
        	}
            errCode = cudaMemcpy(dev_threadArray[i], threadArray[i], sizeof(unsigned int) * size, cudaMemcpyHostToDevice);
            if(errCode != cudaSuccess) {
        		cout << "\nError: threadArray copy -- error with code " << errCode << endl;
                cout.flush();
        	}
        }




    }
    */





	///////////////////////////////////
	//LAUNCH KERNEL IN BATCHES
	///////////////////////////////////

	//since we use the strided scheme, some of the batch sizes
	//are off by 1 of each other, a first group of batches will
	//have 1 extra data point to process, and we calculate which batch numbers will
	//have that.  The batchSize is the lower value (+1 is added to the first ones)

	unsigned int batchSize = (*DBSIZE) / numBatches;
	unsigned int batchesThatHaveOneMore = (*DBSIZE) - (batchSize * numBatches); //batch number 0- < this value have one more
	printf("\nBatches that have one more GPU thread: %u batchSize(N): %u, \n", batchesThatHaveOneMore, batchSize);

	uint64_t totalResultsLoop = 0;

    double kernelTime[3];
    kernelTime[0] = 0.0;
    kernelTime[1] = 0.0;
    kernelTime[2] = 0.0;

    //unsigned int elementToWork = 0; // for the work queue

	//FOR LOOP OVER THE NUMBER OF BATCHES STARTS HERE
	//i=0...numBatches
#pragma omp parallel for schedule(static,1) reduction(+:totalResultsLoop) num_threads(GPUSTREAMS)
	// for(int i = 0; i < 3; i++)
	for (int i = 0; i < numBatches; i++)
	{

		int tid = omp_get_thread_num();

		printf("\ntid: %d, starting iteration: %d",tid,i);

		//N NOW BECOMES THE NUMBER OF POINTS TO PROCESS PER BATCH
		//AS ONE GPU THREAD PROCESSES A SINGLE POINT

		if (i < batchesThatHaveOneMore)
		{
			N[tid] = batchSize + 1;
			printf("\nN (GPU threads): %d, tid: %d", N[tid], tid);
		}
		else
		{
			N[tid] = batchSize;
			printf("\nN (1 less): %d tid: %d", N[tid], tid);
		}

		//set relevant parameters for the batched execution that get reset

		//copy N to device
		//N IS THE NUMBER OF THREADS
		errCode = cudaMemcpyAsync( &dev_N[tid], &N[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
		if(errCode != cudaSuccess) {
			cout << "\nError: N Got error with code " << errCode << endl;
		}

		//the batched result set size (reset to 0):
		cnt[tid] = 0;
		errCode = cudaMemcpyAsync( &dev_cnt[tid], &cnt[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
		if(errCode != cudaSuccess) {
			cout << "\nError: dev_cnt memcpy Got error with code " << errCode << endl;
		}

		//the offset for batching, which keeps track of where to start processing at each batch
		batchOffset[tid] = numBatches; //for the strided
		errCode = cudaMemcpyAsync( &dev_offset[tid], &batchOffset[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
		if(errCode != cudaSuccess) {
			cout << "\nError: dev_offset memcpy Got error with code " << errCode << endl;
		}

		//the batch number for batching with strided
		batchNumber[tid] = i;
		errCode = cudaMemcpyAsync( &dev_batchNumber[tid], &batchNumber[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
		if(errCode != cudaSuccess) {
			cout << "\nError: dev_batchNumber memcpy Got error with code " << errCode << endl;
		}

		const int TOTALBLOCKS = ceil( (1.0 * (N[tid])) / (1.0 * BLOCKSIZE) );
		printf("\ntotal blocks: %d", TOTALBLOCKS);

		//execute kernel
		//0 is shared memory pool

		double beginKernel = omp_get_wtime();

        switch(searchMode)
        {
            // Original global memory kernel
            case 3:
                #if SORT_BY_WORKLOAD
                    kernelNDGridIndexGlobal<<< THREADPERPOINT * TOTALBLOCKS, BLOCKSIZE, 0, stream[tid] >>>(dev_debug1, dev_debug2, &dev_N[tid],
                            &dev_offset[tid], &dev_batchNumber[tid], dev_database, dev_sortedCells, dev_epsilon, dev_grid, dev_indexLookupArr,
                            dev_gridCellLookupArr, dev_minArr, dev_nCells, &dev_cnt[tid], dev_nNonEmptyCells, dev_gridCellNDMask,
                            dev_gridCellNDMaskOffsets, dev_pointIDKey[tid], dev_pointInDistValue[tid]);
                #else
                    kernelNDGridIndexGlobal<<< THREADPERPOINT * TOTALBLOCKS, BLOCKSIZE, 0, stream[tid] >>>(dev_debug1, dev_debug2, &dev_N[tid],
                            &dev_offset[tid], &dev_batchNumber[tid], dev_database, NULL, dev_epsilon, dev_grid, dev_indexLookupArr,
                            dev_gridCellLookupArr, dev_minArr, dev_nCells, &dev_cnt[tid], dev_nNonEmptyCells, dev_gridCellNDMask,
                            dev_gridCellNDMaskOffsets, dev_pointIDKey[tid], dev_pointInDistValue[tid]);
                #endif
                break;
            // Unicomp
            case 4:
                #if SORT_BY_WORKLOAD
                    kernelNDGridIndexGlobalUnicomp<<< THREADPERPOINT * TOTALBLOCKS, BLOCKSIZE, 0, stream[tid] >>>(dev_debug1, dev_debug2, &dev_N[tid],
                            &dev_offset[tid], &dev_batchNumber[tid], dev_database, dev_sortedCells, dev_epsilon, dev_grid, dev_indexLookupArr,
                            dev_gridCellLookupArr, dev_minArr, dev_nCells, &dev_cnt[tid], dev_nNonEmptyCells, dev_gridCellNDMask,
                            dev_gridCellNDMaskOffsets, dev_pointIDKey[tid], dev_pointInDistValue[tid]);
                #else
                    kernelNDGridIndexGlobalUnicomp<<< THREADPERPOINT * TOTALBLOCKS, BLOCKSIZE, 0, stream[tid] >>>(dev_debug1, dev_debug2, &dev_N[tid],
                            &dev_offset[tid], &dev_batchNumber[tid], dev_database, NULL, dev_epsilon, dev_grid, dev_indexLookupArr,
                            dev_gridCellLookupArr, dev_minArr, dev_nCells, &dev_cnt[tid], dev_nNonEmptyCells, dev_gridCellNDMask,
                            dev_gridCellNDMaskOffsets, dev_pointIDKey[tid], dev_pointInDistValue[tid]);
                #endif
                break;
            // B-Unicomp
            case 5:
                #if SORT_BY_WORKLOAD
                    kernelNDGridIndexGlobalBUnicomp<<< THREADPERPOINT * TOTALBLOCKS, BLOCKSIZE, 0, stream[tid] >>>(dev_debug1, dev_debug2, &dev_N[tid],
                            &dev_offset[tid], &dev_batchNumber[tid], dev_database, dev_sortedCells, dev_epsilon, dev_grid, dev_indexLookupArr,
                            dev_gridCellLookupArr, dev_minArr, dev_nCells, &dev_cnt[tid], dev_nNonEmptyCells, dev_gridCellNDMask,
                            dev_gridCellNDMaskOffsets, dev_pointIDKey[tid], dev_pointInDistValue[tid]);
                #else
                    kernelNDGridIndexGlobalBUnicomp<<< THREADPERPOINT * TOTALBLOCKS, BLOCKSIZE, 0, stream[tid] >>>(dev_debug1, dev_debug2, &dev_N[tid],
                            &dev_offset[tid], &dev_batchNumber[tid], dev_database, NULL, dev_epsilon, dev_grid, dev_indexLookupArr,
                            dev_gridCellLookupArr, dev_minArr, dev_nCells, &dev_cnt[tid], dev_nNonEmptyCells, dev_gridCellNDMask,
                            dev_gridCellNDMaskOffsets, dev_pointIDKey[tid], dev_pointInDistValue[tid]);
                #endif
                break;
            // Linear ID Unicomp
            case 6:
                #if SORT_BY_WORKLOAD
                    kernelNDGridIndexGlobalLinearIDUnicomp<<< THREADPERPOINT * TOTALBLOCKS, BLOCKSIZE, 0, stream[tid] >>>(dev_debug1, dev_debug2, &dev_N[tid],
                            &dev_offset[tid], &dev_batchNumber[tid], dev_database, dev_sortedCells, dev_epsilon, dev_grid, dev_indexLookupArr,
                            dev_gridCellLookupArr, dev_minArr, dev_nCells, &dev_cnt[tid], dev_nNonEmptyCells, dev_gridCellNDMask,
                            dev_gridCellNDMaskOffsets, dev_pointIDKey[tid], dev_pointInDistValue[tid]);
                #else
                    kernelNDGridIndexGlobalLinearIDUnicomp<<< THREADPERPOINT * TOTALBLOCKS, BLOCKSIZE, 0, stream[tid] >>>(dev_debug1, dev_debug2, &dev_N[tid],
                            &dev_offset[tid], &dev_batchNumber[tid], dev_database, NULL, dev_epsilon, dev_grid, dev_indexLookupArr,
                            dev_gridCellLookupArr, dev_minArr, dev_nCells, &dev_cnt[tid], dev_nNonEmptyCells, dev_gridCellNDMask,
                            dev_gridCellNDMaskOffsets, dev_pointIDKey[tid], dev_pointInDistValue[tid]);
                #endif
                break;
            // Original global memory kernel + sorting cells
            case 7:
                #if SORT_BY_WORKLOAD
                    kernelNDGridIndexGlobalSortedCells<<< THREADPERPOINT * TOTALBLOCKS, BLOCKSIZE, 0, stream[tid] >>>(dev_debug1, dev_debug2, &dev_N[tid],
                            &dev_offset[tid], &dev_batchNumber[tid], dev_database, dev_sortedCells, dev_epsilon, dev_grid, dev_indexLookupArr,
                            dev_gridCellLookupArr, dev_minArr, dev_nCells, &dev_cnt[tid], dev_nNonEmptyCells, dev_gridCellNDMask,
                            dev_gridCellNDMaskOffsets, dev_pointIDKey[tid], dev_pointInDistValue[tid]);
                #endif
                break;
            // dynamic threads per point
            case 8:{
                /*
                unsigned int totalBlocks = ((totalNbThreads / BLOCKSIZE) / numBatches) + 1;
                kernelNDGridIndexGlobalSortedCellsDynamicThreadsFixed<<< totalBlocks, BLOCKSIZE, 0, stream[tid] >>>(dev_debug1, dev_debug2, &dev_N[tid],
                        &dev_offset[tid], &dev_batchNumber[tid], dev_database, dev_sortedCellsDynamicThreads, dev_threadArray[tid],
                        totalNbThreads, dev_epsilon, dev_grid, dev_indexLookupArr, dev_gridCellLookupArr, dev_minArr, dev_nCells, &dev_cnt[tid],
                        dev_nNonEmptyCells, dev_gridCellNDMask, dev_gridCellNDMaskOffsets, dev_pointIDKey[tid], dev_pointInDistValue[tid]);
                break;
                */
                #if SORT_BY_WORKLOAD
                    if(i < (numBatches / 3))
                    {
                        printf("Batch %d, using %d threads per point\n", i, NB_THREAD_THIRD1);
                        kernelNDGridIndexGlobalSortedCellsDynamicThreadsV3<<< NB_THREAD_THIRD1 * TOTALBLOCKS, BLOCKSIZE, 0, stream[tid] >>>(dev_debug1, dev_debug2, &dev_N[tid],
                                &dev_offset[tid], &dev_batchNumber[tid], dev_database, dev_sortedCells, dev_epsilon, dev_grid, dev_indexLookupArr,
                                dev_gridCellLookupArr, dev_minArr, dev_nCells, &dev_cnt[tid], dev_nNonEmptyCells, dev_gridCellNDMask,
                                dev_gridCellNDMaskOffsets, dev_pointIDKey[tid], dev_pointInDistValue[tid], NB_THREAD_THIRD1);
                    }else{
                        if(i < (2 * (numBatches / 3)))
                        {
                            printf("Batch %d, using %d threads per point\n", i, NB_THREAD_THIRD2);
                            kernelNDGridIndexGlobalSortedCellsDynamicThreadsV3<<< NB_THREAD_THIRD2 * TOTALBLOCKS, BLOCKSIZE, 0, stream[tid] >>>(dev_debug1, dev_debug2, &dev_N[tid],
                                    &dev_offset[tid], &dev_batchNumber[tid], dev_database, dev_sortedCells, dev_epsilon, dev_grid, dev_indexLookupArr,
                                    dev_gridCellLookupArr, dev_minArr, dev_nCells, &dev_cnt[tid], dev_nNonEmptyCells, dev_gridCellNDMask,
                                    dev_gridCellNDMaskOffsets, dev_pointIDKey[tid], dev_pointInDistValue[tid], NB_THREAD_THIRD2);
                        }else{
                            printf("Batch %d, using %d threads per point\n", i, NB_THREAD_THIRD3);
                            kernelNDGridIndexGlobalSortedCellsDynamicThreadsV3<<< NB_THREAD_THIRD3 * TOTALBLOCKS, BLOCKSIZE, 0, stream[tid] >>>(dev_debug1, dev_debug2, &dev_N[tid],
                                    &dev_offset[tid], &dev_batchNumber[tid], dev_database, dev_sortedCells, dev_epsilon, dev_grid, dev_indexLookupArr,
                                    dev_gridCellLookupArr, dev_minArr, dev_nCells, &dev_cnt[tid], dev_nNonEmptyCells, dev_gridCellNDMask,
                                    dev_gridCellNDMaskOffsets, dev_pointIDKey[tid], dev_pointInDistValue[tid], NB_THREAD_THIRD3);
                        }
                    }
                #endif
                break;
            }
            // work queue
            case 9:
                #if SORT_BY_WORKLOAD
                    kernelNDGridIndexGlobalWorkQueue<<< THREADPERPOINT * TOTALBLOCKS, BLOCKSIZE, 0, stream[tid] >>>(dev_debug1, dev_debug2, &dev_N[tid],
                            &dev_offset[tid], &dev_batchNumber[tid], dev_database, dev_sortedCells, dev_originPointIndex, dev_epsilon, dev_grid, dev_indexLookupArr,
                            dev_gridCellLookupArr, dev_minArr, dev_nCells, &dev_cnt[tid], dev_nNonEmptyCells, dev_gridCellNDMask,
                            dev_gridCellNDMaskOffsets, dev_pointIDKey[tid], dev_pointInDistValue[tid], dev_elementToWork, (*DBSIZE));
                #endif
                break;
            case 10:
                #if SORT_BY_WORKLOAD
                    kernelNDGridIndexGlobalWorkQueueLidUnicomp<<< THREADPERPOINT * TOTALBLOCKS, BLOCKSIZE, 0, stream[tid] >>>(dev_debug1, dev_debug2, &dev_N[tid],
                            &dev_offset[tid], &dev_batchNumber[tid], dev_database, dev_sortedCells, dev_epsilon, dev_grid, dev_indexLookupArr,
                            dev_gridCellLookupArr, dev_minArr, dev_nCells, &dev_cnt[tid], dev_nNonEmptyCells, dev_gridCellNDMask,
                            dev_gridCellNDMaskOffsets, dev_pointIDKey[tid], dev_pointInDistValue[tid], dev_elementToWork, (*DBSIZE));
                #endif
                break;
        }

		// errCode=cudaDeviceSynchronize();
		// cout <<"\n\nError from device synchronize: "<<errCode;

        errCode = cudaGetLastError();
		cout << "\n\nKERNEL LAUNCH RETURN: " << errCode << endl;
        printf("%s: %s\n\n", cudaGetErrorName(errCode), cudaGetErrorString(errCode));
        // printf("Details: %s\n\n", cudaGetErrorString(errCode));
		if ( cudaSuccess != cudaGetLastError() ){
			cout << "\n\nERROR IN KERNEL LAUNCH. ERROR: " << cudaSuccess << endl << endl;
            //printf("\nDetails: %s", cudaGetErrorString(cudaSuccess));
		}

		// find the size of the number of results
		errCode = cudaMemcpyAsync( &cnt[tid], &dev_cnt[tid], sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[tid] );
		if(errCode != cudaSuccess) {
			cout << "\nError: getting cnt from GPU Got error with code " << errCode << endl;
            cout << cudaGetErrorName(errCode) << ": " << cudaGetErrorString(errCode) << endl;
		}
		else{
			printf("\nGPU: result set size within epsilon (GPU grid): %d", cnt[tid]);
		}

		double endKernel = omp_get_wtime();
        // kernelTime[tid] += endKernel - beginKernel;
		//cout << "Single kernel execution time = " << endKernel - beginKernel << " ms" << endl;

		//printf("\ncnt = %d\n", cnt);

		////////////////////////////////////
		//SORT THE TABLE DATA ON THE GPU
		//THERE IS NO ORDERING BETWEEN EACH POINT AND THE ONES THAT IT'S WITHIN THE DISTANCE OF
		////////////////////////////////////

		/////////////////////////////
		//ONE PROBLEM WITH NOT TRANSFERING THE RESULT OFF OF THE DEVICE IS THAT
		//YOU CAN'T RESIZE THE RESULTS TO BE THE SIZE OF *CNT
		//SO THEN YOU HAVE POTENTIALLY LOTS OF WASTED SPACE
		/////////////////////////////

		//sort by key with the data already on the device:
		//wrap raw pointer with a device_ptr to use with Thrust functions
		thrust::device_ptr<int> dev_keys_ptr(dev_pointIDKey[tid]);
		thrust::device_ptr<int> dev_data_ptr(dev_pointInDistValue[tid]);

        // printf("\nThrust pointers to keys and data");

		//XXXXXXXXXXXXXXXX
		//THRUST USING STREAMS REQUIRES THRUST V1.8
		//XXXXXXXXXXXXXXXX

		try{
			thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), dev_keys_ptr, dev_keys_ptr + cnt[tid], dev_data_ptr);
		}
		catch(std::bad_alloc &e)
		{
			std::cerr << "Ran out of memory while sorting, " << GPUBufferSize << std::endl;
			exit(-1);
		}

        // printf("\nThrust sort by key");
        // printf("\nCopy size: %d", cnt[tid] * sizeof(int));

		//thrust with streams into individual buffers for each batch
		cudaMemcpyAsync(thrust::raw_pointer_cast(pointIDKey[tid]), thrust::raw_pointer_cast(dev_keys_ptr), cnt[tid] * sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);
		cudaMemcpyAsync(thrust::raw_pointer_cast(pointInDistValue[tid]), thrust::raw_pointer_cast(dev_data_ptr), cnt[tid] * sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);

        // printf("\nAsync memcpy of pointers");

		//need to make sure the data is copied before constructing portion of the neighbor table
		cudaStreamSynchronize(stream[tid]);

        // printf("\nStream synchronization");

		double tableconstuctstart = omp_get_wtime();
		//set the number of neighbors in the pointer struct:
		(*pointersToNeighbors)[i].sizeOfDataArr = cnt[tid];
		(*pointersToNeighbors)[i].dataPtr = new int[cnt[tid]];

		constructNeighborTableKeyValueWithPtrs(pointIDKey[tid], pointInDistValue[tid], neighborTable, (*pointersToNeighbors)[i].dataPtr, &cnt[tid]);

		//cout <<"\nIn make neighbortable. Data array ptr: "<<(*pointersToNeighbors)[i].dataPtr<<" , size of data array: "<<(*pointersToNeighbors)[i].sizeOfDataArr;cout.flush();

		double tableconstuctend = omp_get_wtime();

		printf("\nTable construct time: %f", tableconstuctend - tableconstuctstart);

		//add the batched result set size to the total count
		totalResultsLoop += cnt[tid];

		printf("\nRunning total of total size of result array, tid: %d: %lu", tid, totalResultsLoop);
		//}

	} //END LOOP OVER THE GPU BATCHES

	printf("\n\033[31;01m~~~TOTAL RESULT SET SIZE ON HOST:  %lu\033[00m", totalResultsLoop);
	*totalNeighbors = totalResultsLoop;

	double tKernelResultsEnd = omp_get_wtime();

	printf("\nTime to launch kernel and execute everything (get results etc.) except freeing memory: %f\n", tKernelResultsEnd - tKernelResultsStart);

    // if(9 == searchMode || 10 == searchMode)
    // {
        // neighborTableLookup * neighborTableTmp = new neighborTableLookup[NDdataPoints.size()];

        // for (int i = 0; i < NDdataPoints.size(); i++){
    	// //for (int i=0; i<10; i++){
    	//  	// sort to compare against CPU implementation
    	//  	std::sort(neighborTable[i].dataPtr + neighborTable[i].indexmin, neighborTable[i].dataPtr + neighborTable[i].indexmax + 1);
    	//  	printf("\npoint id: %d, neighbors: ", i);
    	//  	printf("nb neighbor %d\n", neighborTable[i].indexmax - neighborTable[i].indexmin + 1);
    	//  	for (int j = neighborTable[i].indexmin; j < neighborTable[i].indexmax; j++){
    	//  		printf("%d,", neighborTable[i].dataPtr[j]);
    	//  	}
        //
    	// }

        // int tmpCounter = 0;
        // neighborTableLookup * tmpTable = new neighborTableLookup;
        // neighborDataPtrs * tmpPointer = new neighborDataPtrs;
        // int tmpId;
        // int tmpMin;
        // int tmpMax;
        // int * tmpPtr;
        // for(int i = 0; i < (*nNonEmptyCells); i++)
        // {
            // cout << "Non empty cell " << i << endl;
        //     int min = index[sortedCellsTmp[i].cellId].indexmin;
        //     int max = index[sortedCellsTmp[i].cellId].indexmax;
        //     for(int j = min; j <= max; j++)
        //     {
        //         tmpId = neighborTable[tmpCounter].pointID;
        //         tmpMin = neighborTable[tmpCounter].indexmin;
        //         tmpMax = neighborTable[tmpCounter].indexmax;
        //         tmpPtr = neighborTable[tmpCounter].dataPtr;
        //
		// neighborTable[tmpCounter].pointID = neighborTable[j].pointID;
		// neighborTable[tmpCounter].indexmin = neighborTable[j].indexmin;
		// neighborTable[tmpCounter].indexmax = neighborTable[j].indexmax;
		// neighborTable[tmpCounter].dataPtr = neighborTable[j].dataPtr;
        //
		// neighborTable[j].pointID = tmpId;
		// neighborTable[j].indexmin = tmpMin;
		// neighborTable[j].indexmax = tmpMax;
		// neighborTable[j].dataPtr = tmpPtr;

                // cout << "Point " << j << endl;
                // (*tmpTable) = neighborTable[tmpCounter];
                // cout << "1" << endl;
                // neighborTable[tmpCounter] = neighborTable[j];
                // cout << "2" << endl;
                // neighborTable[j] = (*tmpTable);
                // cout << "3" << endl;

                // (*tmpPointer) = (*pointersToNeighbors)[tmpCounter];
                // cout << "4" << endl;
                // pointersToNeighbors[tmpCounter] = pointersToNeighbors[j];
                // cout << "5" << endl;
                // (*pointersToNeighbors)[j] = (*tmpPointer);
                // cout << "6" << endl;

    //             tmpCounter++;
    //         }
    //     }
    // }

    // kernelTime[0] /= numBatches;
    // kernelTime[1] /= numBatches;
    // kernelTime[2] /= numBatches;
    // printf("~~~ Single kernel invocation time: B0 = %f, B1 = %f, B2 = %f\n", kernelTime[0], kernelTime[1], kernelTime[2]);

	///////////////////////////////////
	//END GET RESULT SET
	///////////////////////////////////



	///////////////////////////////////
	//OPTIONAL DEBUG VALUES
	///////////////////////////////////

	// double tStartdebug=omp_get_wtime();

	// errCode=cudaMemcpy(debug1, dev_debug1, sizeof(unsigned int), cudaMemcpyDeviceToHost );

	// if(errCode != cudaSuccess) {
	// cout << "\nError: getting debug1 from GPU Got error with code " << errCode << endl;
	// }
	// else
	// {
	// 	printf("\nDebug1 value: %u",*debug1);
	// }

	// errCode=cudaMemcpy(debug2, dev_debug2, sizeof(unsigned int), cudaMemcpyDeviceToHost );

	// if(errCode != cudaSuccess) {
	// cout << "\nError: getting debug1 from GPU Got error with code " << errCode << endl;
	// }
	// else
	// {
	// 	printf("\nDebug2 value: %u",*debug2);
	// }

	// double tEnddebug=omp_get_wtime();
	// printf("\nTime to retrieve debug values: %f", tEnddebug - tStartdebug);


	///////////////////////////////////
	//END OPTIONAL DEBUG VALUES
	///////////////////////////////////


	///////////////////////////////////
	//FREE MEMORY FROM THE GPU
	///////////////////////////////////
	// if (NUM_TRIALS>1)
	// {

	double tFreeStart = omp_get_wtime();

	for (int i = 0; i < GPUSTREAMS; i++)
    {
		errCode = cudaStreamDestroy(stream[i]);
		if(errCode != cudaSuccess) {
			cout << "\nError: destroying stream" << errCode << endl;
		}
	}

	free(DBSIZE);
	free(database);
	free(totalResultSetCnt);
	free(cnt);
	free(N);
	free(batchOffset);
	free(batchNumber);
	free(debug1);
	free(debug2);

	//free the data on the device
	cudaFree(dev_pointIDKey);
	cudaFree(dev_pointInDistValue);

	cudaFree(dev_database);
	cudaFree(dev_debug1);
	cudaFree(dev_debug2);
	cudaFree(dev_epsilon);
	cudaFree(dev_grid);
	cudaFree(dev_gridCellLookupArr);
	cudaFree(dev_gridCellNDMask);
	cudaFree(dev_indexLookupArr);
	cudaFree(dev_minArr);
	cudaFree(dev_nCells);
	// cudaFree(dev_nNDMaskElems);
	cudaFree(dev_nNonEmptyCells);
	cudaFree(dev_N);
	cudaFree(dev_cnt);
	cudaFree(dev_offset);
	cudaFree(dev_batchNumber);


	//free data related to the individual streams for each batch
	for (int i = 0; i < GPUSTREAMS; i++)
    {
		//free the data on the device
		cudaFree(dev_pointIDKey[i]);
		cudaFree(dev_pointInDistValue[i]);

		//free on the host
		cudaFreeHost(pointIDKey[i]);
		cudaFreeHost(pointInDistValue[i]);
	}

	//cudaFree(dev_pointIDKey);
	//cudaFree(dev_pointInDistValue);
	//free pinned memory on host
	cudaFreeHost(pointIDKey);
	cudaFreeHost(pointInDistValue);

	double tFreeEnd = omp_get_wtime();

	printf("\nTime freeing memory: %f", tFreeEnd - tFreeStart);
	// }
	cout << "\n** last error at end of fn batches (could be from freeing memory): " << cudaGetLastError();

} // NDGridIndexGlobal


void warmUpGPU(){
	// initialize all ten integers of a device_vector to 1
	thrust::device_vector<int> D(10, 1);
	// set the first seven elements of a vector to 9
	thrust::fill(D.begin(), D.begin() + 7, 9);
	// initialize a host_vector with the first five elements of D
	thrust::host_vector<int> H(D.begin(), D.begin() + 5);
	// set the elements of H to 0, 1, 2, 3, ...
	thrust::sequence(H.begin(), H.end()); // copy all of H back to the beginning of D
	thrust::copy(H.begin(), H.end(), D.begin());
	// print D
	for(int i = 0; i < D.size(); i++)
		std::cout << " D[" << i << "] = " << D[i];


	return;
}






void constructNeighborTableKeyValueWithPtrs(int * pointIDKey, int * pointInDistValue, struct neighborTableLookup * neighborTable, int * pointersToNeighbors, unsigned int * cnt)
{


	//copy the value data:
	std::copy(pointInDistValue, pointInDistValue+(*cnt), pointersToNeighbors);



	//Step 1: find all of the unique keys and their positions in the key array
	unsigned int numUniqueKeys=0;

	std::vector<keyData> uniqueKeyData;

	keyData tmp;
	tmp.key=pointIDKey[0];
	tmp.position=0;
	uniqueKeyData.push_back(tmp);

	//we assign the ith data item when iterating over i+1th data item,
	//so we go 1 loop iteration beyond the number (*cnt)
	for (int i=1; i<(*cnt)+1; i++){
		if (pointIDKey[i-1]!=pointIDKey[i]){
			numUniqueKeys++;
			tmp.key=pointIDKey[i];
			tmp.position=i;
			uniqueKeyData.push_back(tmp);
		}
	}

	//insert into the neighbor table the values based on the positions of
	//the unique keys obtained above.
	for (int i=0; i<uniqueKeyData.size()-1; i++) {
		int keyElem=uniqueKeyData[i].key;
		neighborTable[keyElem].pointID=keyElem;
		neighborTable[keyElem].indexmin=uniqueKeyData[i].position;
		neighborTable[keyElem].indexmax=uniqueKeyData[i+1].position-1;

		//update the pointer to the data array for the values
		neighborTable[keyElem].dataPtr=pointersToNeighbors;
	}

}
