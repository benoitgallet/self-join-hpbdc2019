#include <cstdlib>
#include <stdio.h>
#include <random>
#include "omp.h"
#include <algorithm>
#include <string.h>
#include <fstream>
#include <iostream>
#include <string>
#include "GPU.h"
#include "kernel.h"
#include <math.h>
#include <queue>
#include <iomanip>
#include <set>
#include <algorithm>
#include<thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>

//for printing defines as strings
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)


using namespace std;

//function prototypes
uint64_t getLinearID_nDimensions(unsigned int * indexes, unsigned int * dimLen, unsigned int nDimensions);
void populateNDGridIndexAndLookupArray(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, struct gridCellLookup ** gridCellLookupArr, struct grid ** index, unsigned int * indexLookupArr,  DTYPE* minArr, unsigned int * nCells, uint64_t totalCells, unsigned int * nNonEmptyCells, unsigned int ** gridCellNDMask, unsigned int * gridCellNDMaskOffsets, unsigned int * nNDMaskElems);
void generateNDGridDimensions(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, DTYPE* minArr, DTYPE* maxArr, unsigned int * nCells, uint64_t * totalCells);
void importNDDataset(std::vector<std::vector <DTYPE> > *dataPoints, char * fname);
void sortInNDBins(std::vector<std::vector <DTYPE> > *dataPoints);


int main(int argc, char *argv[])
{

	/////////////////////////
	// Get information from command line
	//1) the dataset, 2) epsilon, 3) number of dimensions
	/////////////////////////

	//Read in parameters from file:
	//dataset filename and cluster instance file
	if (argc!=5)
	{
		cout <<"\n\nIncorrect number of input parameters.  \nShould be dataset file, epsilon, number of dimensions, searchmode\n";
		return 0;
	}

	//copy parameters from commandline:
	//char inputFname[]="data/test_data_removed_nan.txt";
	char inputFname[500];
	char inputEpsilon[500];
	char inputnumdim[500];

	strcpy(inputFname,argv[1]);
	strcpy(inputEpsilon,argv[2]);
	strcpy(inputnumdim,argv[3]);

    int SEARCHMODE = atoi(argv[4]);

	DTYPE epsilon=atof(inputEpsilon);
	unsigned int NDIM=atoi(inputnumdim);

	if (GPUNUMDIM!=NDIM){
		printf("\nERROR: The number of dimensions defined for the GPU is not the same as the number of dimensions\n \
		 passed into the computer program on the command line. GPUNUMDIM=%d, NDIM=%d Exiting!!!",GPUNUMDIM,NDIM);
		return 0;
	}

	printf("\nDataset file: %s", inputFname);
	printf("\nEpsilon: %f", epsilon);
	printf("\nNumber of dimensions (NDIM): %d\n", NDIM);

	//////////////////////////////
	//import the dataset:
	/////////////////////////////


	std::vector<std::vector <DTYPE> > NDdataPoints;
	importNDDataset(&NDdataPoints, inputFname);
	sortInNDBins(&NDdataPoints);

	ofstream gpu_stats;

	if(SEARCHMODE == 3){
		cout << "\n\n#########################################" << endl;
		cout << "# Search mode is : global memory kernel #" << endl;
		cout << "#########################################\n\n" << endl;
		char fname[] = "gpu_global_stats.txt";
		gpu_stats.open(fname,ios::app);
	}

	if(SEARCHMODE == 4){
		cout << "\n\n############################" << endl;
		cout << "# Search mode is : Unicomp #" << endl;
		cout << "############################\n\n" << endl;
		char fname[] = "gpu_unicomp_stats.txt";
		gpu_stats.open(fname,ios::app);
	}

	if(SEARCHMODE == 5){
		cout << "\n\n##############################" << endl;
		cout << "# Search mode is : B-Unicomp #" << endl;
		cout << "##############################\n\n" << endl;
		char fname[] = "gpu_b-unicomp_stats.txt";
		gpu_stats.open(fname,ios::app);
	}

	if(SEARCHMODE == 6){
		cout << "\n\n######################################" << endl;
		cout << "# Search mode is : Linear ID Unicomp #" << endl;
		cout << "######################################\n\n" << endl;
		char fname[] = "gpu_linID-unicomp_stats.txt";
		gpu_stats.open(fname,ios::app);
	}

	if(SEARCHMODE == 7){
		cout << "\n\n##############################################" << endl;
		cout << "# Search mode is : sorting cells by workload #" << endl;
		cout << "##############################################\n\n" << endl;
		char fname[] = "gpu_sorting-cells_stats.txt";
		gpu_stats.open(fname,ios::app);
	}

	if(SEARCHMODE == 8){
		cout << "\n\n#######################################################" << endl;
		cout << "# Search mode is : dynamic threads per points (fixed) #" << endl;
		cout << "#######################################################\n\n" << endl;
		char fname[] = "gpu_dynamic-threads-fixed_stats.txt";
		gpu_stats.open(fname,ios::app);
	}

	if(SEARCHMODE == 9){
		cout << "\n\n##################################" << endl;
		cout << "# Search mode is : working queue #" << endl;
		cout << "##################################\n\n" << endl;
		char fname[] = "gpu_working-queue_stats.txt";
		gpu_stats.open(fname,ios::app);
	}

	if(SEARCHMODE == 10){
		cout << "\n\n################################################" << endl;
		cout << "# Search mode is : working queue + LID-Unicomp #" << endl;
		cout << "################################################\n\n" << endl;
		char fname[] = "gpu_working-queue_lid-unicomp_stats.txt";
		gpu_stats.open(fname,ios::app);
	}

	printf("\n*****************\nWarming up GPU:\n*****************\n");
	warmUpGPU();
	printf("\n*****************\n");

	DTYPE * minArr = new DTYPE[NUMINDEXEDDIM];
	DTYPE * maxArr = new DTYPE[NUMINDEXEDDIM];
	unsigned int * nCells = new unsigned int[NUMINDEXEDDIM];
	uint64_t totalCells = 0;
	unsigned int nNonEmptyCells = 0;
	uint64_t totalNeighbors = 0;
	double totalTime = 0;

	generateNDGridDimensions(&NDdataPoints,epsilon, minArr, maxArr, nCells, &totalCells);
	printf("\nGrid: total cells (including empty) %lu",totalCells);

	// allocate memory for index now that we know the number of cells
	//the grid struct itself
	//the grid lookup array that accompanys the grid -- so we only send the non-empty cells
	struct grid * index; //allocate in the populateDNGridIndexAndLookupArray -- only index the non-empty cells
	struct gridCellLookup * gridCellLookupArr; //allocate in the populateDNGridIndexAndLookupArray -- list of non-empty cells

	//the grid cell mask tells you what cells are non-empty in each dimension
	//used for finding the non-empty cells that you want
	unsigned int * gridCellNDMask; //allocate in the populateDNGridIndexAndLookupArray -- list of cells in each n-dimension that have elements in them
	unsigned int * nNDMaskElems = new unsigned int; //size of the above array
	unsigned int * gridCellNDMaskOffsets = new unsigned int [NUMINDEXEDDIM * 2]; //offsets into the above array for each dimension
	//as [min,max,min,max,min,max] (for 3-D)

	//ids of the elements in the database that are found in each grid cell
	unsigned int * indexLookupArr = new unsigned int[NDdataPoints.size()];
	populateNDGridIndexAndLookupArray(&NDdataPoints, epsilon, &gridCellLookupArr, &index, indexLookupArr, minArr,  nCells, totalCells, &nNonEmptyCells, &gridCellNDMask, gridCellNDMaskOffsets, nNDMaskElems);

	//Neighbortable storage -- the result
	neighborTableLookup * neighborTable = new neighborTableLookup[NDdataPoints.size()];
	std::vector<struct neighborDataPtrs> pointersToNeighbors;

	pointersToNeighbors.clear();

	double tstart = omp_get_wtime();

	if((3 <= SEARCHMODE) && (SEARCHMODE <= 10))
	{
		distanceTableNDGridBatches(SEARCHMODE, &NDdataPoints, &epsilon, index, gridCellLookupArr, &nNonEmptyCells,  minArr, nCells, indexLookupArr, neighborTable, &pointersToNeighbors, &totalNeighbors, gridCellNDMask, gridCellNDMaskOffsets, nNDMaskElems);
	}

	double tend = omp_get_wtime();

	printf("\n\033[31;01m~~~Time: %f\033[00m",(tend - tstart));

	totalTime += (tend - tstart);

	gpu_stats << totalTime << ", " << inputFname << ", " << epsilon << ", " << totalNeighbors
		<< ", GPUNUMDIM/NUMINDEXEDDIM/DTYPE/BLOCKSIZE/THperPOINT/SORT: " << GPUNUMDIM << ", "
		<< NUMINDEXEDDIM << ", " << STR(DTYPE) << ", " << STR(BLOCKSIZE) << ", "
		<< STR(THREADPERPOINT) << ", " << STR(SORT_BY_WORKLOAD) << endl;

	gpu_stats.close();

	//TESTING: Print NeighborTable:

	for (int i = 0; i < NDdataPoints.size(); i++){
	//for (int i=0; i<10; i++){
	 	// sort to compare against CPU implementation
	 	std::sort(neighborTable[i].dataPtr + neighborTable[i].indexmin, neighborTable[i].dataPtr + neighborTable[i].indexmax + 1);
	 	printf("\npoint id: %d, neighbors: ", i);
	 	printf("nb neighbor %d\n", neighborTable[i].indexmax - neighborTable[i].indexmin + 1);
	 	for (int j = neighborTable[i].indexmin; j < neighborTable[i].indexmax; j++){
	 		printf("%d,", neighborTable[i].dataPtr[j]);
	 	}

	}

	printf("\n\n\n");
	return 0;
}


struct cmpStruct {
	cmpStruct(std::vector <std::vector <DTYPE>> points) {this -> points = points;}
	bool operator() (int a, int b) {
		return points[a][0] < points[b][0];
	}

	std::vector<std::vector<DTYPE>> points;
};



void populateNDGridIndexAndLookupArray(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, struct gridCellLookup ** gridCellLookupArr, struct grid ** index, unsigned int * indexLookupArr,  DTYPE* minArr, unsigned int * nCells, uint64_t totalCells, unsigned int * nNonEmptyCells, unsigned int ** gridCellNDMask, unsigned int * gridCellNDMaskOffsets, unsigned int * nNDMaskElems)
{

	/////////////////////////////////
	//Populate grid lookup array
	//and corresponding indicies in the lookup array
	/////////////////////////////////
	printf("\n\n*****************************\nPopulating Grid Index and lookup array:\n*****************************\n");
	// printf("\nSize of dataset: %lu", NDdataPoints->size());


	///////////////////////////////
	//First, we need to figure out how many non-empty cells there will be
	//For memory allocation
	//Need to do a scan of the dataset and calculate this
	//Also need to keep track of the list of uniquie linear grid cell IDs for inserting into the grid
	///////////////////////////////
	std::set<uint64_t> uniqueGridCellLinearIds;
	std::vector<uint64_t>uniqueGridCellLinearIdsVect; //for random access

	for (int i = 0; i < NDdataPoints->size(); i++){
		unsigned int tmpNDCellIdx[NUMINDEXEDDIM];
		for (int j = 0; j < NUMINDEXEDDIM; j++){
			tmpNDCellIdx[j] = (((*NDdataPoints)[i][j] - minArr[j]) / epsilon);
		}
		uint64_t linearID = getLinearID_nDimensions(tmpNDCellIdx, nCells, NUMINDEXEDDIM);
		uniqueGridCellLinearIds.insert(linearID);

	}

	// printf("uniqueGridCellLinearIds: %d",uniqueGridCellLinearIds.size());

	//copy the set to the vector (sets can't do binary searches -- no random access)
	std::copy(uniqueGridCellLinearIds.begin(), uniqueGridCellLinearIds.end(), std::back_inserter(uniqueGridCellLinearIdsVect));




	///////////////////////////////////////////////


	std::vector<uint64_t> * gridElemIDs;
	gridElemIDs = new std::vector<uint64_t>[uniqueGridCellLinearIds.size()];

	//Create ND array mask:
	//This mask determines which cells in each dimension has points in them.
	std::set<unsigned int> NDArrMask[NUMINDEXEDDIM];

	vector<uint64_t>::iterator lower;


	for (int i=0; i<NDdataPoints->size(); i++){
		unsigned int tmpNDCellID[NUMINDEXEDDIM];
		for (int j=0; j<NUMINDEXEDDIM; j++){
			tmpNDCellID[j]=(((*NDdataPoints)[i][j]-minArr[j])/epsilon);

			//add value to the ND array mask
			NDArrMask[j].insert(tmpNDCellID[j]);
		}

		//get the linear id of the cell
		uint64_t linearID=getLinearID_nDimensions(tmpNDCellID, nCells, NUMINDEXEDDIM);
		//printf("\nlinear id: %d",linearID);
		//if (linearID > totalCells){

		//	printf("\n\nERROR Linear ID is: %lu, total cells is only: %lu\n\n", linearID, totalCells);
		//}

		//find the index in gridElemIds that corresponds to this grid cell linear id

		lower=std::lower_bound(uniqueGridCellLinearIdsVect.begin(), uniqueGridCellLinearIdsVect.end(),linearID);
		uint64_t gridIdx=lower - uniqueGridCellLinearIdsVect.begin();
		gridElemIDs[gridIdx].push_back(i);
	}




	///////////////////////////////
	//Here we fill a temporary index with points, and then copy the non-empty cells to the actual index
	///////////////////////////////

	struct grid * tmpIndex=new grid[uniqueGridCellLinearIdsVect.size()];

	int cnt=0;



	//populate temp index and lookup array

	for (int i=0; i<uniqueGridCellLinearIdsVect.size(); i++)
	{
		tmpIndex[i].indexmin=cnt;
		for (int j=0; j<gridElemIDs[i].size(); j++)
		{
			if (j>((NDdataPoints->size()-1)))
			{
				printf("\n\n***ERROR Value of a data point is larger than the dataset! %d\n\n", j);
				return;
			}
			indexLookupArr[cnt]=gridElemIDs[i][j];
			cnt++;
		}
		tmpIndex[i].indexmax=cnt-1;
	}

	// printf("\nExiting grid populate method early!");
	// return;

	printf("\nFull cells: %d (%f, fraction full)",(unsigned int)uniqueGridCellLinearIdsVect.size(), uniqueGridCellLinearIdsVect.size()*1.0/double(totalCells));
	printf("\nEmpty cells: %ld (%f, fraction empty)",totalCells-(unsigned int)uniqueGridCellLinearIdsVect.size(), (totalCells-uniqueGridCellLinearIdsVect.size()*1.0)/double(totalCells));

	*nNonEmptyCells=uniqueGridCellLinearIdsVect.size();


	printf("\nSize of index that would be sent to GPU (GiB) -- (if full index sent), excluding the data lookup arr: %f", (double)sizeof(struct grid)*(totalCells)/(1024.0*1024.0*1024.0));
	printf("\nSize of compressed index to be sent to GPU (GiB) , excluding the data and grid lookup arr: %f", (double)sizeof(struct grid)*(uniqueGridCellLinearIdsVect.size()*1.0)/(1024.0*1024.0*1024.0));


	/////////////////////////////////////////
	//copy the tmp index into the actual index that only has the non-empty cells

	//allocate memory for the index that will be sent to the GPU
	*index=new grid[uniqueGridCellLinearIdsVect.size()];
	*gridCellLookupArr= new struct gridCellLookup[uniqueGridCellLinearIdsVect.size()];

	cmpStruct theStruct(*NDdataPoints);

	for (int i=0; i<uniqueGridCellLinearIdsVect.size(); i++){
		(*index)[i].indexmin=tmpIndex[i].indexmin;
		(*index)[i].indexmax=tmpIndex[i].indexmax;
		(*gridCellLookupArr)[i].idx=i;
		(*gridCellLookupArr)[i].gridLinearID=uniqueGridCellLinearIdsVect[i];
	}

	printf("\nWhen copying from entire index to compressed index: number of non-empty cells: %lu",uniqueGridCellLinearIdsVect.size());

	//copy NDArrMask from set to an array

	//find the total size and allocate the array

	unsigned int cntNDOffsets=0;
	unsigned int cntNonEmptyNDMask=0;
	for (int i=0; i<NUMINDEXEDDIM; i++){
		cntNonEmptyNDMask+=NDArrMask[i].size();
	}
	*gridCellNDMask = new unsigned int[cntNonEmptyNDMask];

	*nNDMaskElems=cntNonEmptyNDMask;


	//copy the offsets to the array
	for (int i=0; i<NUMINDEXEDDIM; i++){
		//Min
		gridCellNDMaskOffsets[(i*2)]=cntNDOffsets;
		for (std::set<unsigned int>::iterator it=NDArrMask[i].begin(); it!=NDArrMask[i].end(); ++it){
    		(*gridCellNDMask)[cntNDOffsets]=*it;
    		cntNDOffsets++;
		}
		//max
		gridCellNDMaskOffsets[(i*2)+1]=cntNDOffsets-1;
	}







	delete [] tmpIndex;



} //end function populate grid index and lookup array



//determines the linearized ID for a point in n-dimensions
//indexes: the indexes in the ND array: e.g., arr[4][5][6]
//dimLen: the length of each array e.g., arr[10][10][10]
//nDimensions: the number of dimensions


uint64_t getLinearID_nDimensions(unsigned int * indexes, unsigned int * dimLen, unsigned int nDimensions) {
    // int i;
    // uint64_t offset = 0;
    // for( i = 0; i < nDimensions; i++ ) {
    //     offset += (uint64_t)pow(dimLen[i],i) * (uint64_t)indexes[nDimensions - (i + 1)];
    // }
    // return offset;

    uint64_t index = 0;
	uint64_t multiplier = 1;
	for (int i = 0; i<nDimensions; i++){
  	index += (uint64_t)indexes[i] * multiplier;
  	multiplier *= dimLen[i];
	}

	return index;
}


//min arr- the minimum value of the points in each dimensions - epsilon
//we can use this as an offset to calculate where points are located in the grid
//max arr- the maximum value of the points in each dimensions + epsilon
//returns the time component of sorting the dimensions when SORT=1
void generateNDGridDimensions(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, DTYPE* minArr, DTYPE* maxArr, unsigned int * nCells, uint64_t * totalCells)
{

	printf("\n\n*****************************\nGenerating grid dimensions.\n*****************************\n");

	printf("\nNumber of dimensions data: %d, Number of dimensions indexed: %d", GPUNUMDIM, NUMINDEXEDDIM);

	//make the min/max values for each grid dimension the first data element
	for (int j=0; j<NUMINDEXEDDIM; j++){
		minArr[j]=(*NDdataPoints)[0][j];
		maxArr[j]=(*NDdataPoints)[0][j];
	}



	for (int i=1; i<NDdataPoints->size(); i++)
	{
		for (int j=0; j<NUMINDEXEDDIM; j++){
			if ((*NDdataPoints)[i][j]<minArr[j]){
				minArr[j]=(*NDdataPoints)[i][j];
			}
			if ((*NDdataPoints)[i][j]>maxArr[j]){
				maxArr[j]=(*NDdataPoints)[i][j];
			}
		}
	}


	printf("\n");
	for (int j=0; j<NUMINDEXEDDIM; j++){
		printf("Data Dim: %d, min/max: %f,%f\n",j,minArr[j],maxArr[j]);
	}

	//add buffer around each dim so no weirdness later with putting data into cells
	for (int j=0; j<NUMINDEXEDDIM; j++){
		minArr[j]-=epsilon;
		maxArr[j]+=epsilon;
	}

	for (int j=0; j<NUMINDEXEDDIM; j++){
		printf("Appended by epsilon Dim: %d, min/max: %f,%f\n",j,minArr[j],maxArr[j]);
	}

	//calculate the number of cells:
	for (int j=0; j<NUMINDEXEDDIM; j++){
		nCells[j]=ceil((maxArr[j]-minArr[j])/epsilon);
		printf("Number of cells dim: %d: %d\n",j,nCells[j]);
	}

	//calc total cells: num cells in each dim multiplied
	uint64_t tmpTotalCells=nCells[0];
	for (int j=1; j<NUMINDEXEDDIM; j++){
		tmpTotalCells*=nCells[j];
	}

	*totalCells=tmpTotalCells;

}
