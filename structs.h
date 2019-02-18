

#ifndef STRUCTS_H
#define STRUCTS_H
#include <vector>
#include <stdio.h>
#include <iostream>


//thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "params.h"

struct key_val_sort
{
	unsigned int pid; //point id
	DTYPE value_at_dim;
};




struct dim_reorder_sort
{
	unsigned int dim; //point dimension
	DTYPE variance; //variance of the points in this dimension
};



struct keyData{
	int key;
	int position;
};


//need to pass in the neighbortable thats an array of the dataset size.
//carry around a pointer to the array that has the points within epsilon though
struct neighborTableLookup
{
	int pointID;
	int indexmin;
	int indexmax;
	int * dataPtr;
};



//a struct that points to the arrays of individual data points within epsilon
//and the size of each of these arrays (needed to construct a subsequent neighbor table)
//will be used inside a vector.
struct neighborDataPtrs{
	int * dataPtr;
	int sizeOfDataArr;
};


//the result set:
// struct structresults{
// int pointID;
// int pointInDist;
// };


//the neighbortable.  The index is the point ID, each contains a vector
//only for the GPU Brute force implementation
struct table{
	int pointID;
	std::vector<int> neighbors;
};

//index lookup table for the GPU. Contains the indices for each point in an array
//where the array stores the direct neighbours of all of the points
struct gpulookuptable{
	int indexmin;
	int indexmax;
};

struct grid{
	int indexmin; //Contains the indices for each point in an array where the array stores the ids of the points in the grid
	int indexmax;
};

//key/value pair for the gridCellLookup -- maps the location in an array of non-empty cells
struct gridCellLookup{
	unsigned int idx; //idx in the "grid" struct array
	uint64_t gridLinearID; //The linear ID of the grid cell
	//compare function for linearID
	bool operator<(const gridCellLookup & other) const
	{
		return gridLinearID < other.gridLinearID;
	}
};

struct schedulingCell{
	int nbPoints;
	int cellId;
};




#endif
