#include <stdio.h>
#include <vector>
#include <fstream>
#include <istream>
#include <iostream>
#include <string>
#include <string.h>
#include <sstream>
#include <cstdlib>
// #include "prototypes.h"
#include <algorithm>
// #include "globals.h"

#include "params.h"

bool sortNDComp(const std::vector<DTYPE>& a, const std::vector<DTYPE>& b)
{
    for (int i=0; i<GPUNUMDIM; i++){
      if (int(a[i])<int(b[i])){
      return 1;
      }
      else if(int(a[i])>int(b[i])){
      return 0;
      }  
    }

    return 0;

    
}



void importNDDataset(std::vector<std::vector <DTYPE> > *dataPoints, char * fname)
{

	std::vector<DTYPE>tmpAllData;
	std::ifstream in(fname);
	int cnttmp=0;
	for (std::string f; getline(in, f, ',');){
	
	DTYPE i;
		 std::stringstream ss(f);
	    while (ss >> i)
	    {
	        tmpAllData.push_back(i);
	        //std::cout<<tmpAllData[cnttmp++]<<"\n";
	        if (ss.peek() == ',')
	            ss.ignore();
	    }
  		
  	}	




  	unsigned int cnt=0;
  	const unsigned int totalPoints=(unsigned int)tmpAllData.size()/GPUNUMDIM;
  	printf("\nData import: Total size of all data (1-D) vect (number of points * GPUNUMDIM): %zu",tmpAllData.size());
  	printf("\nData import: Total data points: %d",totalPoints);
  	
  	for (int i=0; i<totalPoints; i++){
  		std::vector<DTYPE>tmpPoint;
  		for (int j=0; j<GPUNUMDIM; j++){
  			tmpPoint.push_back(tmpAllData[cnt]);
  			cnt++;
  		}
  		dataPoints->push_back(tmpPoint);
  	}



}


void sortInNDBins(std::vector<std::vector <DTYPE> > *dataPoints){
  
  std::sort(dataPoints->begin(),dataPoints->end(),sortNDComp);
  
}



