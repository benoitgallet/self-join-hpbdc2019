#include "structs.h"
#include <math.h>

int bin_x(double x);
int bin_y(double x);

//going to sort the data.
//however, the points are sorted based on a binned granularity of 1 degree (can modify this later) 
//this is so that for a given x value, the y values will be ordered together

//compare function for std sort for the vector
//Sorts in the order: x, y, TEC, time
bool compareDataElemStructFunc(const dataElem &elem1, const dataElem &elem2)
{
   //compare based on the x-coodinate
   if ( bin_x(elem1.x) < bin_x(elem2.x))
      {
      return true;
      }
   else if (bin_x(elem1.x) > bin_x(elem2.x))
      {
      return false;
     }
      //if the x-coordinates are equal, compare on the y-coordinate
   else if ( bin_y(elem1.y) < bin_y(elem2.y))
         {
      return true;
         }
   else if (bin_y(elem1.y) > bin_y(elem2.y))
         {
      return false;
         }
         else{
      return false;
         }
}


//calculate the bin for a point, that's in the range of x:0-180 (latitude) 
//1 degree bins
int bin_x(double x)
{
int num_bins=180;

//set x to be a positive value (add 250)
double total_width=180;
return (ceil((x/total_width)*num_bins));
}

//calculate the bin for a point, that's in the range of y:0-360 (longitude)
//1 degree bins
int bin_y(double x)
{
int num_bins=360;

//set x to be a positive value (add 250)
double total_width=360;
return (ceil((x/total_width)*num_bins));
}