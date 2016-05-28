#include "gc_im_container.h"
#include<iostream>

gc_im_container::gc_im_container()
{
images = 0;

long double xMean = 0;
long double x2Mean = 0;
//slope.reserve_on_GPU();
//intercept.reserve_on_GPU();
xy.reserve_on_GPU();
y.reserve_on_GPU();

}


void gc_im_container::clear()
{
  //  slope.clear();
   // intercept.clear();
    images = 0;
    long double xMean = 0;
    long double x2Mean = 0;
    y.clear();
    xy.clear();
}

int gc_im_container::getimageno()
{
    return images;
}













