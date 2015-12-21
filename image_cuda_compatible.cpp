#include "image_cuda_compatible.h"



Image_cuda_compatible::Image_cuda_compatible() {
    im = new unsigned short [width * height]();
    mean = 0;
}






Image_cuda_compatible::Image_cuda_compatible (unsigned short* array)
{
    im = new unsigned short [size];
    readfromarray(array);
}


Image_cuda_compatible::~Image_cuda_compatible() { delete[] im; }



void Image_cuda_compatible::readfromarray(unsigned short* array)
{
    double meandouble = 0;
    for(int i = 0 ; i < size ; i++)
        {
        im[i] = array[i];
        meandouble += im[i];
        //TODO: Error handling.
    }
    mean = float(meandouble / (double) size);

}


//copy constructor
Image_cuda_compatible::Image_cuda_compatible(const Image_cuda_compatible& Image_cuda_compatible)
{
    im = new unsigned short [size]();
    for(int i = 0; i < size ; i++)
        {
        im[i] = Image_cuda_compatible.im[i];
    }

}







float  Image_cuda_compatible::calculate_meanvalue()

{
    double  meanvalue = 0.0; // double for higher precision when summing awful lot of numbers.
    for(int i=0; i<size;i++)
    {
        meanvalue +=  im[i];
    }
    meanvalue = meanvalue / (double) size;

    return (float) meanvalue;

}


