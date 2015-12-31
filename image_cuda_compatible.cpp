#include "image_cuda_compatible.h"



Image_cuda_compatible::Image_cuda_compatible() {
    im = new float [width * height]();
    mean = 0;
}






Image_cuda_compatible::Image_cuda_compatible (float* array)
{
    im = new float [size];
    readfromarray(array);
}


Image_cuda_compatible::~Image_cuda_compatible() { delete[] im; }



void Image_cuda_compatible::readfromarray(float* array)
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
Image_cuda_compatible::Image_cuda_compatible(const Image_cuda_compatible& other)
    :filename(other.filename), directory(other.directory), id(other.id),
      voltage(other.voltage), amperage(other.amperage), exptime(other.exptime),
      im(new float[size]())
{

    for(int i = 0; i < size ; i++)
        {
        im[i] = other.im[i];
    }

}







void  Image_cuda_compatible::calculate_meanvalue_on_CPU()

{
    double  meanvalue = 0.0; // double for higher precision when summing awful lot of numbers.
    for(int i=0; i<size;i++)
    {
        meanvalue +=  im[i];
    }
    meanvalue = meanvalue / (double) size;

mean = (float) meanvalue;
}


float Image_cuda_compatible::getvoltage()
{
    return voltage;
}

float Image_cuda_compatible::getamperage()
{
    return amperage;
}

float Image_cuda_compatible::getexptime()
{
    return exptime;
}

float Image_cuda_compatible::getmean()
{
    return mean;
}

void Image_cuda_compatible::setvoltage(float f)
{
    voltage = f;
    return;
}

void Image_cuda_compatible::setamperage(float f)
{
    amperage = f;
    return;
}

void Image_cuda_compatible::setexptime(float f)
{
    exptime = f;
    return;
}
