#include "image_cuda_compatible.h"



Image_cuda_compatible::Image_cuda_compatible() : voltage(0),amperage(0),exptime(0),mean(0) {
    im = new float [width * height]();

}







Image_cuda_compatible::Image_cuda_compatible (float* array) : Image_cuda_compatible()
{

    readfromarray(array);
}


Image_cuda_compatible::~Image_cuda_compatible() { delete[] im; remove_from_GPU(); }



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
{    std::cout<<"Callling copy constr. of image_cuda_compatible az &"<< this<<std::endl;

      *this = other;
}


Image_cuda_compatible& Image_cuda_compatible::operator=(const Image_cuda_compatible& other)
 {
    std::cout<<"Callling operator= of image_cuda_compatible"<<std::endl;


    if(this != &other)
    {

     voltage = other.voltage;
     amperage= other.amperage;
     exptime = other.exptime;
      mean=other.mean;

         filename = other.filename;
         directory = other.directory;
          id = other.id;
          delete [] im;


     im = new float[size];

     for(int i = 0; i < size ; i++)
         {
         im[i] = other.im[i];
         }
     }
    return *this;


 }

 Image_cuda_compatible&  Image_cuda_compatible::operator+=(const Image_cuda_compatible &other)
 {
    for(int i = 0; i< size; i++)
        {
        im[i] += other.im[i];

    }
    mean+=other.mean;
    return *this;
}



//! Sets every variable to default and removes the image from the GPU.
void Image_cuda_compatible::clear()
{
    for(int i=0;i<size;i++)
    {
        im[i] = 0;
    }
    remove_from_GPU();
    filename="";
    directory="";
    mean=0;
    voltage =0;
    amperage=0;
    exptime = 0;
}

//Working feature but it may not be used in the future.
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

std::string Image_cuda_compatible::getid()
{
    return id;
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

float Image_cuda_compatible::minintensity()
{
float intensity =im[0];
for(int i=1;i<size;i++)
{
    if(im[i] < intensity ) intensity = im[i];
}
return intensity;
}

float Image_cuda_compatible::maxintensity()
{
float intensity =im[0];
for(int i=1;i<size;i++)
{
    if(im[i] > intensity ) intensity = im[i];
}
return intensity;
}
