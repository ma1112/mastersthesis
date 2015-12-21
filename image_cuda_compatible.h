#ifndef IMAGE_CUDA_COMPATIBLE_H
#define IMAGE_CUDA_COMPATIBLE_H

#include <iostream>
#include<stdio.h>
#include <string>
#include <algorithm>
#include <fstream>



class Image_cuda_compatible
{
public:
    Image_cuda_compatible();
    Image_cuda_compatible (unsigned short* array);
    ~Image_cuda_compatible ();
    void readfromarray(unsigned short* array);

    //Copy constructor.
    Image_cuda_compatible(const Image_cuda_compatible& image);

    friend float kernel_call_calculate_image_mean(const Image_cuda_compatible& im);


protected:
    float calculate_meanvalue();
    static const int width = 1536;
    static const int height = 864;
    static const long size = width * height;
    unsigned short* im;
    unsigned short average = 0 ;
    float mean = 0;
    std::string filename ="";
    std::string directory = "";
    std::string id = "";
    short voltage = 0;
    short  amperage = 0;
    short exptime = 0;
};

#endif // IMAGE_CUDA_COMPATIBLE_H
