#ifndef IMAGE_CUDA_COMPATIBLE_H
#define IMAGE_CUDA_COMPATIBLE_H

#include <iostream>
#include<stdio.h>
#include <string>
#include <algorithm>
#include <fstream>


//! \class Image_cuda_compatible
//! \brief: Stores an image and it's info. Does not use QT related elements.
//!
//! Not using QT related elements makes it able to be used in CUDA.

class Image_cuda_compatible
{
public:
    Image_cuda_compatible(); //!< Default constructor.
    Image_cuda_compatible (unsigned short* array);  //!<Constructor that copies image from a this.size long array
    ~Image_cuda_compatible (); //!<Destructor.
    void readfromarray(unsigned short* array); //!< Copies image from an array.


    Image_cuda_compatible(const Image_cuda_compatible& image); //!< Copy constructor.



protected:
    void  calculate_meanvalue_on_CPU();  //!<Calculates the mean value of the image on the CPU.
    static const int width = 1536;  //!<Width of the image. Constant among all images.
    static const int height = 864; //!< Height of the image. Constant among all images.
    static const long size = width * height; //!< Size of the image. Constant among all images.
    unsigned short* im; //!< Array to store image values. unsigned short sould be 16 bit.
    unsigned short average = 0 ; //!< Mean value of the image.
    float mean = 0;  //!< Mean value of the image.
    std::string filename ="";  //!< File name that the image was read from.
    std::string directory = "";  //!< Directory name that the image was read from.
    std::string id = "";  //!< ID of the image.
    //Technical info:
    short voltage = 0;  //!< Voltage of the X-ray tube
    short  amperage = 0;  //!< Amperage of the X-ray tube
    short exptime = 0;  //!< Exposure time.

    //GPU stuff:

    unsigned short* gpu_im = NULL; //!< Poiter to the image on the GPU
    void copy_to_GPU();
    void remove_from_GPU();
    void  calculate_meanvalue_on_GPU(); //!<Calculates the mean value of the image on the GPU.
};

#endif // IMAGE_CUDA_COMPATIBLE_H
