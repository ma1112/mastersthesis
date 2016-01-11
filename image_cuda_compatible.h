#ifndef IMAGE_CUDA_COMPATIBLE_H
#define IMAGE_CUDA_COMPATIBLE_H

#include <iostream>
#include<stdio.h>
#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>
//#include "gc_im_container.h"
class gc_im_container;
class gaincorr;



//! \class Image_cuda_compatible
//! \brief: Stores an image and it's info. Does not use QT related elements.
//!
//! Not using QT related elements makes it able to be used in CUDA.

class Image_cuda_compatible
{
friend class gc_im_container;
friend class gaincorr;

public:
    Image_cuda_compatible(); //!< Default constructor.
    Image_cuda_compatible (float* array);  //!<Constructor that copies image from a this.size long array
   virtual ~Image_cuda_compatible (); //!<Destructor.
    void readfromarray(float* array); //!< Copies image from an array.
    Image_cuda_compatible& operator=( const Image_cuda_compatible& other); //!< Assigment operator
     Image_cuda_compatible& operator+=(const Image_cuda_compatible &other);
     Image_cuda_compatible& operator/=(int n);

     void clear(); //!<Cleans the image.

     float* copy_to_GPU();
     void copy_to_GPU(float* destination);
     void copy_from_GPU(float* d_image);
     void remove_from_GPU();
     void  calculate_meanvalue_on_GPU(); //!<Calculates the mean value of the image on the GPU.

     static const int width = 1536;  //!<Width of the image. Constant among all images.
     static const int height = 864; //!< Height of the image. Constant among all images.
     static const long size = 1327104; //!< Size of the image. Constant among all images.


     void readfromfile(std::string filename ); //!< Reads image data from file.
     void readfromfloatfile(std::string fname);
     void readinfo(); //!< Reads image info from the info file. Info file must be in the same folder as the image.

     void writetofile(std::string filename);
     void writetofloatfile(std::string filename);

    Image_cuda_compatible(const Image_cuda_compatible& other); //!< Copy constructor.


    void  calculate_meanvalue_on_CPU();  //!<Calculates the mean value of the image on the CPU.


    //Getter setters:

    void setvoltage(float f);
    void setamperage(float f);
    void setexptime(float f);
    float getvoltage();
    float getamperage();
    float getexptime();
    float getmean();
    std::string getid();
    float minintensity(); //!<Experimental feature
    float maxintensity();//!<Experimental feature
    std::string getfilename();




protected:

    void initialize(); //!< Initializes the image. Sets everything to 0 or NULL, creates the im array. Used in constructors.
    float* im; //!< Array to store image values. float sould be 16 bit.
    float mean ;  //!< Mean value of the image.
    float deviation; //!< Standard deviation of the pixel values.
    std::string filename ;  //!< File name that the image was read from.
    std::string directory ;  //!< Directory name that the image was read from.
    std::string id ;  //!< ID of the image.
    //Technical info:
    float voltage ;  //!< Voltage of the X-ray tube
    float  amperage ;  //!< Amperage of the X-ray tube
    float exptime ;  //!< Exposure time.


    //GPU stuff:

    float* gpu_im; //!< Poiter to the image on the GPU



};

#endif // IMAGE_CUDA_COMPATIBLE_H
