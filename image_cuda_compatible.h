#ifndef IMAGE_CUDA_COMPATIBLE_H
#define IMAGE_CUDA_COMPATIBLE_H


#include <string>

//#include "gc_im_container.h"
class gc_im_container;
class Gaincorr;



//! \class Image_cuda_compatible
//! \brief: Stores an image and it's info. Does not use QT related elements.
//!
//! Not using QT related elements makes it able to be used in CUDA.

class Image_cuda_compatible
{
friend class gc_im_container;
friend class Gaincorr;

public:
    Image_cuda_compatible(); //!< Default constructor.
   virtual ~Image_cuda_compatible (); //!<Destructor.
    Image_cuda_compatible& operator=( const Image_cuda_compatible& other); //!< Assigment operator
     Image_cuda_compatible& operator+=( Image_cuda_compatible &other);
     Image_cuda_compatible& operator-= (Image_cuda_compatible &other);
     Image_cuda_compatible& operator*=(float n);
     Image_cuda_compatible& operator/=(float n);

     void clear(); //!<Cleans the image.

     void copy_GPU_array(float* d_image);
     void remove_from_GPU();
     void  calculate_meanvalue_on_GPU(); //!<Calculates the mean value of the image on the GPU.



     static const int width = 1536;  //!<Width of the image. Constant among all images.
     static const int height = 864; //!< Height of the image. Constant among all images.
     static const long size = 1327104; //!< Size of the image. Constant among all images.


     void readfromfile(std::string filename ); //!< Reads image data from file.
     void cudaReadFromFile(const char* filename); //! Cuda function that copies data from file to GPU.
     void cudaReadFromFloatFile(const char* filename);

     void readfromfloatfile(std::string fname);
     void readinfo(); //!< Reads image info from the info file. Info file must be in the same folder as the image.

     void writetofile(std::string filename);
     void writetofloatfile(std::string filename);

    Image_cuda_compatible(const Image_cuda_compatible& other); //!< Copy constructor.



    void cudaGetShortArrayToHost(unsigned short *h_sImage);
    void cudaGetArrayToHost(float *h_image);




    //Getter setters:

    void setvoltage(float f);
    void setamperage(float f);
    void setexptime(float f);
    float getvoltage();
    float getamperage();
    float getexptime();
    float getmean();
    float getmax();
    float getmin();
    std::string getid();
    std::string getfilename();




protected:
    void divide_on_GPU(float divisor);
    void multiply_on_GPU(float multiplier);


    void initialize(); //!< Initializes the image. Sets everything to 0 or NULL, creates the im array. Used in constructors.
    // Images are not stored on the host anymore. float* im; //!< Array to store image values. float sould be 16 bit.
    float mean ;  //!< Mean value of the image.
    float max;
    float min;
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
    float* copy_GPU_image(float* other);
    float* reserve_on_GPU();
    void add_on_GPU(Image_cuda_compatible &other);
    void subtract_on_GPU(Image_cuda_compatible &other);



};

#endif // IMAGE_CUDA_COMPATIBLE_H
