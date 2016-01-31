#ifndef GC_IM_CONTAINER_H
#define GC_IM_CONTAINER_H
#include "image_cuda_compatible.h"

/*! \class gc_im_container
    \brief Class to store images before linear fit. Uses special order of data.

    In this class, a huge piece of memory is allocated to store images. Images are stored such way
    that the first pixel of each N images are in the first N place, etc. This special structure of
    the data helps CUDA to read it effectively.
    It also stores image exptime*amperage data separately.

    */

class gc_im_container
{
public:
    gc_im_container(); //!< Default constructor.
    ~gc_im_container(); //!< Default destructor. Deassigns memory from the GPU.
   void  add(Image_cuda_compatible &im); //!< Adds in image to the containter.

   void  inicialize(int n); //!< Assings memory for n images on the GPU.
    void clear(); //!< Deassings memory
   int getimageno(); //!< Returns the number of images in the container.
   float calculateXmean(); //!< Calculates and returns mean of amp*exptime of the images in the container.
   float calculateX2mean(); //!<Calculates and returns mean of (amp*exptime)^2 of the images in the container.

   //! Fits a line on the image pixel data in the function of exptime*amperage of the image.
   //! Line slope and intercept are stored in the given images.
   void calculate(Image_cuda_compatible &slope, Image_cuda_compatible &intercept);


private:
    float* d_images; //!< Array for images, on the GPU.
    int images; //!< counts added images.
    int size; //! Size of the array. (There is room for size number of images.)
    float* d_settings; //! Array for exptime*amperage products on the GPU.
};

#endif // GC_IM_CONTAINER_H
