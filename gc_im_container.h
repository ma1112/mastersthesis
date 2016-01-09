#ifndef GC_IM_CONTAINER_H
#define GC_IM_CONTAINER_H
#include "image_cuda_compatible.h"


class gc_im_container
{
public:
    gc_im_container();
    ~gc_im_container();
   void  add(Image_cuda_compatible im);
   void  inicialize(int n);
   void removefromhost();
   void removefromgpu();
   void copy_to_GPU(float *&d_im, float *&d_set);
   void reserveIfEmpty( int n);
   int getsize();
   int getimageno();
   float calculateXmean();
   float calculateX2mean();



private:
    float* h_images;
    float* d_images;
    int images;
    int size;
    float* h_settings;
    float* d_settings;
};

#endif // GC_IM_CONTAINER_H
