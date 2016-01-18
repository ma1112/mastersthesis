#ifndef GC_IM_CONTAINER_H
#define GC_IM_CONTAINER_H
#include "image_cuda_compatible.h"


class gc_im_container
{
public:
    gc_im_container();
    ~gc_im_container();
   void  add(Image_cuda_compatible &im);

   void  inicialize(int n);
    void clear();
   int getimageno();
   float calculateXmean();
   float calculateX2mean();
   void calculate(Image_cuda_compatible &slope, Image_cuda_compatible &intercept);



private:
    float* d_images;
    int images;
    int size;
    float* d_settings;
};

#endif // GC_IM_CONTAINER_H
