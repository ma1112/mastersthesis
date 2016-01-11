#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gc_im_container.h"

gc_im_container::~gc_im_container()
{
   removefromgpu();
   removefromhost();
}

void  gc_im_container::inicialize(int n)
{
    removefromhost();

    if( n > 0)
    {
        size = n;
        cudaHostAlloc( (void**) &h_images, sizeof(float) *1327104 * size, cudaHostAllocDefault);
        cudaHostAlloc( (void**) &h_settings, sizeof(float)  * size, cudaHostAllocDefault);

        cudaMemset(&h_images, 0,sizeof(float) *1327104 * size );

        cudaMemset(&h_settings, 0,sizeof(float)  * size );
    }
return;
}

void gc_im_container::removefromgpu()
{
    if(d_images != NULL)
    {
        cudaFree(d_images);
        d_images = NULL;
    }
    if(d_settings != NULL)
    {
        cudaFree(d_settings);
        d_settings = NULL;
    }
}

void gc_im_container::removefromhost()
{
    if(h_images != NULL)

    {
        cudaFreeHost(h_images);
        h_images = NULL;
    }
    if(h_settings != NULL)
    {
        cudaFreeHost(h_settings);
        h_settings=NULL;
    }
}





void gc_im_container::copy_to_GPU(float*& d_im, float*& d_set)
{
    if(images > 0 && h_images != NULL && h_settings!= NULL)
    {
        removefromgpu();

        if(images > 0)
        {
            cudaMalloc( (void**) &d_im, sizeof(float) *1327104 * size );
            cudaMalloc( (void**) &d_set, sizeof(float)  * size );

            d_images = d_im;
            d_settings = d_settings;
        }
        cudaMemcpy(d_images, h_images, sizeof(float) *1327104 * size , cudaMemcpyHostToDevice );
        cudaMemcpy(d_settings, h_settings, sizeof(float) * size , cudaMemcpyHostToDevice );
    }

}

void gc_im_container::reserveIfEmpty(int n)
{
    if( size ==0)
    {

        inicialize(n);
    }
    return;
}


