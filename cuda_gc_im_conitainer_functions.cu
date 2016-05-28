#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gc_im_container.h"
#include<iostream>
#include"book.cuh"




__global__ void kernel_gaincorr_calculator(float x, float denominator, float* d_xy, float* d_y, float* d_slope, float* d_intercept)
{


    unsigned int tid = threadIdx.x;
    unsigned int pixel = blockIdx.x*blockDim.x + tid; //thread is computing pixel-th pixel slope and intercept.



    d_slope[pixel] = (d_xy[pixel] - x*d_y[pixel]) / denominator;
    d_intercept[pixel] = d_y[pixel] - d_slope[pixel] * x;


    return;
}


__global__ void kernel_add_xy(float x,  float* d_xy, float* d_y)
{
    unsigned int tid = threadIdx.x;
    unsigned int pixel = blockIdx.x*blockDim.x + tid; //thread is computing pixel-th pixel slope and intercept.

    d_xy[pixel] += d_y[pixel] * x;
    return;
}







void gc_im_container::calculate(Image_cuda_compatible &slope, Image_cuda_compatible &intercept)
{
    xy /= images;
    y /=images;
    x /= images;
    x2 /=images;
    float denominator = x2 - x*x;
kernel_gaincorr_calculator<<<2592,512>>>(x, denominator, xy.gpu_im, y.gpu_im, slope.gpu_im, intercept.gpu_im);

            slope.calculate_meanvalue_on_GPU();
            intercept.calculate_meanvalue_on_GPU();
            std::cout << "slope mean: " << slope.getmean()<<std::endl; // DEBUG
            std::cout << "intercept mean : " << intercept.getmean() << std::endl;



    return;
}

void gc_im_container::add(Image_cuda_compatible &im)
{
    if(im.getmax() > 16382)
    {
        std::cout << " Warning: tried to add saturated image to linear fit." << std::endl;
        return;
    }
    y +=im;
    float x_now = im.getexptime() * im.getamperage();
    kernel_add_xy<<<2592,512>>>(x_now,  xy.gpu_im, y.gpu_im);
    images++;
    x += x_now,
    x2 += x_now*x_now;
}


/*
//! Kernel for calculating gain correction factors. (Linear fit on pixels). Launch with <<<x,32>>>

//! d_images should be formatted as gc_im_container formats it: First n values belong to the


__global__ void kernel_gaincorr_calculator(float* d_images, float* d_settings, float* d_slope, float* d_intercept, int n, int size, float xmean, float denominator)
{
    __shared__ float y[32];
    __shared__ float xy[32];


    unsigned int tid = threadIdx.x;
    unsigned int pixel = blockIdx.x*blockDim.x + tid; //thread is computing pixel-th pixel slope and intercept.



            xy[tid] = 0;
            y[tid] = 0;
            int gloffset = pixel * size;

    for(int i =0;i<n;i++)
    {

        y[tid] += d_images[gloffset+i];
        xy[tid] +=  d_images[gloffset+i] * d_settings[i];
    }
    y[tid] = y[tid] / n;
    xy[tid] = xy[tid] / n;

    d_slope[pixel] = (xy[tid] - (xmean * y[tid])) / denominator;
    d_intercept[pixel] = y[tid] - d_slope[pixel] * xmean;


    return;
}
*/















/*


__global__ void kernel_gc_im_add(float* d_images,  float* d_other, int size, int images)
{

    unsigned int pixel = blockIdx.x*blockDim.x + threadIdx.x; //thread is computing pixel-th pixel
    d_images[images +  pixel * size] = d_other[pixel];
    return;
}
__global__ void kernel_set_settings(float* d_settings, float value, int images)
{
    d_settings[images] = value;
}

__global__ void kernel_calculate_xmean_atomic(float* d_settings, int images, float* xmean )
{
    if(threadIdx.x == 0)
    {
        *xmean =0.0f;
    }
    __syncthreads();
    if(threadIdx.x <= images )
    {
        atomicAdd( xmean, d_settings[threadIdx.x]);
    }
    __syncthreads();
    if(threadIdx.x ==0)
    {
       *xmean = *xmean / images;
    }
    return;

}



__global__ void kernel_calculate_x2mean_atomic(float* d_settings, int images, float* x2mean )
{
    if(threadIdx.x == 0)
    {
        *x2mean =0.0f;
    }
    __syncthreads();
    if(threadIdx.x <= images )
    {
        atomicAdd( x2mean, d_settings[threadIdx.x] * d_settings[threadIdx.x]);
    }
    __syncthreads();
    if(threadIdx.x ==0)
    {
       *x2mean = *x2mean / images;
    }
    return;

}


float gc_im_container::calculateXmean()
{
    float* d_xmean;
    float h_xmean;
    HANDLE_ERROR (cudaMalloc( (void**)&d_xmean, sizeof(float)));
    kernel_calculate_xmean_atomic<<<1,images>>>(d_settings, images, d_xmean);
    HANDLE_ERROR (cudaMemcpy(&h_xmean,d_xmean,sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR (cudaFree(d_xmean));
    return h_xmean;
}


float gc_im_container::calculateX2mean()
{
    float* d_x2mean;
    float h_x2mean;
   HANDLE_ERROR (cudaMalloc( (void**)&d_x2mean, sizeof(float)));
    kernel_calculate_x2mean_atomic<<<1,images>>>(d_settings, images, d_x2mean);
    HANDLE_ERROR (cudaMemcpy(&h_x2mean,d_x2mean,sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR (cudaFree(d_x2mean));
    return h_x2mean;
}

*/



/*
void gc_im_container::calculate(Image_cuda_compatible &slope, Image_cuda_compatible &intercept)
{

            float xmean = calculateXmean();
            float x2mean = calculateX2mean();
            float denominator = x2mean - (xmean * xmean);
            slope.reserve_on_GPU();
            intercept.reserve_on_GPU();

            kernel_gaincorr_calculator<<<41472,32>>>(d_images, d_settings, slope.gpu_im, intercept.gpu_im, images, size, xmean, denominator);  //41472 * 32  = 1327104



            slope.calculate_meanvalue_on_GPU();
            intercept.calculate_meanvalue_on_GPU();
            std::cout << "slope mean: " << slope.getmean()<<std::endl; // DEBUG
            std::cout << "intercept mean : " << intercept.getmean() << std::endl;



    return;
}


void gc_im_container::add(Image_cuda_compatible &im)
{
    if(images < size)
    {
        kernel_gc_im_add<<<2592,512>>> (d_images, im.gpu_im, size, images);
        kernel_set_settings<<<1,1>>> (d_settings, (im.getamperage() * im.getexptime()), images );
        images++;
    }
    else
    {
        std::cout << "WARNING Gc_im_Container is full!!!" <<std::endl;
    }


}
*/
/*

gc_im_container::~gc_im_container()
{
clear();
}

void  gc_im_container::inicialize(int n)
{
    clear();

    if( n > 0)
    {
        size = n;
        HANDLE_ERROR (cudaMalloc( (void**)&d_images,  sizeof(float) *1327104 * size));
        HANDLE_ERROR (cudaMalloc( (void**)&d_settings,  sizeof(float)  * size));

        HANDLE_ERROR (cudaMemset(d_images, 0,sizeof(float) *1327104 * size ));
        HANDLE_ERROR (cudaMemset(d_settings, 0,sizeof(float)  * size ));

    }
return;
}

void gc_im_container::clear()
{
    if(d_images != NULL)
    {
        HANDLE_ERROR (cudaFree(d_images));
        d_images = NULL;
    }
    if(d_settings != NULL)
    {
        HANDLE_ERROR (cudaFree(d_settings));
        d_settings =NULL;
    }
    images =0;
    size =0;
}

*/









