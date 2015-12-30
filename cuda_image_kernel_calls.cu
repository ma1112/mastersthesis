#include "cuda_reduce_kernels.cuh"
#include "cuda_image_kernel_calls.h"
#include "book.h"




//! Copies image to the GPU and calculates the mean intensity on the GPU.
//! The image is than deleted from the GPU.
float kernel_call_calculate_image_mean(const Image_cuda_compatible& im)
{
    long imagesize = im.size;
    unsigned short* d_image;
    cudaMalloc( (void**)&d_image,imagesize*sizeof(unsigned short));
  double* d_data;
  cudaMalloc( (void**)&d_data, sizeof(double) * 1024);
  double* d_sum;
  HANDLE_ERROR (cudaMalloc( (void**)&d_sum, sizeof(double)));

  cudaMemcpy(d_image,im.im,im.size * sizeof(unsigned short),cudaMemcpyHostToDevice);
  kernel_reduce_sum_first_step<1024><<<64, 1024,  1024*sizeof(double)>>>(d_image,d_data, imagesize);
  kernel_reduce_sum_second_step<64><<<1,64, 64*sizeof(double)>>>(d_data, d_sum);
  double *h_sum;
h_sum = (double*) malloc(sizeof(double));
HANDLE_ERROR (cudaMemcpy(h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
float mean = (float) ( (*h_sum )/ imagesize);



free(h_sum);
  cudaFree(d_image);
  cudaFree(d_data);
  cudaFree(d_sum);

return  mean;
}







