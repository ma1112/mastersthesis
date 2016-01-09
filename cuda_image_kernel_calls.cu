#include "cuda_reduce_kernels.cuh"
//#include "cuda_image_kernel_calls.h"
#include "image_cuda_compatible.h"
#include "book.h"



void Image_cuda_compatible::remove_from_GPU()
{
    if(gpu_im != NULL)
        {
        cudaFree(gpu_im);
        gpu_im = NULL;
    }
}

void Image_cuda_compatible::copy_to_GPU()
{
    if(gpu_im == NULL)
        {
        cudaMalloc( (void**)&gpu_im,size*sizeof(float));
    }
    cudaMemcpy(gpu_im,im,size * sizeof(float),cudaMemcpyHostToDevice);

}

void Image_cuda_compatible::copy_to_GPU(float* destination)
{
    cudaMemcpy(destination,im,size * sizeof(float),cudaMemcpyHostToDevice);
}




//! Copies image to the GPU and calculates the mean intensity on the GPU.
void Image_cuda_compatible::calculate_meanvalue_on_GPU()
{
    copy_to_GPU();


  double* d_data;
  cudaMalloc( (void**)&d_data, sizeof(double) * 1024);
  double* d_sum;
  HANDLE_ERROR (cudaMalloc( (void**)&d_sum, sizeof(double)));

  kernel_reduce_sum_first_step<1024><<<64, 1024,  1024*sizeof(double)>>>(gpu_im,d_data, size);
  kernel_reduce_sum_second_step<64><<<1,64, 64*sizeof(double)>>>(d_data, d_sum);
  double *h_sum;
h_sum = (double*) malloc(sizeof(double));
HANDLE_ERROR (cudaMemcpy(h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
 mean = (float) ( (*h_sum )/ size);



free(h_sum);
  cudaFree(d_data);
  cudaFree(d_sum);

}

void Image_cuda_compatible::copy_from_GPU(float* d_image)
{
    cudaMemcpy(im,d_image, size*sizeof(float), cudaMemcpyDeviceToHost);
}








