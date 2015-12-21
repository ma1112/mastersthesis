#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_kernel_calls.h"
#include "book.h" // from "Cuda by Example"
//#include <stdio.h>
//#include <iostream>


//sums a long array to a Blockdim size array. Extern shared is used! Need second step to sum the last array.
//Highly optimised version, slightly modified, from
//http://docs.nvidia.com/cuda/samples/6_Advanced/reduction/doc/reduction.pdf

//NOTE!!!!:::
//Google's first hit on "cuda reduction" IS A LIE. ( https://a248.e.akamai.net/f/248/10/10/developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf)
//(It is outdated and not working.)
//warp synchronous programming should be avoided unless using volatile compiler switch.
//(I had to learn it the hard way...)



template <unsigned int blockSize>
__device__ void warpReduce(volatile double *sdata, unsigned int tid) {
if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}






template <int blockSize>
__global__ void kernel_reduce_sum_first_step(unsigned short *g_idata, double *g_odata, unsigned int n)
{
extern __shared__ double sdata[];
 unsigned int tid = threadIdx.x;
 unsigned int i = blockIdx.x*(blockSize) + tid; //modified here
 unsigned int gridSize = blockSize*gridDim.x; //modified here
sdata[tid] = 0.0f;
while (i < n) { sdata[tid] += g_idata[i]; i += gridSize;  } //modified here
__syncthreads();
if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); } //added line
if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
if (tid < 32) {
    warpReduce<blockSize>(sdata, tid);
}
if (tid == 0){ g_odata[blockIdx.x] = sdata[0]; }


}



//second step on reducing.
template <unsigned int blockSize>
__global__ void kernel_reduce_sum_second_step(double * d_in, double* d_out )
{
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    sdata[tid] = d_in[tid];
    __syncthreads();
    if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) {
     warpReduce<blockSize>(sdata, tid);

}
    if (tid == 0) { *d_out =  sdata[0];}


}

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




