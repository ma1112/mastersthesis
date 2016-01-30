    #include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"stdio.h"


//sums a long array to a Blockdim size array. Extern shared is used! Need second step to sum the last array.
//Highly optimised version, slightly modified, from
//http://docs.nvidia.com/cuda/samples/6_Advanced/reduction/doc/reduction.pdf

//NOTE!!!!:::
//Google's first hit on "cuda reduction" IS A LIE. ( https://a248.e.akamai.net/f/248/10/10/developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf)
//(It is outdated and not working.)
//warp synchronous programming should be avoided unless using volatile compiler switch.
//(I had to learn it the hard way...)


//! Adds 64 or less elements on GPU, using only 1 warp.
template <unsigned int blockSize>
__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
if (blockSize >= 64) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid + blockSize] = fminf(sdata[tid + blockSize] , sdata[tid + blockSize + 32]);
    sdata[tid + 2*blockSize] = fmaxf(sdata[tid + 2*blockSize] , sdata[tid + 2* blockSize + 32]);
    }

if (blockSize >= 32){
    sdata[tid] += sdata[tid + 16];
    sdata[tid + blockSize] = fminf(sdata[tid + blockSize] , sdata[tid + blockSize + 16]);
    sdata[tid + 2*blockSize] = fmaxf(sdata[tid + 2*blockSize] , sdata[tid + 2* blockSize + 16]);
    }
if (blockSize >= 16) {
    sdata[tid] += sdata[tid + 8];
    sdata[tid + blockSize] = fminf(sdata[tid + blockSize] , sdata[tid + blockSize + 8]);
    sdata[tid + 2*blockSize] = fmaxf(sdata[tid + 2*blockSize] , sdata[tid + 2* blockSize + 8]);
    }
if (blockSize >= 8) {
    sdata[tid] += sdata[tid + 4];
    sdata[tid + blockSize] = fminf(sdata[tid + blockSize] , sdata[tid + blockSize + 4]);
    sdata[tid + 2*blockSize] = fmaxf(sdata[tid + 2*blockSize] , sdata[tid + 2* blockSize + 4]);
    }
if (blockSize >= 4) {
    sdata[tid] += sdata[tid + 2];
    sdata[tid + blockSize] = fminf(sdata[tid + blockSize] , sdata[tid + blockSize + 2]);
    sdata[tid + 2*blockSize] = fmaxf(sdata[tid + 2*blockSize] , sdata[tid + 2* blockSize + 2]);
    }
if (blockSize >= 2) {
    sdata[tid] += sdata[tid + 1];
    sdata[tid + blockSize] = fminf(sdata[tid + blockSize] , sdata[tid + blockSize + 1]);
    sdata[tid + 2*blockSize] = fmaxf(sdata[tid + 2*blockSize] , sdata[tid + 2* blockSize + 1]);
    }
}







//! First step of a reduce procedure on GPU.
//!
//! Adds the elements of the n long float array *g_idta
//! and stores partial sums in the blockSize long float *g_odata array.
template <int blockSize>
__global__ void kernel_reduce_sum_first_step(float *g_idata, float *g_odata, unsigned int n)
{
extern __shared__ float sdata[];
 unsigned int tid = threadIdx.x;
 unsigned int i = blockIdx.x*(blockSize) + tid; //modified here
 unsigned int gridSize = blockSize*gridDim.x; //modified here
sdata[tid] = 0.0f;
sdata[blockSize + tid] = 1e30f; //min
sdata[2* blockSize + tid] = 0.0f; //max
while (i < n) {
    sdata[tid] += g_idata[i];
    sdata[blockSize + tid] = fminf( sdata[ blockSize + tid], g_idata[i]);
    sdata[2*blockSize + tid] = fmaxf(sdata[2*blockSize + tid] , g_idata[i]);
    i += gridSize;  } //modified here
__syncthreads();
if (blockSize >= 1024) { if (tid < 512) {
        sdata[tid] += sdata[tid + 512];
        sdata[blockSize + tid] = fminf(sdata[blockSize + tid] , sdata[blockSize + tid + 512]);
        sdata[2*blockSize + tid] = fmaxf(sdata[2*blockSize + tid] , sdata[2*blockSize + tid + 512]);
   } __syncthreads(); } //added line
if (blockSize >= 512) { if (tid < 256) {
        sdata[tid] += sdata[tid + 256];
        sdata[blockSize + tid] = fminf(sdata[blockSize + tid] , sdata[blockSize + tid + 256]);
        sdata[2*blockSize + tid] = fmaxf(sdata[2*blockSize + tid] , sdata[2*blockSize + tid + 256]);
    } __syncthreads(); }
if (blockSize >= 256) { if (tid < 128) {
        sdata[tid] += sdata[tid + 128];
        sdata[blockSize + tid] = fminf(sdata[blockSize + tid] , sdata[blockSize + tid + 128]);
        sdata[2*blockSize + tid] = fmaxf(sdata[2*blockSize + tid] , sdata[2*blockSize + tid + 128]);
    } __syncthreads(); }
if (blockSize >= 128) { if (tid < 64) {
        sdata[tid] += sdata[tid + 128];
        sdata[blockSize + tid] = fminf(sdata[blockSize + tid] , sdata[blockSize + tid + 128]);
        sdata[2*blockSize + tid] = fmaxf(sdata[2*blockSize + tid] , sdata[2*blockSize + tid + 128]);
    } __syncthreads(); }
if (tid < 32) {
    warpReduce<blockSize>(sdata, tid);
}
if (tid == 0){
    g_odata[blockIdx.x] = sdata[0]; //sum
    g_odata[blockIdx.x + gridDim.x] = sdata[blockSize]; //min
    g_odata[blockIdx.x + 2* gridDim.x] = sdata[ 2* blockSize]; //max
}


}



//! Second step of a reduce procedure on GPU.
//!
//! Adds the elements of d_in and stores the sum in the d_out float.
template <unsigned int blockSize>
__global__ void kernel_reduce_sum_second_step(float * d_in, float* d_sum, float* d_min, float* d_max )
{
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    sdata[tid] = d_in[tid];
    sdata[tid+blockSize] = d_in[tid + blockSize];
    sdata[tid+ 2 * blockSize] = d_in[tid + 2* blockSize];
    __syncthreads();
    if (blockSize >= 1024) { if (tid < 512) {
            sdata[tid] += sdata[tid + 512];
            sdata[tid + blockSize] = fminf(sdata[tid + blockSize] , sdata[tid + blockSize + 512]);
            sdata[tid + 2 * blockSize] = fmaxf(sdata[tid + 2 * blockSize] , sdata[tid + 2* blockSize + 512]);
        } __syncthreads(); }
    if (blockSize >= 512) { if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
            sdata[tid + blockSize] = fminf(sdata[tid + blockSize] , sdata[tid + blockSize + 256]);
            sdata[tid + 2 * blockSize] = fmaxf(sdata[tid + 2 * blockSize] , sdata[tid + 2* blockSize + 256]);
        } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
            sdata[tid + blockSize] = fminf(sdata[tid + blockSize] , sdata[tid + blockSize + 128]);
            sdata[tid + 2 * blockSize] = fmaxf(sdata[tid + 2 * blockSize] , sdata[tid + 2* blockSize + 128]);
        } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
            sdata[tid + blockSize] = fminf(sdata[tid + blockSize] , sdata[tid + blockSize + 64]);
            sdata[tid + 2 * blockSize] = fmaxf(sdata[tid + 2 * blockSize] , sdata[tid + 2* blockSize + 64]);
        } __syncthreads(); }
    if (tid < 32) {
     warpReduce<blockSize>(sdata, tid);

}
    if (tid == 0) { *d_sum =  sdata[0]; *d_min= sdata[blockSize]; *d_max = sdata[2*blockSize];}


}

