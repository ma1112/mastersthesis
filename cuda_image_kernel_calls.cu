#include "cuda_reduce_kernels.cuh"
//#include "cuda_image_kernel_calls.h"
#include "image_cuda_compatible.h"
#include "book.h"

//! Kernel to add another image's pixel values to this image.
__global__ void kernel_addImage(float* d_this, float* d_other)
{

    unsigned int pixel = blockIdx.x*blockDim.x + threadIdx.x; //thread is computing pixel-th pixel
    d_this[pixel] += d_other[pixel];
    return;

}

//! Kernel to substract another image's pixel values from this image.

__global__ void kernel_subtractImage(float* d_this, float* d_other)
{

    unsigned int pixel = blockIdx.x*blockDim.x + threadIdx.x; //thread is computing pixel-th pixel
    d_this[pixel] -= d_other[pixel];
    return;

}



//! Kernel to divide all pixel values by a float.
__global__ void kernel_divideImage(float* d_this, float divisor)
{

    unsigned int pixel = blockIdx.x*blockDim.x + threadIdx.x; //thread is computing pixel-th pixel
    d_this[pixel] /= divisor;
    return;
}



__global__ void kernel_multiplyImage(float* d_this, float multiplier)
{

    unsigned int pixel = blockIdx.x*blockDim.x + threadIdx.x; //thread is computing pixel-th pixel
    d_this[pixel] *= multiplier;
    return;
}

//! Deassings memory from the GPU.
void Image_cuda_compatible::remove_from_GPU()
{
    if(gpu_im != NULL)
        {
       HANDLE_ERROR ( cudaFree(gpu_im));
        gpu_im = NULL;
    }
}

//! Reserves memory on the GPU for the image and copies data from the CPU memory. Return with the device pointer.

float* Image_cuda_compatible::copy_to_GPU()
{
    if(im != NULL)
    {
      //  std::cout <<"copy_to_GPU(): im!= NULL: im = @" << im <<std::endl;
        reserve_on_GPU();
      //  std::cout <<"copy_to_GPU():reserved on GPU. GPU im = @" << gpu_im <<std::endl;

        HANDLE_ERROR (cudaMemcpy(gpu_im,im,size * sizeof(float),cudaMemcpyHostToDevice));
       // std::cout <<"copy_to_GPU():copied to GPU." <<std::endl;

        return gpu_im;

    }
    else
    {
        return reserve_on_GPU();

    }




}

void Image_cuda_compatible::copy_to_GPU(float* destination)
{
    if(destination != NULL)
    {
        //std::cout <<"Copying image " <<filename <<"to GPU." <<std::endl;
    HANDLE_ERROR ( cudaMemcpy(destination,im,size * sizeof(float),cudaMemcpyHostToDevice));
    }
    else
    {
        std::cout << "WARNNG: space is not reserved on GPU." << std::endl;
    }
}




//! Copies image to the GPU and calculates the mean intensity on the GPU.
void Image_cuda_compatible::calculate_meanvalue_on_GPU()
{
    copy_to_GPU();


  float* d_data;
  HANDLE_ERROR (cudaMalloc( (void**)&d_data, 3*sizeof(float) * 1024));
  float *d_sum, *d_min, *d_max;
  HANDLE_ERROR (cudaMalloc( (void**)&d_sum, sizeof(float)));
  HANDLE_ERROR (cudaMalloc( (void**)&d_min, sizeof(float)));
  HANDLE_ERROR (cudaMalloc( (void**)&d_max, sizeof(float)));


  kernel_reduce_sum_first_step<1024><<<64, 1024,  3*1024*sizeof(float)>>>(gpu_im, d_data, size);
  kernel_reduce_sum_second_step<64><<<1,64, 3*64*sizeof(float)>>>(d_data, d_sum, d_min, d_max);
  float *h_sum;
h_sum = (float*) malloc(sizeof(float));
HANDLE_ERROR (cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
 mean =  ( (*h_sum )/ size);
 HANDLE_ERROR (cudaMemcpy(&min, d_min, sizeof(float), cudaMemcpyDeviceToHost));
 HANDLE_ERROR (cudaMemcpy(&max, d_max, sizeof(float), cudaMemcpyDeviceToHost));




free(h_sum);
  cudaFree(d_data);
  cudaFree(d_sum);
  cudaFree(d_min);
  cudaFree(d_max);

}


//! Copies an image from a GPU memory address to image's GPU memory.
void Image_cuda_compatible::copy_GPU_array(float* d_image)
{
    reserve_on_GPU();
    if(d_image != NULL && d_image != gpu_im)
    {
        HANDLE_ERROR ( cudaMemcpy(im,d_image, size*sizeof(float), cudaMemcpyDeviceToHost));
        remove_from_CPU();
    }
    else if (d_image==NULL)

    {
        std::cout << "WARNING: Image is not on the GPU. " << std::endl;
    }
}

//! Copies image to the CPU from the GPU.
void Image_cuda_compatible::copy_from_GPU()
{
    reserve_on_CPU();
    if(gpu_im != NULL)
    {
        HANDLE_ERROR ( cudaMemcpy(im,gpu_im, size*sizeof(float), cudaMemcpyDeviceToHost));
        remove_from_GPU();
    }
    else
    {
        //std::cout << "WARNING: image" << id<<" is not on the GPU. " <<std::endl;
    }
}


//! Reserves memory for the image on the GPU.
float* Image_cuda_compatible::reserve_on_GPU()
{
    if( gpu_im == NULL)
    {

        HANDLE_ERROR( cudaMalloc( (void**)&gpu_im,size*sizeof(float)));
       // std::cout << "Reserving memory on GPU for image "
                  //<<id << "at address @" << gpu_im <<std::endl;
        cudaMemset(gpu_im,0,size*sizeof(float));
    }
    return gpu_im;
}


//! Copies an image from the GPU memory to this image's memory on the GPU.
float* Image_cuda_compatible::copy_GPU_image(float* other)
{
    reserve_on_GPU();
    if(other !=NULL)
    {
       // std::cout << "Copy image data from @" << other<< " to @" <<gpu_im <<std::endl;
        HANDLE_ERROR (cudaMemcpy( gpu_im,other,size * sizeof(float),cudaMemcpyDeviceToDevice));
        remove_from_CPU();
    }
    else
    {
        std::cout <<"WARNING: The image you want to copy is not on the GPU." << std::endl;
    }
    return gpu_im;
}

//! Adds an image's pixel values to this image on the GPU.
void Image_cuda_compatible::add_on_GPU(Image_cuda_compatible &other)
{
//    std::cout << "Add_on_GPU()" << std::endl;
    other.copy_to_GPU();
    copy_to_GPU();
   //std::cout << "kernel_addimage (@" << gpu_im<< ", @" << other.gpu_im<<std::endl;
    kernel_addImage<<<2592,512>>>(gpu_im, other.gpu_im);
   // std::cout <<"done" << std::endl;
    remove_from_CPU();
}

void Image_cuda_compatible::subtract_on_GPU(Image_cuda_compatible &other)
{
    other.copy_to_GPU();
    copy_to_GPU();
    kernel_subtractImage<<<2592,512>>>(gpu_im, other.gpu_im);
    remove_from_CPU();
}

void Image_cuda_compatible::divide_on_GPU(float divisor)
{
    copy_to_GPU();
    kernel_divideImage<<<2592,512>>>(gpu_im, divisor);
    remove_from_CPU();
}

void Image_cuda_compatible::multiply_on_GPU(float multiplier)
{
    copy_to_GPU();
    kernel_multiplyImage<<<2592,512>>>(gpu_im, multiplier);
    remove_from_CPU();
}




