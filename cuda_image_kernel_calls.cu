#include "cuda_reduce_kernels.cuh"
//#include "cuda_image_kernel_calls.h"
#include "image_cuda_compatible.h"
#include "book.h"
#include <stdio.h>
#include <iostream>
#include <string>


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

__global__ void kernel_loadFromUShortArray(unsigned short* d_ushort, float* d_image)
{
    unsigned int pixel = blockIdx.x*blockDim.x + threadIdx.x; //thread is computing pixel-th pixel
    d_image[pixel] = (float) d_ushort[pixel];
    return;
}


__global__ void kernel_exportToUSarray( float* d_image, unsigned short* d_ushort)
{
    unsigned int pixel = blockIdx.x*blockDim.x + threadIdx.x; //thread is computing pixel-th pixel
    d_ushort[pixel]   = (unsigned short) d_image[pixel];
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






//! Copies image to the GPU and calculates the mean intensity on the GPU.
void Image_cuda_compatible::calculate_meanvalue_on_GPU()
{
    if(gpu_im == NULL)
    {
        mean = 0.0f;
        return;
    }


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
  HANDLE_ERROR(cudaFree(d_data));
  HANDLE_ERROR(cudaFree(d_sum));
  HANDLE_ERROR(cudaFree(d_min));
  HANDLE_ERROR(cudaFree(d_max));


}






//! Reserves memory for the image on the GPU.
float* Image_cuda_compatible::reserve_on_GPU()
{
    if( gpu_im == NULL)
    {
      //  std::cout << "gpu_im ==" << gpu_im <<" And now mallocing memory. "
       //           <<std::endl << "filename: " << filename << std::endl;

        HANDLE_ERROR( cudaMalloc( (void**)&gpu_im,size*sizeof(float)));
     //   std::cout << "Malloc succesful & " << gpu_im <<std::endl;
       // std::cout << "Reserving memory on GPU for image "
                  //<<id << "at address @" << gpu_im <<std::endl;
       HANDLE_ERROR( cudaMemset(gpu_im,0,size*sizeof(float)));
    }

    return gpu_im;
}


//! Copies an image from the GPU memory to this image's memory on the GPU.
float* Image_cuda_compatible::copy_GPU_image(float* other)
{
    reserve_on_GPU();
    if(other !=NULL)
    {
        //std::cout << "Copy image data from @" << other<< " to @" <<gpu_im <<std::endl;
        HANDLE_ERROR (cudaMemcpy( gpu_im,other,size * sizeof(float),cudaMemcpyDeviceToDevice));
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

   //std::cout << "kernel_addimage (@" << gpu_im<< ", @" << other.gpu_im<<std::endl;
    kernel_addImage<<<2592,512>>>(gpu_im, other.gpu_im);
   // std::cout <<"done" << std::endl;
}

void Image_cuda_compatible::subtract_on_GPU(Image_cuda_compatible &other)
{

    kernel_subtractImage<<<2592,512>>>(gpu_im, other.gpu_im);

}

void Image_cuda_compatible::divide_on_GPU(float divisor)
{

    kernel_divideImage<<<2592,512>>>(gpu_im, divisor);

}

void Image_cuda_compatible::multiply_on_GPU(float multiplier)
{
    kernel_multiplyImage<<<2592,512>>>(gpu_im, multiplier);

}

void Image_cuda_compatible::cudaReadFromFile(const char* filename)
{

    FILE *file;
    file = fopen(filename,"rb");
    if (!file)
        {
                printf("Unable to open file! %s", filename);
                return;
        }
    unsigned short *temp, *d_temp;
    HANDLE_ERROR(cudaHostAlloc((void**)&temp, size*sizeof(unsigned short), cudaHostAllocDefault));
    fread(temp,sizeof(unsigned short),size,file);
    HANDLE_ERROR(cudaMalloc((void**)&d_temp,size*sizeof(unsigned short)));
    HANDLE_ERROR(cudaMemcpy(d_temp,temp,sizeof(unsigned short) * size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaFreeHost(temp));
    kernel_loadFromUShortArray<<<2592,512>>>(d_temp, reserve_on_GPU());
    HANDLE_ERROR(cudaFree(d_temp));
    fclose(file);


}


void Image_cuda_compatible::cudaReadFromFloatFile(const char* filename)
{
    FILE *file;
    file = fopen(filename,"rb");
    if (!file)
        {
                printf("Unable to open file! %s", filename);
                return;
        }
    float *temp;
    HANDLE_ERROR(cudaHostAlloc((void**)&temp, size*sizeof(float), cudaHostAllocDefault));
    fread(temp,sizeof(float),size,file);
    HANDLE_ERROR(cudaMemcpy(reserve_on_GPU(),temp,sizeof(float) * size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaFreeHost(temp));
    fclose(file);


}

void Image_cuda_compatible::cudaGetShortArrayToHost(unsigned short *h_sImage)
{
    unsigned short *d_usimage;
    HANDLE_ERROR(cudaMalloc((void**) & d_usimage, size*sizeof(unsigned short)));
   kernel_exportToUSarray<<<2592,512>>>( gpu_im, d_usimage);
   HANDLE_ERROR(cudaMemcpy(h_sImage, d_usimage, sizeof(unsigned short) * size , cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaFree(d_usimage));
   return;
}

void Image_cuda_compatible::cudaGetArrayToHost(float *h_image)
{
    HANDLE_ERROR(cudaMemcpy(h_image, gpu_im,sizeof(float) * size, cudaMemcpyDeviceToHost));
    return;
}



//! Writes image values to a binary file, with unsigned int values.

void Image_cuda_compatible::writetofile(std::string filename)
{
    unsigned short* sh_im = new unsigned short[size];
    cudaGetShortArrayToHost(sh_im);


    FILE *file;


    file = fopen(filename.c_str(), "wb");
    if(file == NULL)
    {
            std::cout << "Failed to open file " << filename << "for writing."<< std::endl;
            return;
    }
    fwrite(sh_im, sizeof(unsigned short), size, file );
    delete[] sh_im;
    fclose(file);

}
