#include "cuda_reduce_kernels.cuh"
//#include "cuda_image_kernel_calls.h"
#include "image_cuda_compatible.h"
#include "book.cuh"
#include <stdio.h>
#include <iostream>
#include <string>
#include <algorithm>    // std::max, std::min




//!Kernel to do equalmax. Every pixel becomes max(value on this image, value on other image.)

__global__ void kernel_equalmax(float* d_this, float* d_other)
{
    unsigned int pixel = blockIdx.x*blockDim.x + threadIdx.x; //thread is computing pixel-th pixel
    d_this[pixel] = max(d_other[pixel], d_this[pixel]);
    return;

}

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


//! Kernel to miltiply all pixel values by a float.

__global__ void kernel_multiplyImage(float* d_this, float multiplier)
{

    unsigned int pixel = blockIdx.x*blockDim.x + threadIdx.x; //thread is computing pixel-th pixel
    d_this[pixel] *= multiplier;
    return;
}

//! Kernel to load an image from an unsigned short array on the GPU.

//! Used in cudaReadFromFile() where the file is read into an unsigned short array, copied to the GPU
//! and then assigned to the float array.
__global__ void kernel_loadFromUShortArray(unsigned short* d_ushort, float* d_image)
{
    unsigned int pixel = blockIdx.x*blockDim.x + threadIdx.x; //thread is computing pixel-th pixel
    d_image[pixel] = (float) d_ushort[pixel];
    return;
}

//! Kernel that exports the float array of an image to an unsigned short array.
//! Used when communicating with the CPU.
__global__ void kernel_exportToUSarray( float* d_image, unsigned short* d_ushort)
{
    unsigned int pixel = blockIdx.x*blockDim.x + threadIdx.x; //thread is computing pixel-th pixel
    d_ushort[pixel]   = (unsigned short) d_image[pixel];
    return;
}

//! Kernel that exports the float array of an image to an unsigned char array.
//! Used when communicating with the CPU.
__global__ void kernel_exportToUCArray(float* d_image, unsigned char *d_ucimage, float min, float max)
{
    unsigned int pixel = blockIdx.x*blockDim.x + threadIdx.x; //thread is computing pixel-th pixel
    d_ucimage [pixel] = (unsigned char) ((255 * ( d_image[pixel] - min) / max ));
}


__global__ void kernel_correlate2d(float* d_this, float* d_other, float* result, float meanThis, float meanOther)
{
    unsigned int pixel = blockIdx.x*blockDim.x + threadIdx.x; //thread is computing pixel-th pixel
    result[pixel] = (d_this[pixel] - meanThis) * (d_other[pixel] - meanOther);
    return;
}

//! Kernel to invert only one pixel. Used when drawing a cross to the image.
__global__ void kernel_invert_pixel(float* d_image, int x, int y,const int numCols, float minimum, float maximum)
{
    d_image[x + y*numCols] = 2* maximum;
}


float Image_cuda_compatible::correlateWith(Image_cuda_compatible &other)
{
    calculate_meanvalue_on_GPU();
    other.calculate_meanvalue_on_GPU();
    Image_cuda_compatible corr2;

    kernel_correlate2d<<<2592,512>>>(gpu_im, other.gpu_im, corr2.reserve_on_GPU(), mean, other.mean);
    corr2.calculate_meanvalue_on_GPU();
    return (corr2.getmean() / ( stdev * other.stdev));

}



//! Deassings memory from the GPU.
void Image_cuda_compatible::remove_from_GPU()
{
    if(gpu_im != NULL)
        {
      // std::cout << "removing image" << filename<<" from @" << gpu_im <<std::endl;
       HANDLE_ERROR ( cudaFree(gpu_im));
        gpu_im = NULL;
    }
}






//! Copies image to the GPU and calculates the mean intensity on the GPU.
void Image_cuda_compatible::calculate_meanvalue_on_GPU()
{
    mean = 0.0f;
    max = 1e30f;
    min =0.0f;
    stdev = 0.0f;

    if(gpu_im == NULL)
    {
        std::cout <<"ERROR: When calculating mean on image " << id
                 <<std::endl << "Image is empty. ( NULL)" << std::endl;

        return;
    }

  float* d_data;
  HANDLE_ERROR (cudaMalloc( (void**)&d_data, 3*sizeof(float) * 1024));
  float *d_sum, *d_min, *d_max, *d_stdev;
  HANDLE_ERROR (cudaMalloc( (void**)&d_sum, sizeof(float)));

  HANDLE_ERROR (cudaMalloc( (void**)&d_min, sizeof(float)));
  HANDLE_ERROR (cudaMalloc( (void**)&d_max, sizeof(float)));
  HANDLE_ERROR (cudaMalloc( (void**)&d_stdev, sizeof(float)));



  kernel_reduce_sum_first_step<1024><<<64, 1024,  4*1024*sizeof(float)>>>(gpu_im, d_data, size);
  kernel_reduce_sum_second_step<64><<<1,64, 4*64*sizeof(float)>>>(d_data, d_sum, d_min, d_max, d_stdev);
  float *h_sum;
h_sum = (float*) malloc(sizeof(float));
HANDLE_ERROR (cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
 mean =  ( (*h_sum )/ size);
 HANDLE_ERROR (cudaMemcpy(&min, d_min, sizeof(float), cudaMemcpyDeviceToHost));
 HANDLE_ERROR (cudaMemcpy(&max, d_max, sizeof(float), cudaMemcpyDeviceToHost));
 HANDLE_ERROR (cudaMemcpy(&stdev, d_stdev, sizeof(float), cudaMemcpyDeviceToHost));
 stdev /=size;
 if(stdev - mean*mean < 0)
 {
     std::cout <<"WARNING: Stdev < 0 at image " << id << std::endl;
     stdev = max;
 }
 else
 {
    stdev = sqrt(stdev - mean*mean);
 }

free(h_sum);
  HANDLE_ERROR(cudaFree(d_data));
  HANDLE_ERROR(cudaFree(d_sum));
  HANDLE_ERROR(cudaFree(d_min));
  HANDLE_ERROR(cudaFree(d_max));
  HANDLE_ERROR(cudaFree(d_stdev));



}






//! Reserves memory for the image on the GPU.
float* Image_cuda_compatible::reserve_on_GPU()
{
    if( gpu_im == NULL)
    {
     //   std::cout << "gpu_im ==" << gpu_im <<" And now mallocing memory. "
      //           <<std::endl << "filename: " << filename
      //          <<"size:" << size <<std::endl;

        HANDLE_ERROR( cudaMalloc( (void**)&gpu_im,size*sizeof(float)));
     //   std::cout << "Malloc succesful & " << gpu_im <<std::endl;
       // std::cout << "Reserving memory on GPU for image "
       //            <<id << "at address @" << gpu_im <<std::endl;
       HANDLE_ERROR( cudaMemset(gpu_im,0,size*sizeof(float)));
       //std::cout << "reserved memory @" << gpu_im << std::endl;
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

//! Every pixel becomes the maximum of itself or the pixel at the other image.
void Image_cuda_compatible::equalmax(Image_cuda_compatible &other)
{
    if(other.gpu_im==NULL)
    {
        return;
    }
    reserve_on_GPU();
    kernel_equalmax<<<2592,512>>>(gpu_im, other.gpu_im);
}




//! Adds an image's pixel values to this image on the GPU.
void Image_cuda_compatible::add_on_GPU(Image_cuda_compatible &other)
{
//    std::cout << "Add_on_GPU()" << std::endl;

   //std::cout << "kernel_addimage (@" << gpu_im<< ", @" << other.gpu_im<<std::endl;
    kernel_addImage<<<2592,512>>>(gpu_im, other.gpu_im);
   // std::cout <<"done" << std::endl;
}

//! Subtracts an image's pixel values to this image on the GPU.
void Image_cuda_compatible::subtract_on_GPU(Image_cuda_compatible &other)
{

    kernel_subtractImage<<<2592,512>>>(gpu_im, other.gpu_im);

}

//! Divides an image's pixel values to this image on the GPU.
void Image_cuda_compatible::divide_on_GPU(float divisor)
{

    kernel_divideImage<<<2592,512>>>(gpu_im, divisor);

}

//! Multiplies an image's pixel values to this image on the GPU.
void Image_cuda_compatible::multiply_on_GPU(float multiplier)
{
    kernel_multiplyImage<<<2592,512>>>(gpu_im, multiplier);

}










//! Clears pixels around the given (x,y) coordinates, within a
//! square(!) with r/2+1 side.
void Image_cuda_compatible::clearwitinradius(int x, int y, int r)
{
    //int pixel = x + y * width;
    for (int dy = -r; dy <=r; dy++)
    {
        for( int dx = -r; dx <=r; dx++)
        {
            int pixel2 = x + dx + (y+ dy)*width;
            if(x + dx >=0 && x + dx < width && y+ dy >=0 && y+ dy < height )
            {
            HANDLE_ERROR( cudaMemset(gpu_im + pixel2,0,sizeof(float) ));
            }
        }

    }

}

//! Reads the image from an unsigned short binary file to a float array on the GPU.
//! (Data from the scanner is unsighed short.)
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

//! Reads the image from a float binary file to a float array on the GPU.
//! This program is able to save an image as a binary float array.)
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

//! Exports the (float) image array from the GPU to the CPU, as an unsigned short array.
//! Used when saving image as a binary file.
void Image_cuda_compatible::cudaGetShortArrayToHost(unsigned short *h_sImage)
{
    unsigned short *d_usimage;
    HANDLE_ERROR(cudaMalloc((void**) & d_usimage, size*sizeof(unsigned short)));
   kernel_exportToUSarray<<<2592,512>>>( gpu_im, d_usimage);
   HANDLE_ERROR(cudaMemcpy(h_sImage, d_usimage, sizeof(unsigned short) * size , cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaFree(d_usimage));
   return;
}

//! Exports the (float) image array from the GPU to the CPU, as an float array.
//! Used when saving image as a float binary file.
void Image_cuda_compatible::cudaGetArrayToHost(float *h_image)
{
    if(gpu_im == NULL)
    {
        std::cout<< "ERROR: trying to copy empty (NULL) image to host" << filename << std::endl;
    }
    HANDLE_ERROR(cudaMemcpy(h_image, gpu_im,sizeof(float) * size, cudaMemcpyDeviceToHost));
    return;
}



//! Writes image values to a binary file, with unsigned short values.

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

//! Exports the (float) image array from the GPU to the CPU, as an unsigned char array.
//! Used when saving image as a JPEG file or drawing it.
void Image_cuda_compatible::cudaGetUCArrayToHost(unsigned char *h_image)
{
    unsigned char *d_ucimage;
    calculate_meanvalue_on_GPU();
    HANDLE_ERROR(cudaMalloc((void**) & d_ucimage, size*sizeof(unsigned char)));
    if( getmin() == getmax())
    {
        memset(h_image,0,sizeof(unsigned char) * size);
        return;
    }
    kernel_exportToUCArray<<<2592,512>>>(gpu_im, d_ucimage, getmin(), getmax());
    HANDLE_ERROR(cudaMemcpy(h_image, d_ucimage, sizeof(unsigned char) * size , cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(d_ucimage));
    return;

}


//! Draws a cross to the image. Used when displaying Hough circle midpoints.
void Image_cuda_compatible::drawCross(int x, int y, int size)
{
    if( y < 0 || y >= height || x< 0 || x >= width)
    {
        std::cout << "WARNING : Could no draw cross at coordinates " << x << "," << y <<
                     ". (Out of boundary)." <<std::endl;
        return;
    }
    if( size <= 0)
    {
        std::cout << "Could not draw criss with size " << size <<"." << std::endl;
        return;
    }


    for(int dx = (int)(- round(size * 0.5f)); dx <= (int)( round(size * 0.5f)); dx++)
    {
        int xnew;
        xnew = x + dx;
        xnew  = std::min(xnew,(width-1));
        xnew = std::max(0,xnew);
        for(int j=-2; j<=2; j++)
        {
            if( 0 <= y + j && height > y +j)
            {
                kernel_invert_pixel<<<1,1>>>(reserve_on_GPU(), xnew, y+j,width, getmin(),getmax());

            }
        }

    }
    for(int dy = (int)(- round(size * 0.5f)); dy <= (int)( round(size * 0.5f)); dy++)
    {
        int ynew;
        ynew = y + dy;
        ynew  =std::min(ynew, height-1);
        ynew = std::max(0,ynew);
        for(int j = -2; j<=2; j++)
        {
            if( 0 <= x + j && width > x +j )
            {
                kernel_invert_pixel<<<1,1>>>(reserve_on_GPU(), x+j, ynew,width, getmin(),getmax());

            }
        }
    }


}
