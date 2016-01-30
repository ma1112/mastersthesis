#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gaincorr.h"
#include "math.h"
#include <cuda.h>
















__global__ void kernel_do_gaincorr (float* d_slope, float* d_intercept, int* d_saturation, float* d_image)
{

    unsigned int tid = threadIdx.x;
    unsigned int pixel = blockIdx.x*blockDim.x + tid; //thread is computing pixel-th pixel

    //printf(" pixel: %d \t tid: %d \t blockIdx : %d \t blockDim : %d \n", pixel,tid, blockIdx.x, blockDim.x);



        d_image[pixel] = (d_image[pixel] - d_intercept[pixel] ) / d_slope[pixel]  * 16383.0f / *d_saturation;
    return;
}







void Gaincorr::gaincorrigateimage(Image_cuda_compatible& image)
{
    //Rounding voltage to multiply of 5
    int voltage = (int) (round(image.getvoltage()));
    int remainder  = voltage %5;

    if(remainder != 0)

    {
        voltage = voltage + 5 - remainder;
    }

//DEBUG
    if(saturation.find(voltage) == saturation.end())
    {
        std::cout <<"Error: no calbration data found for image" <<image.getid() << "With voltage " << image.getvoltage() <<std::endl;
        return;
    }

    if(slopes.find(voltage) == slopes.end())
    {
        std::cout <<"Error: no slope data found for image" <<image.getid() << "With voltage " << image.getvoltage() <<std::endl;
        return;
    }

    if(intercepts.find(voltage) == intercepts.end())
    {
        std::cout <<"Error: no slope data found for image" <<image.getid() << "With voltage " << image.getvoltage() <<std::endl;
        return;
    }



    int* d_saturation;
    //DEBUG
    int sat = saturation[voltage];
    float* d_slope;
    float* d_intercept;
    d_slope = slopes.find(voltage)->second.copy_to_GPU();
    d_intercept = intercepts.find(voltage)->second.copy_to_GPU();
    float* d_image;

    d_image= image.copy_to_GPU();

    cudaMalloc( (void**)&d_saturation, sizeof(int) );
   cudaMemcpy(d_saturation, &sat, sizeof(int), cudaMemcpyHostToDevice );





    kernel_do_gaincorr<<<41472,32>>>( d_slope,  d_intercept, d_saturation,  d_image );
    image.copy_GPU_array(d_image);
    image.remove_from_CPU();








    cudaFree(d_saturation);

}



