#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gaincorr.h"
#include "math.h"
#include <cuda.h>


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









void Gaincorr::calculate()
{

    for ( std::map<int, gc_im_container >::iterator iter = imagemap.begin(); iter != imagemap.end();iter++)
    {
        int voltage = iter -> first;
        int images = iter->second.getimageno();
        int size = iter->second.getsize();
        if ( images < 4) //DEBUG //TODO Set higher value!
        {
            std::cout << "ERROR when gain calibrating: Not enough images with voltage " << voltage << std::endl;
        }

        else
        {
            std::cout <<"Gain calibrating " << images <<" images with " <<voltage <<" kV voltage." <<std::endl;
            float* d_slope;
            float* d_intercept;
            cudaMalloc( (void**)&d_slope, sizeof(float) * 1327104);
            cudaMalloc( (void**)&d_intercept, sizeof(float) * 1327104);
            float* d_images;
            float* d_settings;



            iter->second.copy_to_GPU(d_images, d_settings);
            float xmean = iter->second.calculateXmean();
            float x2mean = iter->second.calculateX2mean();
            float denominator = x2mean - (xmean * xmean);

            kernel_gaincorr_calculator<<<41472,32>>>(d_images, d_settings, d_slope, d_intercept, images, size, xmean, denominator);  //41472 * 32  = 1327104


            slope.copy_from_GPU(d_slope);
            intercept.copy_from_GPU(d_intercept);
            slope.calculate_meanvalue_on_CPU();
            std::cout << "slope mean: " << slope.getmean()<<std::endl; // DEBUG


            std::ostringstream oss;

            oss << voltage;

            std::string ending = oss.str();
            ending.append(".binf");

            std::string slopename = "slope";
            std::string slopefile =gcfolder;
            slopefile.append("/");
            slopefile.append(slopename);
            slopefile.append(ending);

            std::string interceptfile = gcfolder;
            interceptfile.append("/");

            std::string interceptname = "intercept";
            interceptfile.append(interceptname);
            interceptfile.append(ending);


           slope.writetofloatfile(slopefile);
           intercept.writetofloatfile(interceptfile);

           slopes[voltage] =slope;
           intercepts[voltage] = intercept;



            cudaFree(d_slope);
            cudaFree(d_intercept);





        }// end if (images < 10) ... else {


    } // end of for(every voltage...)

    image.clear();
    images_temp.clear();
    images.clear();
    imagemap.clear();
    intercept.clear();
    slope.clear();

    return;
}



__global__ void kernel_do_gaincorr (float* d_slope, float* d_intercept, float* d_saturation, float* d_image, float* d_settings )
{

    unsigned int tid = threadIdx.x;
    unsigned int pixel = blockIdx.x*blockDim.x + tid; //thread is computing pixel-th pixel

    //printf(" pixel: %d \t tid: %d \t blockIdx : %d \t blockDim : %d \n", pixel,tid, blockIdx.x, blockDim.x);



        d_image[pixel] = (d_image[pixel] - d_intercept[pixel] ) / d_slope[pixel]  * 16383 / *d_saturation;
    return;
}



void Gaincorr::corrigateimage(Image_cuda_compatible& image)
{
    //Rounding voltage to multiply of 5
    int voltage = (int) (round(image.getvoltage()));
    int remainder  = voltage %5;

    if(remainder != 0)

    {
        voltage = voltage + 5 - remainder;
    }

//DEBUG
    //if(saturation.find(voltage) == saturation.end())
    //{
     //   std::cout <<"Error: no calbration data found for image" <<image.getid() << "With voltage " << image.getvoltage() <<std::endl;
      //  return;
    //}

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



    float* d_saturation;
    //DEBUG
    float sat = 100000; //DEBUG !!!
    float* d_slope;
    float* d_intercept;
    d_slope = slopes.find(voltage)->second.copy_to_GPU();
    d_intercept = intercepts.find(voltage)->second.copy_to_GPU();
    float* d_image;

    d_image= image.copy_to_GPU();

    cudaMalloc( (void**)&d_saturation, sizeof(float) );
   cudaMemcpy(d_saturation, &sat, sizeof(float), cudaMemcpyHostToDevice );

   float settings = image.getamperage() * image.getexptime();
   float* d_settings;

   cudaMalloc( (void**)&d_settings, sizeof(float) );
  cudaMemcpy(d_settings, &settings, sizeof(float), cudaMemcpyHostToDevice );

    kernel_do_gaincorr<<<41472,32>>>( d_slope,  d_intercept, d_saturation,  d_image , d_settings);
    image.copy_from_GPU(d_image);







    cudaFree(d_saturation);
    cudaFree(d_settings);

}

