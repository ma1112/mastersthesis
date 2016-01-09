#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gaincorr.h"





__global__ void kernel_gaincorr_calculator(float* d_images, float* d_settings, float* d_slope, float* d_intercept, int n, float xmean, float denominator)
{
    __shared__ float y[32];
    __shared__ float xy[32];


    unsigned int tid = threadIdx.x;
    unsigned int pixel = blockIdx.x*blockDim.x + tid; //thread is computing pixel-th pixel slope and intercept.



            xy[tid] = 0;
            y[tid] = 0;
            int gloffset = pixel * n;

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

            kernel_gaincorr_calculator<<<41472,32>>>(d_images, d_settings, d_slope, d_intercept, images, xmean, denominator);  //41472 * 32  = 1327104


            slope.copy_from_GPU(d_slope);
            intercept.copy_from_GPU(d_intercept);
            slope.calculate_meanvalue_on_CPU();
            std::cout << "slope mean: " << slope.getmean()<<std::endl;


            std::ostringstream oss;

            oss << iter->first;

            std::string ending = oss.str();
            ending.append(".binf");
            std::string slopename = "slope";
            slopename.append(ending);
            std::string interceptname = "intercept";
            interceptname.append(ending);


           slope.writetofloatfile(slopename);
           intercept.writetofloatfile(interceptname);



            cudaFree(d_slope);
            cudaFree(d_intercept);



        }// end if (images < 10) ... else {


    }
    return;
}


