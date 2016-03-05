#include "geomcorr.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include"book.cuh"
#include"math_constants.h"
#include "thrust/extrema.h"
#include "thrust/device_vector.h"
#include"thrust/device_ptr.h"
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <iostream>
#include <fstream>


Geomcorr::Geomcorr()
{

HANDLE_ERROR (cudaMalloc( (void**)&d_filter, 81* sizeof(float)));
float filter[81] = {14.7107721f, 13.3313922f, 10.91566001f, 8.588367701f, 7.62541433f, 8.588367701f, 10.91566001f, 13.3313922f, 14.7107721f,
                    13.3313922f, 9.452580065f, 4.056226896f, -0.747480748f, -2.678033129f, -0.747480748f, 4.056226896f, 9.452580065f, 13.3313922f,
                    10.91566001f, 4.056226896f, -4.795873011f, -12.42974587f, -15.45925025f, -12.42974587f, -4.795873011f, 4.056226896f, 10.91566001f,
                    8.588367701f, -0.747480748f, -12.42974587f, -22.36392898f, -26.28366654f, -22.36392898f, -12.42974587f, -0.747480748f, 8.588367701f,
                    7.62541433f, -2.678033129f, -15.45925025f, -26.28366654f, -30.54741993f, -26.28366654f, -15.45925025f, -2.678033129f, 7.62541433f,
                    8.588367701f, -0.747480748f, -12.42974587f, -22.36392898f, -26.28366654f, -22.36392898f, -12.42974587f, -0.747480748f, 8.588367701f,
                    10.91566001f, 4.056226896f, -4.795873011f, -12.42974587f, -15.45925025f, -12.42974587f, -4.795873011f, 4.056226896f, 10.91566001f,
                    13.3313922f, 9.452580065f, 4.056226896f, -0.747480748f, -2.678033129f, -0.747480748f, 4.056226896f, 9.452580065f, 13.3313922f,
                    14.7107721f, 13.3313922f, 10.91566001f, 8.588367701f, 7.62541433f, 8.588367701f, 10.91566001f, 13.3313922f, 14.7107721f


 };
HANDLE_ERROR (cudaMemcpy(d_filter,filter,81* sizeof(float),cudaMemcpyHostToDevice));

d_coordinates = NULL;
d_coordinatesFromThatImage = NULL;
n=size=addedCoordinates=0;
}

Geomcorr::~Geomcorr()
{
    if(d_coordinates != NULL)
    {
        HANDLE_ERROR(cudaFree(d_coordinates));
    }

}


//Kernel written like it is suggested in
// https://www.nvidia.com/content/nvision2008/tech_presentations/Game_Developer_Track/NVISION08-Image_Processing_and_Video_with_CUDA.pdf

__global__ void kernel_convolve_image(const float* image, float* out, const float* kernel,const int  kernelRadius   , const int numCols, const int numRows)
{

    __shared__ float smem[(16+2*4)*(16+2*4)];
    int x = blockIdx.x * 16 + threadIdx.x - kernelRadius;
    int y = blockIdx.y * 16 + threadIdx.y - kernelRadius;

    x = max(0,x);
    x=min(x,numCols-1);
    y=max(y,0);
    y=min(y,numRows - 1);
    unsigned int pixel = y*numCols + x;
    unsigned int bindex = threadIdx.y * blockDim.y + threadIdx.x;
    // each thread copies its pixel of the block to the shared memory
    smem[bindex]  = image[pixel];
    __syncthreads();

//only threads inside the apron will write results

if((threadIdx.x >= kernelRadius) && (threadIdx.x < 16 + kernelRadius )
        && (threadIdx.y >= kernelRadius) && (threadIdx.y < 16+kernelRadius))
    {
        float sum = 0.0f;
        int kernelOffset = kernelRadius*((2*kernelRadius)+2);
        for( int dy = -kernelRadius; dy<=kernelRadius; dy++)
        {
            for(int dx = -kernelRadius; dx <= kernelRadius; dx++)
            {
                sum += (smem[bindex + (dy*blockDim.x) + dx] *
                        kernel[dx + dy*(2*kernelRadius+1) + kernelOffset]) ;

            }
        }

        //if(abs(sum)>100000)
        {
            out[pixel] = sum ;
        }
       // else
       // {
       //     out[pixel] = 0;
       // }
    }
}


//! Kernel that performs hough tranformation of a grayscale image on the GPU.
//!
//! Hough transforms *image  ( that has size numCols by numRows) by searching for circles
//! with r radius. Also preforms grayscale to binary conversion, as pixels with intensity greater than
//! threshold are considered as whites. Hough transform is saved to image *out.

__global__ void kernel_hough_transform(float* image, float* out, int r, int numCols, int numRows , float threshold)
{
    int x0 = blockIdx.x * 16+ threadIdx.x;
    int y0 = blockIdx.y * 16 + threadIdx.y;



    if(image[x0+y0*numCols] > threshold)
    {

        int x, y;

        // Draw a circle with Midpoint Circle Algorythm (Integer based version)
        //Note: Every thread reaches exactly one point at a time, every thread reaches another. ( No need to use AtomicAdd)
        x= r;
        y=0;
        int decisionOver2 = 1 - x;
        while(y<=x)
        {
            if(x+x0>=0 && x+x0<numCols && y+y0 >=0 && y+y0 <numRows)
                    out[x + x0 + ( y + y0) * numCols] +=1; // Octant 1
            if(y+x0>=0 && y+x0<numCols && x+y0 >=0 && x+y0 <numRows)
                    out[y + x0 + ( x + y0) * numCols] +=1; // Octant 2
            if(-x+x0>=0 && -x+x0<numCols && y+y0 >=0 && y+y0 <numRows)
                    out[-x + x0 +( y + y0) * numCols] +=1; // Octant 4
            if(-y+x0>=0 && -y+x0<numCols && x+y0 >=0 && x+y0 <numRows)
                    out[-y + x0 +( x + y0) * numCols] +=1; // Octant 3
            if(-x+x0>=0 && -x+x0<numCols && -y+y0 >=0 && -y+y0 <numRows)
                    out[-x + x0 +(-y + y0) * numCols] +=1; // Octant 5
            if(-y+x0>=0 && -y+x0<numCols && -x+y0 >=0 && -x+y0 <numRows)
                    out[-y + x0+(-x + y0) * numCols]+=1; // Octant 6
            if(x+x0>=0 && x+x0<numCols && -y+y0 >=0 && -y+y0 <numRows)
                    out[x + x0 +(-y + y0) * numCols]+=1; // Octant 7
            if(y+x0>=0 && y+x0<numCols && -x+y0 >=0 && -x+y0 <numRows)
                    out[y + x0 +(-x + y0) * numCols] +=1; // Octant 8

                y++;
                if (decisionOver2<=0)
                {
                  decisionOver2 += 2 * y + 1;   // Change in decision criterion for y -> y+1
                }
                else
                {
                  x--;
                  decisionOver2 += 2 * (y - x) + 1;   // Change for y -> y+1, x -> x-1
                }
        }



    }

}

__global__ void kernel_zero_crossing_extractor(float* image, float* out, int numCols, int numRows, float threshold)
{
    int x = blockIdx.x * blockDim.x+ threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pixel = y*numCols + x;

    //edge of the picture is ignored ( cannot compare with values out of the picture)
    if( x ==0 || y ==0 || x == numCols-1 || y == numRows-1)
    {
        out[pixel] = 0.0f;
        return;
    }


    // check if it is a zero crossing
    //check zero crossing if value is higher than the threshold ( to avoid noise fluctuation)
    // left <-> right

    if( (image[pixel] > threshold && image[pixel-1] < -threshold )
            || ( image[pixel ]< -threshold && image[pixel-1] > threshold))
    {
        out[pixel] = 100.0f;
    }
    //up<->down
    else if ((image[pixel] > threshold && image [pixel-numCols] < -threshold)
             || (image[pixel] < -threshold && image [pixel-numCols] > threshold) )
    {
        out [pixel] = 100.0f;
    }
    else
    {
        out[pixel] = 0.0f;
    }
    return;

}

/* NOT USED ANYMORE.
__global__ void kernel_local_maximum_extractor(float* formerImage, float* newImage, int numCols, int numRows, int temp, int r)
{
    int x = blockIdx.x * blockDim.x+ threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pixel = y*numCols + x;

    //edge of the picture is ignored ( cannot compare with values out of the picture)
    if( x ==0 || y ==0 || x == numCols-1 || y == numRows-1)
    {
        return;
    }
    //Checking if pixel is a local maxima on former image.

    if((formerImage[pixel] - newImage[pixel] < 12.0f)|| (formerImage[pixel] < 18.0f))
        // if it is a middle of a circle, Intensity on Hough transform should drop ...
              // Intensity should be at least 18 in the middle of a circle
    {
        return;
    }

    //Inspecting former picture ( supposely pixel value is a local maxima.
    //Inspecting new image. (supposely pixel value is a local minima and the former maximum is hogher than maximum on this image.)

    float formerEnviromnentMax =0;
    float newEnvironmentMax=0;
    float newEnvironmentMin=1000000;

    float formerValue,newValue;
    for(int dx = -1; dx <=1; dx++)
    {
        for(int dy = -1; dy <= 1; dy++)
        {
            if(dx || dy) // if dx !=0 and dy!=0, so given pixel is not in the middle...
            {
                formerValue = formerImage[pixel + dx - numCols * dy];
                newValue = newImage[pixel + dx - numCols * dy];
                if(formerValue >formerEnviromnentMax )
                {
                    formerEnviromnentMax = formerValue;
                }
                if(newValue > newEnvironmentMax)
                {
                    newEnvironmentMax = newValue;
                }
                if  ( newValue < newEnvironmentMin)
                {
                    newEnvironmentMin = newValue;
                }
            }
        }
    }

    formerValue = formerImage[pixel];
    newValue = newImage[pixel];
    //I Consider a point as a middle of a circle ( on the former image), if the pixel value was a local maximum, it is bigget then the new local maximum, and is now a local minimum.

    if(formerValue > formerEnviromnentMax +1 && formerValue > newEnvironmentMax *1.7f && newValue <=newEnvironmentMin )
        printf("Image %d, r= %d, Maximum value at x=%d, y= %d\n",temp,r,x,y);
    return;
}
*/

__global__ void kernel_vertical_line_remover(float* image, float* out, int numCols, int numRows)
{
    int x = blockIdx.x * blockDim.x+ threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pixel = y*numCols + x;

    //edge of the picture is ignored ( cannot compare with values out of the picture)
    if( x <2 || y <2 || x == numCols-2 || y == numRows-2)
    {
        out[pixel] = 0.0f;
        return;
    }

    if( image[pixel] <1)
    {
        out[pixel] = 0;
        return;
    }

    int pixels = 0;
    for( int i= -2; i<=2; i++)
    {
        if(image[pixel - i * numCols] >1 )
            pixels++;
    }


    // if part of a line
    if ( pixels >2 )
    {
        out[pixel] =0;
        return;
    }

        out[pixel] =100;
        return;


}


//! Adds the nearest point to every ellipse from the given coordinates.


__global__ void kernel_addCoordinates(int* d_coordinates, int* d_coordinatesFromThatImage, int n, int size, int addedCoordinates, int numCols, int numRows)
{
    int i = threadIdx.x;
    if( i >=n)
    {
        printf("ERROR at kernel_addCoordinates. ThreadIdx is %d but n is only %d\n", i,n);
        return;
    }


    if( addedCoordinates ==0) // if there is no coordinate in the container
    {
        d_coordinates[i*2*size + 0] = d_coordinatesFromThatImage[i*2];
        d_coordinates[i*2*size + 1] = d_coordinatesFromThatImage[i*2 +1];
        return;
    }
    //last x and y of the given ellipse:
    int x = d_coordinates[i*2*size + 2*(addedCoordinates-1)];
    int y = d_coordinates[i*2*size + 2*(addedCoordinates-1) +1 ];


    int xnew, ynew, xnearest, ynearest, dist, mindist;
    mindist = numCols* numCols + numRows * numRows;
    //searching for the nearest coordinates among d_coordinatesFromThatImage
    for(int j=0; j<n; j++)
    {
        xnew = d_coordinatesFromThatImage[j*2];
        ynew = d_coordinatesFromThatImage[j*2 +1];
        dist = (xnew - x) * (xnew -x) + (ynew-y) * (ynew -y);
        if( dist < mindist)
        {
            xnearest = xnew;
            ynearest = ynew;
            mindist = dist;
        }

    }


    if( (ynearest < 20 || ynearest  > numRows - 20 || xnearest < 20 || xnearest  > numCols -20) // last coordinate is at the edge of the picture -> the ball is likely get off the picture
            ||(abs(ynearest - y) > 10 || ( abs(xnearest - x) > 50))) // outlier
    {
        d_coordinates[i*2*size + 2*(addedCoordinates)] = x;
        d_coordinates[i*2*size + 2*(addedCoordinates) +1] = y;
    }
    else
    {   d_coordinates[i*2*size + 2*(addedCoordinates)] = xnearest;
        d_coordinates[i*2*size + 2*(addedCoordinates) +1] = ynearest;
    }

    return;
}



//!Extracts the center of n number of circles from to image to the GPU array *d_choordinates.
//!
//! Performs Hough transformation on the GPU, by searching for a wide range of radiuses:
//! 5<=r<=25.

void Geomcorr::extractCoordinates(Image_cuda_compatible &image)
{
    if(d_coordinatesFromThatImage == NULL)
    {
        std::cout <<"ERROR! initialize gaomcorr container first." << std::endl;
    }

    const dim3 blockSize(16 + filterWidth-1 , 16 + filterWidth-1);
    const dim3 gridSize((image.width + 15) / 16  , (image.height + 15 ) / 16);

    Image_cuda_compatible convolvedImage,transformedImage, zeroCrossImage, cleanZeroCrossing,maxImg;
    transformedImage.reserve_on_GPU();
    maxImg.reserve_on_GPU();
    maxImg.clear();



    kernel_convolve_image<<<gridSize,blockSize>>>(image.gpu_im,convolvedImage.reserve_on_GPU(),d_filter,4,image.width,image.height);

    convolvedImage.calculate_meanvalue_on_GPU();
    const dim3 newblockSize(16,16);
    const dim3 newgridSize(96,54);

   kernel_zero_crossing_extractor<<<newgridSize,newblockSize>>>(convolvedImage.reserve_on_GPU(), zeroCrossImage.reserve_on_GPU(), zeroCrossImage.width, zeroCrossImage.height, convolvedImage.getstdev() * 0.5f);

   kernel_vertical_line_remover<<<newgridSize,newblockSize>>>(zeroCrossImage.reserve_on_GPU(),cleanZeroCrossing.reserve_on_GPU(), cleanZeroCrossing.width,cleanZeroCrossing.height);


    //Executing Hough tranformations with multiple r values
   int rMin = 5;
   int rMax = 24;

   //r=5 here.

    for( int i=rMin; i<=rMax;i++)
    {
       //printf("Image %s, r= %d\n\n", image.getid().c_str(), i);


            transformedImage.clear();
            kernel_hough_transform<<<newgridSize,newblockSize>>>(cleanZeroCrossing.reserve_on_GPU(), transformedImage.reserve_on_GPU(), i, image.width, image.height , 1);
            char str[3];
            sprintf(str,"%d",i);

           // transformedImage.writetofloatfile("C:/awing/hough/proba/"  + image.getid() + "_r_" + str +".binf");
           // transformedImage.saveAsJPEG("C:/awing/hough/proba/"  + image.getid() + "_r_" + str +".jpg");
            maxImg.equalmax(transformedImage);




    //kernel_local_maximum_extractor<<<newgridSize,newblockSize>>>(transformedImage.reserve_on_GPU(), image.width, image.height);

    }

    //maxImg.saveAsJPEG("C:/awing/hough/proba/"  + image.getid() + "_max_.jpg");
    //maxImg.writetofloatfile("C:/awing/hough/proba/"  + image.getid() + "_max_.binf");
    //Extracting maximums

    for( int i =0; i< n; i++)
    {
    thrust::device_ptr<float> d_ptr(maxImg.gpu_im);
    //thrust::device_vector<float> d_vec(d_ptr, d_ptr + 100);
    // Tried with device_vector, does not work for me ... ( on my laptop).
    thrust::device_vector<float>::iterator iter =
      thrust::max_element(d_ptr, d_ptr+maxImg.size);

    unsigned int position = &(*iter) - d_ptr;
    float max_val = *iter;

    //std::cout  <<" The maximum value is " << max_val << " at position: x " << position % maxImg.width <<" y: " << position / maxImg.width << std::endl;

    maxImg.clearwitinradius(position % maxImg.width, position / maxImg.width , 2*rMax+1 );

    int x= position % maxImg.width;
    int y = position / maxImg.width;
    HANDLE_ERROR(cudaMemcpy(d_coordinatesFromThatImage+i*2,&x,sizeof(int),cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_coordinatesFromThatImage+i*2 +1,&y,sizeof(int),cudaMemcpyHostToDevice));


    }
    //image.copy_GPU_image(d_ezt);

}


void Geomcorr::initializeDeviceVector(int n, int size)
{
    if(d_coordinates != NULL)
    {
        HANDLE_ERROR(cudaFree(d_coordinates));
    }
    if(d_coordinatesFromThatImage != NULL)
    {
        HANDLE_ERROR(cudaFree(d_coordinatesFromThatImage));
    }

  HANDLE_ERROR( cudaMalloc((void**)&d_coordinates,n*size*2*sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**) &d_coordinatesFromThatImage,2*n*sizeof(int)));
  HANDLE_ERROR(cudaMemset(d_coordinates,0,n*size*2*sizeof(int)));
  HANDLE_ERROR(cudaMemset(d_coordinatesFromThatImage,0,2*n*sizeof(int)));
  addedCoordinates =0;
  this->n=n;
  this->size =size;
}

void Geomcorr::addCoordinates()
{


    if(d_coordinates == NULL || d_coordinatesFromThatImage ==NULL)
    {
        std::cout <<"ERROR: Got a null pointer at geomcorr::addCoordinates(). Please initialize first! " <<std::endl;
        return;
    }
    if(addedCoordinates == size)
    {
        std::cout<<"ERROR: geomcorr containter is full. (size = " << size <<std::endl;
        return;
    }


    kernel_addCoordinates<<<1,n>>>( d_coordinates, d_coordinatesFromThatImage, n, size, addedCoordinates, Image_cuda_compatible::width, Image_cuda_compatible::height);
    addedCoordinates++;
}


void Geomcorr::exportText(std::string filename)
{
    int *coordinates = new int[2*n*size];
    cudaMemcpy(coordinates,d_coordinates,n*2*size*sizeof(int),cudaMemcpyDeviceToHost);
    std::ofstream file;
    file.open(filename);
    for( int i=0;i<n;i++)
    {
        file << "x\ty" << std::endl;

        for( int j=0; j<addedCoordinates; j++)
        {
            file << coordinates[2*i*size + 2*j]<<"\t" << coordinates[2*i * size + 2*j+1] << std::endl;
        }
    }
    file.close();
    delete[] coordinates;

}
