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

#include <cusolverSp.h>
#include <cuda_runtime_api.h>


/*   ********************************************
 *   ************* C O N S T R U C T O R ********
 *   *******************************************/

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
n=size=0;
u=0;
d_addedCoordinates=NULL;
}

Geomcorr::~Geomcorr()
{
    if(d_coordinates != NULL)
    {
        HANDLE_ERROR(cudaFree(d_coordinates));
    }

    if(d_coordinatesFromThatImage != NULL)
    {
        HANDLE_ERROR(cudaFree(d_coordinatesFromThatImage));
    }

    if(d_addedCoordinates != NULL)
    {
        HANDLE_ERROR(cudaFree(d_addedCoordinates));
    }

}

/*   ********************************************
 *   ************* K E R N E L S*****************
 *   *******************************************/



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


__global__ void kernel_addCoordinates(int* d_coordinates, int* d_coordinatesFromThatImage, int n, int u, int size, int* d_addedCoordinates, int numCols, int numRows)
{
    int i = threadIdx.x;
    if( i >=n)
    {
        printf("ERROR at kernel_addCoordinates. ThreadIdx is %d but n is only %d\n", i,n);
        return;
    }

    if(d_addedCoordinates[i] == size)
    {
        printf("ERROR at kernel_addCoordinates. Container is full at sircle number %d.\n",i);
        return;
    }



    if( d_addedCoordinates[i] ==0) // if there is no coordinate in the container
    {
        d_coordinates[i*2*size + 0] = d_coordinatesFromThatImage[i*2];
        d_coordinates[i*2*size + 1] = d_coordinatesFromThatImage[i*2 +1];
        d_addedCoordinates[i]=1;
        return;
    }
    //last x and y of the given ellipse:
    int x = d_coordinates[i*2*size + 2*(d_addedCoordinates[i]-1)];
    int y = d_coordinates[i*2*size + 2*(d_addedCoordinates[i]-1) +1 ];


    int xnew, ynew, xnearest, ynearest, dist, mindist;
    mindist = numCols* numCols + numRows * numRows;
    //searching for the nearest coordinates among d_coordinatesFromThatImage
    for(int j=0; j<n+u; j++)
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
        //nothing.
        return;
    }

    if(xnearest== x && ynearest ==y) // duplicate...
    {
        return;
    }

       d_coordinates[i*2*size + 2*(d_addedCoordinates[i])] = xnearest;
        d_coordinates[i*2*size + 2*(d_addedCoordinates[i]) +1] = ynearest;
        d_addedCoordinates[i]+=1;


    return;
}

//! Kernel to fill the gpu matrix and vector before fitting ellipse with least square fitting.

//! Matrix is stored in CSR storage format. Every thread is working on one point of the ellipse.
//! Matrix rows are: [u^2 -2u -2v +2uv 1], vector is [-v^2].


__global__ void kernel_fill_matrix(float* d_csrValA, int* csrRowPtrA, int* csrColIndA, float* d_vector,  int* d_coordinates, int addedCoordinates, int eliipseNo, int size)
{
    int i = threadIdx.x; // i=1....n. i-th thread is filling the data of the i-th point. of the n-th ellipse.
    if(addedCoordinates >= size)
    {
        printf("ERROR: Out of boundaryy at kernel_fill_matrix.\n");
        return;
    }

    if(i >=addedCoordinates )
    {
        return;
    }

    float u, v;
    u=(float) d_coordinates[2*i +2*eliipseNo*size];
    v= (float) d_coordinates[2*i+1 +2*eliipseNo*size];


    d_csrValA[5*i] = u*u;
    d_csrValA[5*i+1] = -2.0f*u;
    d_csrValA[5*i+2] = -2.0f*v;

    d_csrValA[5*i+3] = 2.0f*u*v;
    d_csrValA[5*i+4] = 1.0f;
    csrRowPtrA[i] = 5*i;
    csrColIndA[5*i] =0;
    csrColIndA[5*i +1] =1;
    csrColIndA[5*i +2] =2;
    csrColIndA[5*i +3] =3;
    csrColIndA[5*i +4] =4;


    d_vector[i] = -1.0f*v*v;

    if(i==0)
    {
        csrRowPtrA[addedCoordinates] = addedCoordinates * 5;
    }

}






/*   ********************************************
 *   ************* F U N C T I O N S ************
 *   *******************************************/


//!Extracts the center of n number of circles from to image to the GPU array *d_choordinates.
//!
//! Performs Hough transformation on the GPU, by searching for a wide range of radiuses:
//! 5<=r<=25.


void Geomcorr::extractCoordinates(Image_cuda_compatible &image, bool drawOnly, bool onlyN)
{
    if(d_coordinatesFromThatImage == NULL && drawOnly == false)
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

    int circles;
    if(onlyN)
        circles = n;
    else
        circles = n+u;

    for( int i =0; i< circles; i++)
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
    if(drawOnly)
    {
        image.drawCross(x,y);
    }
    else
        {
        image.drawCross(x,y); //cinema
        HANDLE_ERROR(cudaMemcpy(d_coordinatesFromThatImage+i*2,&x,sizeof(int),cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_coordinatesFromThatImage+i*2 +1,&y,sizeof(int),cudaMemcpyHostToDevice));
        }


    }

}


void Geomcorr::initializeDeviceVector(int n, int size, int u)
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
  HANDLE_ERROR(cudaMalloc((void**) &d_coordinatesFromThatImage,2*(n+u)*sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**) &d_addedCoordinates,n*sizeof(int)));

  HANDLE_ERROR(cudaMemset(d_coordinates,0,n*size*2*sizeof(int)));
  HANDLE_ERROR(cudaMemset(d_coordinatesFromThatImage,0,2*(n +u)*sizeof(int)));
  HANDLE_ERROR(cudaMemset(d_addedCoordinates,0,n*sizeof(int)));



  this->n=n;
  this->u = u;
  this->size =size;
}

void Geomcorr::addCoordinates()
{


    if(d_coordinates == NULL || d_coordinatesFromThatImage ==NULL)
    {
        std::cout <<"ERROR: Got a null pointer at geomcorr::addCoordinates(). Please initialize first! " <<std::endl;
        return;
    }



    kernel_addCoordinates<<<1,n>>>( d_coordinates, d_coordinatesFromThatImage, n, u, size, d_addedCoordinates, Image_cuda_compatible::width, Image_cuda_compatible::height);

}


void Geomcorr::exportText(std::string filename)
{
    int *coordinates = new int[2*n*size];
    int* addedCoordinates = new int[n];
    cudaMemcpy(coordinates,d_coordinates,n*2*size*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(addedCoordinates,d_addedCoordinates,n*sizeof(int),cudaMemcpyDeviceToHost);

    std::ofstream file;
    file.open(filename);
    for( int i=0;i<n;i++)
    {
        file << "x\ty" << std::endl;

        for( int j=0; j<addedCoordinates[i]; j++)
        {
            file << coordinates[2*i*size + 2*j]<<"\t" << coordinates[2*i * size + 2*j+1] << std::endl;
        }
    }
    file.close();
    delete[] coordinates;

}

void Geomcorr::fitEllipse(int i, float* a, float* b, float* c, float* u, float* v, float* error)
{




    int addedCoordinates = 0;
    HANDLE_ERROR(cudaMemcpy(&addedCoordinates,d_addedCoordinates+i,sizeof(int),cudaMemcpyDeviceToHost));

    if(addedCoordinates < 10)
    {
        std::cout << "ERROR! There are only " << addedCoordinates <<" number of points at ellipse " << i << std::endl;
        *a=*b=*c=*u=*v=0;
        *error = 50000000000.0f;
        return;
    }

    //See Cuda documentation [Cusolver Library]
    cusolverSpHandle_t handle;
    cusolverStatus_t  status;
    status = cusolverSpCreate(&handle);
    if(status != CUSOLVER_STATUS_SUCCESS)
    {
        std::cout << "ERROR: Cusolver cannot be initialized." << std::endl;
        *a=*b=*c=*u=*v=0;
        *error = 50000000000.0f;
        return;
    }
    cusparseMatDescr_t descr = NULL;
    cusparseStatus_t csp;
    csp = cusparseCreateMatDescr(&descr);
    if(csp != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "ERROR: Cusolver cannot be initialized." << std::endl;
        *a=*b=*c=*u=*v=0;
        *error = 50000000000.0f;
        return;
    }
    csp = cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    if(csp != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "ERROR: Cusolver cannot be initialized." << std::endl;
        *a=*b=*c=*u=*v=0;
        *error = 50000000000.0f;
        return;
    }
    csp = cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
    if(csp != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "ERROR: Cusolver cannot be initialized." << std::endl;
        *a=*b=*c=*u=*v=0;
        *error = 50000000000.0f;
        return;
    }

    std::cout << "Fitting ellipse " << i << " with "<< addedCoordinates << "number of coordinates." << std::endl;





    float* d_csrValA; //[u^2 -2u -2v +2uv 1] in addedCoordinates row.
    float* d_vector; // -v^2
    int* d_csrRowPtrA;
    int* d_csrColIndA;



    HANDLE_ERROR(cudaMalloc((void**)&d_csrValA,sizeof(float) * 5 *addedCoordinates) );
    HANDLE_ERROR(cudaMalloc((void**)&d_vector,sizeof(float)  * addedCoordinates ));
    HANDLE_ERROR(cudaMalloc((void**)&d_csrColIndA,sizeof(int) *5 *addedCoordinates) );
    HANDLE_ERROR(cudaMalloc((void**)&d_csrRowPtrA,sizeof(int) * (addedCoordinates +1) ));



    kernel_fill_matrix<<<((addedCoordinates + 1023) / 1024) ,1024 >>>(d_csrValA,d_csrRowPtrA,d_csrColIndA, d_vector, d_coordinates, addedCoordinates, i, size);


    int rankA =0;
    float min_norm =0;
    float tol = 0;    // Does not have a clue what this value should be...
    float* x = (float*)malloc(sizeof(float)* 5);
    int* p =(int*) malloc(sizeof(int) * 5);

    float* csrValA = (float*) malloc(sizeof(float) * 5 * addedCoordinates);
    HANDLE_ERROR(cudaMemcpy(csrValA,d_csrValA,sizeof(float) * 5 * addedCoordinates,cudaMemcpyDeviceToHost));

    int* csrRowPtrA = (int*) malloc ( sizeof(int) * (addedCoordinates + 1));
    HANDLE_ERROR(cudaMemcpy(csrRowPtrA,d_csrRowPtrA,sizeof(int) *(addedCoordinates+1),cudaMemcpyDeviceToHost));

    int* csrColIndA = (int*) malloc( sizeof(int) * 5 * addedCoordinates);
    HANDLE_ERROR(cudaMemcpy(csrColIndA,d_csrColIndA,sizeof(int) *addedCoordinates *5,cudaMemcpyDeviceToHost));


    float* vector = (float*) malloc ( sizeof(float) * addedCoordinates);
    HANDLE_ERROR(cudaMemcpy(vector,d_vector,sizeof(float)  * addedCoordinates,cudaMemcpyDeviceToHost));






    status  = cusolverSpScsrlsqvqrHost(handle,addedCoordinates,5,addedCoordinates*5,descr,csrValA, csrRowPtrA,csrColIndA,vector,tol, &rankA, x, p, &min_norm );


    if(status!=CUSOLVER_STATUS_SUCCESS)
    {
        std::cout << "ERROR! Ellipse fit failed." << std::endl;
        *a=*b=*c=*u=*v=0;
        *error = 50000000000.0f;
    }
    else
    {
        std::cout << " rankA : " << rankA <<std::endl;
        std::cout << "min norm: " << min_norm << std::endl;

        std::cout << std::endl << std::endl;
        *u =  (float) (x[1] - ( x[2] * x[3] ) ) / (float) (x[0] - x[3] * x[3]);
        *v  = (float) (x[0] * x[2] - x[1] * x[3] ) / (float) (x[0] - x[3] * x[3]);
        *a = (float) x[0] / (float) (x[0] * (*u) * (*u) + (*v) * (*v) + 2 * x[3] * (*u) * (*v)  - x[4]);
        *b = (*a) / (float) x[0];
        *c = x[3] * (*b);


        std::cout << "u =" << *u ;
        std::cout << "v =" << *v ;
        std::cout << "a =" << sqrt(1/(*a)) ;
        std::cout << "b =" << sqrt(1/(*b)) ;
        std::cout << "c =" << *c ;

        std::cout << std::endl << std::endl ;




    }





    cudaDeviceSynchronize();

    cusolverSpDestroy(handle);

    cudaFree(d_csrValA);
    cudaFree(d_vector);
    cudaFree(d_csrColIndA);
    cudaFree(d_csrRowPtrA);
    free( x);
    free(p);
    free(csrValA);
    free(csrRowPtrA);
    free(csrColIndA);
    free(vector);
}

