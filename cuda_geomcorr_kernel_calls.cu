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
filterWidth = 9;
eta = 0.0;
HANDLE_ERROR (cudaMalloc( (void**)&d_filter, 81* sizeof(float)));
float filter[81] = {

    /*14.7107721f, 13.3313922f, 10.91566001f, 8.588367701f, 7.62541433f, 8.588367701f, 10.91566001f, 13.3313922f, 14.7107721f,
                    13.3313922f, 9.452580065f, 4.056226896f, -0.747480748f, -2.678033129f, -0.747480748f, 4.056226896f, 9.452580065f, 13.3313922f,
                    10.91566001f, 4.056226896f, -4.795873011f, -12.42974587f, -15.45925025f, -12.42974587f, -4.795873011f, 4.056226896f, 10.91566001f,
                    8.588367701f, -0.747480748f, -12.42974587f, -22.36392898f, -26.28366654f, -22.36392898f, -12.42974587f, -0.747480748f, 8.588367701f,
                    7.62541433f, -2.678033129f, -15.45925025f, -26.28366654f, -30.54741993f, -26.28366654f, -15.45925025f, -2.678033129f, 7.62541433f,
                    8.588367701f, -0.747480748f, -12.42974587f, -22.36392898f, -26.28366654f, -22.36392898f, -12.42974587f, -0.747480748f, 8.588367701f,
                    10.91566001f, 4.056226896f, -4.795873011f, -12.42974587f, -15.45925025f, -12.42974587f, -4.795873011f, 4.056226896f, 10.91566001f,
                    13.3313922f, 9.452580065f, 4.056226896f, -0.747480748f, -2.678033129f, -0.747480748f, 4.056226896f, 9.452580065f, 13.3313922f,
                    14.7107721f, 13.3313922f, 10.91566001f, 8.588367701f, 7.62541433f, 8.588367701f, 10.91566001f, 13.3313922f, 14.7107721f*/

    0.00033147f , 0.001484032f , 0.004053273f , 0.007087544f , 0.008447815f , 0.007087544f , 0.004053273f , 0.001484032f , 0.00033147f,
    0.001484032f , 0.005911549f , 0.013649965f , 0.019647961f , 0.021186881f , 0.019647961f , 0.013649965f , 0.005911549f , 0.001484032f,
    0.004053273f , 0.013649965f , 0.021961039f , 0.012496394f , 0.001194649f , 0.012496394f , 0.021961039f , 0.013649965f , 0.004053273f,
    0.007087544f , 0.019647961f , 0.012496394f , -0.04775627f , -0.093734924f , -0.04775627f , 0.012496394f , 0.019647961f , 0.007087544f,
    0.008447815f , 0.021186881f , 0.001194649f , -0.093734924f , -0.162403003f , -0.093734924f , 0.001194649f , 0.021186881f , 0.008447815f,
    0.007087544f , 0.019647961f , 0.012496394f , -0.04775627f , -0.093734924f , -0.04775627f , 0.012496394f , 0.019647961f , 0.007087544f,
    0.004053273f , 0.013649965f , 0.021961039f , 0.012496394f , 0.001194649f , 0.012496394f , 0.021961039f , 0.013649965f , 0.004053273f,
    0.001484032f , 0.005911549f , 0.013649965f , 0.019647961f , 0.021186881f , 0.019647961f , 0.013649965f , 0.005911549f , 0.001484032f,
    0.00033147f , 0.001484032f , 0.004053273f , 0.007087544f , 0.008447815f , 0.007087544f , 0.004053273f , 0.001484032f , 0.00033147f



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


__global__ void kernel_addCoordinates(int* d_coordinates, int* d_coordinatesFromThatImage, float* d_image, int n, int u, int size, int* d_addedCoordinates, int numCols, int numRows)
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

    if (!( (ynearest < 10 || ynearest  > numRows - 10 || xnearest < 10 || xnearest  > numCols -10) // last coordinate is at the edge of the picture -> the ball is likely get off the picture
            ||(abs(ynearest - y) > 15 || ( abs(xnearest - x) > 100)))) // NOT outlier
    {
        d_coordinates[i*2*size + 2*(d_addedCoordinates[i])] = xnearest;
         d_coordinates[i*2*size + 2*(d_addedCoordinates[i]) +1] = ynearest;
         d_addedCoordinates[i]+=1;
        return;
    }

    else // no global maxima near -> search for local maxima on image.
    {
        //average of the last 2 coordinates:
        int xAvg = (int) round( x * 0.5f + d_coordinates[i*2*size + 2*(d_addedCoordinates[i]-2)] * 0.5f);
        int yAvg = (int) round( y * 0.5f + d_coordinates[i*2*size + 2*(d_addedCoordinates[i]-2) +1 ] * 0.5f);


        if( d_image == NULL)
        {
            printf( "ERROR: there is no image loaded in kernel_addCoordinates \n");
            d_coordinates[i*2*size + 2*(d_addedCoordinates[i])] = x;
             d_coordinates[i*2*size + 2*(d_addedCoordinates[i]) +1] = y;
             d_addedCoordinates[i]+=1;
             return;
        }

        float localmax = 0;
        float xLocal = 0;
        float yLocal = 0;
        for(int xx = max(0,(xAvg-8)); xx< min((xAvg+8),numCols); xx++)
        {
            for(int yy = max(0,yAvg-3); yy < min ( yAvg+3, numRows); yy++)
            {
                if(d_image[xx + yy*numCols] > localmax )
                {
                    localmax = d_image[xx + yy*numCols];
                    xLocal = xx;
                    yLocal = yy;
                }

            }
        }
        if ( xLocal > 0 && yLocal > 0)
        {

            d_coordinates[i*2*size + 2*(d_addedCoordinates[i])] = xLocal;
             d_coordinates[i*2*size + 2*(d_addedCoordinates[i]) +1] = yLocal;
             d_addedCoordinates[i]+=1;
             return;
        }
        else
        {
            d_coordinates[i*2*size + 2*(d_addedCoordinates[i])] = xAvg;
             d_coordinates[i*2*size + 2*(d_addedCoordinates[i]) +1] = yAvg;
             d_addedCoordinates[i]+=1;
        }
    }

    return;
}

//! Kernel to fill the gpu matrix and vector before calculating eta.

//! Matrix is stored in CSR storage format. Every thread is working on an ellipse pair.
//! Matrix rows are: [u_j - u_i;  v_j - v_i; ], vector is [v_i * u_j - v_j * u_i].


__global__ void kernel_fill_eta_matrix(float* d_csrValA, int* csrRowPtrA, int* csrColIndA, float* d_vector,  int* d_coordinates, int addedCoordinates, int eliipseNo, int size)
{
    int i = threadIdx.x ; // i=1....n. i-th thread is filling the data of the i-th point. of the n-th ellipse.
    int N = addedCoordinates;
    int Nover2 = (N%2 ==0)? N/2-1 : (N-1)/2;
    if(addedCoordinates > size)
    {
        printf("ERROR: Out of boundaryy at kernel_fill_matrix. addedCoordinates is %d and size is %d \n",addedCoordinates, size);
        return;
    }

    if(i >= Nover2)
    {
        return;
    }

    float uj, vj;
    float ui, vi;
    ui=(float) d_coordinates[2*i +2*eliipseNo*size];
    vi= (float) d_coordinates[2*i+1 +2*eliipseNo*size];

    uj=(float) d_coordinates[2*(i + Nover2)  +2*eliipseNo*size];
    vj= (float) d_coordinates[2*(i + Nover2)+1 +2*eliipseNo*size];


    d_csrValA[2*i] = uj-ui;
    d_csrValA[2*i+1] = -(vj-vi);

    csrRowPtrA[i] = 2*i;

    csrColIndA[2*i] =0;
    csrColIndA[2*i +1] =1;



    d_vector[i] = vi*uj - vj*ui;

    if(i==0)
    {
        csrRowPtrA[Nover2 +1] = ( Nover2 +1) * 2;
    }

}



//! Kernel to fill the gpu matrix and vector before fitting ellipse with least square fitting.

//! Matrix is stored in CSR storage format. Every thread is working on one point of the ellipse.
//! Matrix rows are: [u^2 -2u -2v +2uv 1], vector is [-v^2].


__global__ void kernel_fill_matrix(double eta, float* d_csrValA, int* csrRowPtrA, int* csrColIndA, float* d_vector,  int* d_coordinates, int addedCoordinates, int eliipseNo, int size)
{
    int i = threadIdx.x; // i=1....n. i-th thread is filling the data of the i-th point. of the n-th ellipse.
    if(addedCoordinates > size)
    {
        printf("ERROR: Out of boundaryy at kernel_fill_matrix.\n");
        return;
    }

    if(i >=addedCoordinates )
    {
        return;
    }

    float u0, v0, u ,v;
    u0=(float) d_coordinates[2*i +2*eliipseNo*size];
    v0= (float) d_coordinates[2*i+1 +2*eliipseNo*size];

    //Corrigating with eta:

    u = u0 * cos(eta) - v0 * sin(eta);
    v = u0 * sin(eta) + v0 * cos(eta);


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

    //convolvedImage.saveAsJPEG("C:/awing/hough/kozeli/" + image.getid()  +"_convolved.jpg");
    //convolvedImage.writetofloatfile("C:/awing/hough/kozeli/" + image.getid()  +"_convolved.binf");
    std::string id = image.getid();


    const dim3 newblockSize(16,16);
    const dim3 newgridSize(96,54);

   kernel_zero_crossing_extractor<<<newgridSize,newblockSize>>>(convolvedImage.reserve_on_GPU(), zeroCrossImage.reserve_on_GPU(), zeroCrossImage.width, zeroCrossImage.height, convolvedImage.getstdev() * 0.5f);


   //zeroCrossImage.saveAsJPEG("C:/awing/hough/kozeli/" + image.getid()  +"_zerocross.jpg");
   //zeroCrossImage.writetofloatfile("C:/awing/hough/kozeli/" + image.getid()  +"_zerocross.binf");



   //DEBUG


   kernel_vertical_line_remover<<<newgridSize,newblockSize>>>(zeroCrossImage.reserve_on_GPU(),cleanZeroCrossing.reserve_on_GPU(), cleanZeroCrossing.width,cleanZeroCrossing.height);

   //cleanZeroCrossing.saveAsJPEG("C:/awing/hough/kozeli/" + image.getid()  +"_cleanzerocross.jpg");
   //cleanZeroCrossing.writetofloatfile("C:/awing/hough/kozeli/" + image.getid()  +"_cleanzerocross.binf");


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

           // transformedImage.writetofloatfile("C:/awing/hough/kozeli/"  + image.getid() + "_r_" + str +".binf");
           // transformedImage.saveAsJPEG("C:/awing/hough/kozeli/"  + image.getid() + "_r_" + str +".jpg");
            maxImg.equalmax(transformedImage);






    //kernel_local_maximum_extractor<<<newgridSize,newblockSize>>>(transformedImage.reserve_on_GPU(), image.width, image.height);

    }

    if(!drawOnly)
    {
        image = maxImg;
    }

   // maxImg.saveAsJPEG("C:/awing/hough/kozeli/" + id  +"_max.jpg");
    //maxImg.writetofloatfile("C:/awing/hough/kozeli/" + id  +"_max.binf");


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



    kernel_addCoordinates<<<1,n>>>( d_coordinates, d_coordinatesFromThatImage, NULL, n, u, size, d_addedCoordinates, Image_cuda_compatible::width, Image_cuda_compatible::height);

}

void Geomcorr::addCoordinates(Image_cuda_compatible& image)
{

    if(d_coordinates == NULL || d_coordinatesFromThatImage ==NULL)
    {
        std::cout <<"ERROR: Got a null pointer at geomcorr::addCoordinates(). Please initialize first! " <<std::endl;
        return;
    }

    kernel_addCoordinates<<<1,n>>>( d_coordinates, d_coordinatesFromThatImage, image.reserve_on_GPU(), n, u, size, d_addedCoordinates, Image_cuda_compatible::width, Image_cuda_compatible::height);

}


bool Geomcorr::exportText(std::string filename)
{
    int *coordinates = new int[2*n*size];
    int* addedCoordinates = new int[n];
    HANDLE_ERROR(cudaMemcpy(coordinates,d_coordinates,n*2*size*sizeof(int),cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(addedCoordinates,d_addedCoordinates,n*sizeof(int),cudaMemcpyDeviceToHost));

    std::ofstream file;
    file.open(filename.c_str());
    if ( !file.is_open())
        return false;
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
    return true;

}


//! Calculates D and V0 with Wu method.

void Geomcorr::dAndVWithWu(float* a, float* b, float* v, float* D, float* v0 )
{

    int rows = (n * (n-1) ) * 0.5;
    std::cout << " welcome todAndVWithWu. n = " << n << " and rows = " << rows <<std::endl;


    *D= 0.0f;
    *v0 = 0.0f;

    std::cout << " initialvalue of D and v0 set. " << std::endl;



    //See Cuda documentation [Cusolver Library]
    cusolverSpHandle_t handle;
    cusolverStatus_t  status;
    status = cusolverSpCreate(&handle);
    if(status != CUSOLVER_STATUS_SUCCESS)
    {
        std::cout << "ERROR: Cusolver cannot be initialized." << std::endl;
        return;
    }
    std::cout << " handle created" << std::endl;
    cusparseMatDescr_t descr = NULL;
    cusparseStatus_t csp;
    csp = cusparseCreateMatDescr(&descr);
    if(csp != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "ERROR: Cusolver cannot be initialized." << std::endl;
        return;
    }
    std::cout << " descr created" << std::endl;

    csp = cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    if(csp != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "ERROR: Cusolver cannot be initialized." << std::endl;
        return;
    }
    std::cout << " csp created" << std::endl;

    csp = cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
    if(csp != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "ERROR: Cusolver cannot be initialized." << std::endl;
        return;
    }
    std::cout << " cps modified." << std::endl;

    if(n==0)
    {
        return;
    }

    int rankA =0;
    float min_norm =0;
    float tol = 0;    // Does not have a clue what this value should be...
    float* x = (float*)malloc(sizeof(float)* 2);
    int* p =(int*) malloc(sizeof(int) * 2);

    float* csrValA = (float*) malloc(sizeof(float) * 2 * rows);

    int* csrRowPtrA = (int*) malloc ( sizeof(int) * (rows + 1));


    int* csrColIndA = (int*) malloc( sizeof(int) * 2 * rows);


    float* vector = (float*) malloc ( sizeof(float) * rows);


    //Uploading matrixes to linear regression:

    int index = 0;
    std::cout << "bump " << std::endl;
    for(int i=0; i< n; i++)
    {
        std::cout << " i =" << i << std::endl;
        for(int j=i+1; j<n; j++)
        {

            std::cout << "Uploading matrix. i=" << i << ", j=" << j<<" and index =  " << index << std::endl;

            csrValA[2*index] = 1.0f;
            csrValA[2*index+1] = (float) ( 0.5 / (long double) (v[i]-v[j]) * (long double) (a[i] / b[i] - a[j]/ b[j])) ;


            csrRowPtrA[index] = 2*index;

            csrColIndA[2*index] =0;
            csrColIndA[2*index +1] =1;

            vector[index] = 0.5f * (v[i] + v[j]) - 0.5f / (v[i] - v[j]) * (1.0f / b[i] - 1.0f / b[j]);
            std::cout << "Matrix row " << index << " : " << 1.0f <<  csrValA[2*index+1] <<std::endl;
            std::cout << "vector [" << index << "]: " <<  vector[index] << std::endl;

            index++;
        }
    }
    csrRowPtrA[rows] = (rows) * 2;


std::cout << " cusolverSpScsrlsqvqrHost" << std::endl;
    status  = cusolverSpScsrlsqvqrHost(handle,rows,2,rows*2,descr,csrValA, csrRowPtrA,csrColIndA,vector,tol, &rankA, x, p, &min_norm );


    if(status!=CUSOLVER_STATUS_SUCCESS)
    {
        std::cout << "ERROR! linear regression failed." << std::endl;

    }
    else
    {
        std::cout << "linear regression with wu method to D and v0. Resuls are:"
                  << std::endl << " v0: " << x[0] << std::endl
                  << "D: " << sqrt(x[1]) << std::endl;
        *D = sqrt(x[1]);
        *v0 = x[0];
    }



    cudaDeviceSynchronize();

    cusolverSpDestroy(handle);


    free( x);
    free(p);
    free(csrValA);
    free(csrRowPtrA);
    free(csrColIndA);
    free(vector);



}

//! Calculates eta and store's it in the Geomcorr class' eta variable.

void Geomcorr::calculateEta()
{



    //See Cuda documentation [Cusolver Library]
    cusolverSpHandle_t handle;
    cusolverStatus_t  status;
    status = cusolverSpCreate(&handle);
    if(status != CUSOLVER_STATUS_SUCCESS)
    {
        std::cout << "ERROR: Cusolver cannot be initialized while calculating eta." << std::endl;
        return;
    }
    cusparseMatDescr_t descr = NULL;
    cusparseStatus_t csp;
    csp = cusparseCreateMatDescr(&descr);
    if(csp != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "ERROR: Cusolver cannot be initialized while calculating eta." << std::endl;
        return;
    }
    csp = cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    if(csp != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "ERROR: Cusolver cannot be initialized while calculating eta." << std::endl;
        return;
    }
    csp = cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
    if(csp != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "ERROR: Cusolver cannot be initialized while calculating eta." << std::endl;
        return;
    }

    float* d_csrValA; // [uj-ui; vj-vi]
    float* d_vector; // [vi*uj - vj*ui]
    int* d_csrRowPtrA;
    int* d_csrColIndA;

    int N = 0;
    HANDLE_ERROR(cudaMemcpy(&N,d_addedCoordinates,sizeof(int),cudaMemcpyDeviceToHost));
    int Nover2 = (N%2 ==0)? N/2-1 : (N-1)/2;

    HANDLE_ERROR(cudaMalloc((void**)&d_csrValA,sizeof(float) * 2 *Nover2) );
    HANDLE_ERROR(cudaMalloc((void**)&d_vector,sizeof(float)  * Nover2 ));
    HANDLE_ERROR(cudaMalloc((void**)&d_csrColIndA,sizeof(int) *2 *Nover2) );
    HANDLE_ERROR(cudaMalloc((void**)&d_csrRowPtrA,sizeof(int) * (Nover2 +1) ));

    float* uhat = new float [n]();
    float* vhat = new float [n]();
    float* norm = new float[n]();
    for(int i=0; i<n; i++)
    {
        norm[i] = 100000000000;


     kernel_fill_eta_matrix<<<((Nover2 + 1023) / 1024) ,1024 >>>(d_csrValA,d_csrRowPtrA,d_csrColIndA, d_vector, d_coordinates, N, i, size);


        int rankA =0;
        float min_norm =0;
        float tol = 0;    // Does not have a clue what this value should be...
        float* x = (float*)malloc(sizeof(float)* 5);
        int* p =(int*) malloc(sizeof(int) * 5);

        float* csrValA = (float*) malloc(sizeof(float) * 2 * Nover2);
        HANDLE_ERROR(cudaMemcpy(csrValA,d_csrValA,sizeof(float) * 2 * Nover2,cudaMemcpyDeviceToHost));

        int* csrRowPtrA = (int*) malloc ( sizeof(int) * (Nover2 + 1));
        HANDLE_ERROR(cudaMemcpy(csrRowPtrA,d_csrRowPtrA,sizeof(int) *(Nover2+1),cudaMemcpyDeviceToHost));

        int* csrColIndA = (int*) malloc( sizeof(int) * 2 * Nover2);
        HANDLE_ERROR(cudaMemcpy(csrColIndA,d_csrColIndA,sizeof(int) * Nover2 * 2,cudaMemcpyDeviceToHost));


        float* vector = (float*) malloc ( sizeof(float) * Nover2);
        HANDLE_ERROR(cudaMemcpy(vector,d_vector,sizeof(float)  * Nover2,cudaMemcpyDeviceToHost));

     status  = cusolverSpScsrlsqvqrHost(handle,Nover2,2,Nover2*2,descr,csrValA, csrRowPtrA,csrColIndA,vector,tol, &rankA, x, p, &min_norm );


    if(status!=CUSOLVER_STATUS_SUCCESS)
    {
        std::cout << "ERROR! uhat, vhat  fit failed for ellipse " << i << std::endl;
    }
    else
    {
        std::cout << "v-hat, u-hat fit for ellipse " << i << std::endl;
        std::cout << " rankA : " << rankA <<std::endl;
        std::cout << "min norm: " << min_norm << std::endl;
        norm[i] = min_norm;

        std::cout << std::endl << std::endl;
        vhat[i] = x[0];
        uhat[i] = x[1];


        std::cout <<" vhat[i] =  " <<x[0]<< std::endl;
        std::cout << " uhat[i] = " << x[1] << std::endl;
        std::cout << std::endl << std::endl ;

    }







    cudaDeviceSynchronize();
    free( x);
    free(p);
    free(csrValA);
    free(csrRowPtrA);
    free(csrColIndA);
    free(vector);
    }

    //fitting eta:
/*
    double xmean =0;
    double ymean = 0;
    double x2mean =0;
    double xymean =0;
    for(int i=0; i<n;i++)
    {
        xmean += vhat[i];
        ymean += uhat[i];
        x2mean += vhat[i] * vhat[i];
        xymean += vhat[i] * uhat[i];
    }
    xmean /= n;
    ymean/=n;
    x2mean /=n;
    xymean/=n;
    */

    long double S = 0.0;
    long double Sx = 0.0;
    long double Sy = 0.0;
    long double Sxx = 0.0;
    long double Sxy = 0.0;
    long double delta = 0.0;


    for(int i=0; i<n; i++)
    {
        long double oos2 = pow(1.0/ norm[i],2); //one over sigma ^2
        S+=  oos2;
        Sx += vhat[i] * oos2;
        Sy += uhat[i] *  oos2;
        Sxx += vhat[i] * vhat[i] *  oos2;
        Sxy += uhat[i] * vhat[i] *  oos2;


    }

    delta = S*Sxx - Sx*Sx;

    long double taneta = (S*Sxy - Sx * Sy) / delta;
    eta = std::atan(taneta);
    long double dtaneta = sqrt(S/delta) ;
    long double deta = 1 / (dtaneta * dtaneta + 1);

    std::cout << "eta is " << eta << " radian that is "
              << eta * 0.5 / 3.1415926535897932384626433 * 360.0 << " degree. " << std::endl;
    std::cout << "error of eta is " << deta<< " radian that is "              << deta *100.0 / eta << " percent " << std::endl << std::endl;







    cusolverSpDestroy(handle);

    HANDLE_ERROR(cudaFree(d_csrValA));
    HANDLE_ERROR(cudaFree(d_vector));
    HANDLE_ERROR(cudaFree(d_csrColIndA));
    HANDLE_ERROR(cudaFree(d_csrRowPtrA));


    delete[] vhat;
    delete[] uhat;
    delete [] norm;

}

void Geomcorr::fitEllipse(int i, float* a, float* b, float* c, float* u, float* v, double* error)
{




    int addedCoordinates = 0;
    HANDLE_ERROR(cudaMemcpy(&addedCoordinates,d_addedCoordinates+i,sizeof(int),cudaMemcpyDeviceToHost));

    if(addedCoordinates < 10)
    {
        std::cout << "ERROR! There are only " << addedCoordinates <<" number of points at ellipse " << i << std::endl;
        *a=*b=*c=*u=*v=0;
        *error = *(error+1) = *(error+2) = *(error+3) = *(error+4) =  50000000000.0f;
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
        *error = *(error+1) = *(error+2) = *(error+3) = *(error+4) =  50000000000.0f;
        return;
    }
    cusparseMatDescr_t descr = NULL;
    cusparseStatus_t csp;
    csp = cusparseCreateMatDescr(&descr);
    if(csp != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "ERROR: Cusolver cannot be initialized." << std::endl;
        *a=*b=*c=*u=*v=0;
        *error = *(error+1) = *(error+2) = *(error+3) = *(error+4) =  50000000000.0f;
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
        *error = *(error+1) = *(error+2) = *(error+3) = *(error+4) =  50000000000.0f;
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



    kernel_fill_matrix<<<((addedCoordinates + 1023) / 1024) ,1024 >>>(eta, d_csrValA,d_csrRowPtrA,d_csrColIndA, d_vector, d_coordinates, addedCoordinates, i, size);


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
        *error = *(error+1) = *(error+2) = *(error+3) = *(error+4) =  50000000000.0f;

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
        std::cout << "a ( axis in pixels) =" << sqrt(1/(*a)) ;
        std::cout << "b (axis in pixels) =" << sqrt(1/(*b)) ;
        std::cout << "c  =" << *c ;

        std::cout << std::endl << std::endl ;

        //calculating error:

        long double stdevNumerator = 0.0;

        for(int j=0; j<addedCoordinates;j++)
        {
            long double numeratorTemp = 0.0;
            for(int i=0;i<5;i++)
            {
                numeratorTemp += x[i]* csrValA[5*j+i];
            }
            numeratorTemp-=vector[j];
            stdevNumerator+= (numeratorTemp*numeratorTemp);
            //std::cout << "stdevNumerator at j = "<< j << " is " << stdevNumerator << std::endl;
        }


        double *xError = new double[5]();

        for(int i=0;i<5;i++)
        {
            long double stdevDenominator = 0.0;

            for(int j=0;j<addedCoordinates;j++)
            {
                stdevDenominator+=csrValA[5*j+i] * csrValA[5*j+i];
                //std::cout << "stdevDenominator at i = "<< i << " and j = " <<j << " is " << stdevDenominator << std::endl;
            }
            stdevDenominator *= (addedCoordinates-5);
            std::cout << "parameter " << i << " : " << x[i];
            xError[i] = sqrt(stdevNumerator / stdevDenominator);
            std::cout << "Error of parameter " << i << ": " << sqrt(stdevNumerator / stdevDenominator) << std::endl;
            std::cout << "Relative error: " << sqrt(stdevNumerator / stdevDenominator) / x[i] * 100.0 << " per cent." << std::endl;
        }
        std::cout << std::endl;

        //du =
        error[0] = sqrt( pow(xError[1] / ( x[0] - x[3] * x[3]),2)
                + pow((x[3] / (x[0]  -  x[3] * x[3])) * xError[2],2)
                + pow(( -x[2] * (x[0] - x[3] * x[3]) + (x[1] - x[2] * x[3] )  * 2 * x[3]) / pow((x[0] - x[3] * x[3]),2) * xError[3],2)
                + pow((((*u)) / (x[0] - x[3] - x[3]) ) * xError[0],2) );
        // dv =
        error[1] =sqrt( pow(xError[0] * (x[2]*(x[0] - x[3] * x[3]) - ( x[0] * x[2] - x[1] * x[3])) / pow((x[0] - x[3] * x[3]),2),2)
                + pow(xError[1] * (x[3] / (x[0] - x[3] * x[3])),2)
                + pow(xError[2] * (x[0] / (x[0] - x[3] * x[3])),2)
                + pow(xError[3] * (-1.0 * x[1] * (x[0] - x[3]* x[3]) + 2 * x[3] * (x[0] * x[2] - x[1] * x[3])) / pow(x[0] - x[3] * x[3],2) ,2) );
        //da:
        error[2] = sqrt( pow(xError[0] * ( (1.0 / (*b) - x[0] * ((*u)) * ((*u))) * (*b) * (*b)),2)
                + pow(error[0] * (((*a)) * (*b) * (x[0] * 2.0 * ((*u)) + 2.0 * x[3] * ((*v)))),2)
                + pow(error[1] * ( 2.0 * ((*a)) * (*b) * (((*v)) + x[3] * ((*u)))) ,2)
                + pow( xError[3] * ((*a)) * (*b) * (2 * ((*u)) * ((*v))) ,2)
                +  pow( xError[4] * ((*a)) * (*b),2));

        // db
        error[3] = sqrt( pow( error[2] / x[0] ,2)
                + pow( xError[0] * ((*a)) / x[0] / x[0] ,2) );
        //dc
        error[4] = sqrt( pow  (xError[3] * ((*b)),2) + pow (error[3] * x[3] ,2));

        printf("\n\nPrinting fitting data with error: \n\n");

        printf(" u = %lf and real. error is %lf percent\n " , (*u) , error[0] / (*u) * 100.0  );
        printf(" v = %lf and real. error is %lf percent \n" , (*v) , error[1] / (*v) * 100.0  );
        printf(" a = %lf and real. error is %lf percent \n" , (*a) , error[2] / (*a) * 100.0  );
        printf(" b = %lf and real. error is %lf percent \n" , (*b) , error[3] / (*b) * 100.0  );
        printf(" c = %lf and real. error is %lf percent \n" , (*c) , error[4] / (*c) * 100.0  );

        free(xError);
    }


    cudaDeviceSynchronize();

    cusolverSpDestroy(handle);

   HANDLE_ERROR( cudaFree(d_csrValA));
    HANDLE_ERROR(cudaFree(d_vector));
    HANDLE_ERROR(cudaFree(d_csrColIndA));
    HANDLE_ERROR(cudaFree(d_csrRowPtrA));
    free( x);
    free(p);
    free(csrValA);
    free(csrRowPtrA);
    free(csrColIndA);
    free(vector);
}


double Geomcorr::calculatePhase(int i, float u)
{
    if( i>=n)
    {
        std::cout << "ERROR: asking  for phase for ellipse " << i << " of " << n << " ( out of boundary)" <<  std::endl;
        return 0;
    }
    if( u < 0 || u > Image_cuda_compatible::width)
    {
        std::cout << "ERROR: invalid u at calculatePhase: u = " << u << std::endl;
        return 0;
    }

    int addedCoordinates = 0;
    HANDLE_ERROR(cudaMemcpy(&addedCoordinates,d_addedCoordinates+i,sizeof(int),cudaMemcpyDeviceToHost));

    float* coordinates = new float[2*addedCoordinates]();
    HANDLE_ERROR(cudaMemcpy(coordinates, d_coordinates + i * 2 *  size  , 2* sizeof(float) * addedCoordinates, cudaMemcpyDeviceToHost ));

    float minu = 0.0f;

    for( int j = 1; j <addedCoordinates-1; j++ )
    {
        float x  =coordinates[2*j];
        float xNext = coordinates[2*j+1];
        if( x < u && xNext > u )
        {
            delete[] coordinates;
            return (double) j / addedCoordinates * 2.0 *CUDART_PI ;
        }
        if(x > u && xNext < u)
        {
            delete[] coordinates;
            return (double) j / addedCoordinates * 2.0 *CUDART_PI  + CUDART_PI;
        }
    }
    delete[] coordinates;
    std::cout << "ERROR: could not determine phase with i = " << i << " and u = " << u << std::endl;
    return 0;


}


//! Copies the coordinates from the n-th ellipse to the given arrays on the CPU.
bool Geomcorr::coordinatesToCPU(int* h_x, int *h_y, int n)
{
    if(n >this->n)
    {
        printf("Could not copy coordinates, because your n  ( %d) is bigger then the number of ellipses. (%d)" , n , this->n);
                return false;
    }

    for(int i =0; i< size; i++)
    {
        HANDLE_ERROR(cudaMemcpy(h_x+ i,d_coordinates + size * n + i * 2,sizeof(int) ,cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(h_y+ i,d_coordinates + size * n + i * 2 + 1,sizeof(int) ,cudaMemcpyDeviceToHost));
    }

    return true;



}


bool Geomcorr::isBallOnLeftSide(int i, float u0)
{
    int u1;
    HANDLE_ERROR(cudaMemcpy(&u1,d_coordinates + size * i ,sizeof(int) ,cudaMemcpyDeviceToHost));
    if( u1 < u0)
        return true;
    else
        return false;

}



