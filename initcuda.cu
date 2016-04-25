#ifndef _MSC_VER
#include "initcuda.cuh"

#include "SemaphoreSet.h"
#include <stdio.h>
//#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>


void initCuda()
{
    SemaphoreSet* semPtr;
    SemaphoreSet* masterSemPtr;

    // Retrieve CUDA device index from PCI device index
    //HPC_SDK::CUDAComputeEngine::initialize();
    //const int cudaDeviceIndex = HPC_SDK::CUDAComputeEngine::getInstance().getDeviceIndex( _algorithmSettings.getCudaDeviceIndex() );
//	int deviceIndex = _algorithmSettings.getCudaDeviceIndex();
    int deviceIndex = -1;

    // Set CUDA device based on the retrieved CUDA device ID
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout<<"deviceIndex "<<deviceIndex<<std::endl;
    masterSemPtr=new SemaphoreSet("/tmp/iaps-gpu-master-lock", 1);
    semPtr= new SemaphoreSet("/tmp/iaps-gpu-lock", deviceCount);
    masterSemPtr->lock();
    if(deviceIndex<0)
    {
        deviceIndex=semPtr->lock();
        std::cout<<"automatic deviceIndex "<<deviceIndex<<std::endl;
    }
    else
    {
        if ( deviceIndex >= deviceCount )
        {
//		 	std::stringstream sstr;
            std::cout << "No such device index: " << deviceIndex << ". Modify it in cudaparams.cfg." << std::endl;
//		 	sstr << "No such device index: " << deviceIndex << ". Modify it in cudaparams.cfg.";
//			TERATOMO_RT_REPORTS_RAISE_INVALID_VALUE(sstr.str().c_str());
        }
        else
        {
            deviceIndex=semPtr->lockSpecific(deviceIndex);
        }
    }
    masterSemPtr->unlock();
    cudaSetDevice(deviceIndex);

// Print device properties
    cudaDeviceProp  devProperties;
    cudaGetDeviceProperties( &devProperties, deviceIndex );

        std::cout << "using device (Id = " <<  "CUDA id = " << deviceIndex << ") \"" << devProperties.name << "\"" << std::endl;
        std::cout << "kernelExecTimeoutEnabled = " << (devProperties.kernelExecTimeoutEnabled > 0 ? "true" : "false") << std::endl;
        std::cout << "compute capability = " << devProperties.major << "." << devProperties.minor << std::endl;
        std::cout << "number of stream processors = " << devProperties.multiProcessorCount * 8 << std::endl;
        std::cout << "available shared memory per block (KB) = " << (float)devProperties.sharedMemPerBlock / 1024.0f << std::endl;
        std::cout << "available constant memory (KB) = " << (float)devProperties.totalConstMem / 1024.0f  << std::endl;
        std::cout << "available global memory (MB) = " << (float)devProperties.totalGlobalMem / 1024.0f / 1024.0f  << std::endl<< std::endl<< std::endl;

} // initCuda
#endif
