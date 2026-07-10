#include <stdio.h>
#include <cuda_runtime.h>


__global__ void helloFromGPU()
{
    printf("Hello World from GPU thread %d!\n", threadIdx.x);
}


int main()
{
    int deviceCount = 0;

    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if(err != cudaSuccess)
    {
        printf("CUDA initialization failed: %s\n",
               cudaGetErrorString(err));
        return -1;
    }


    printf("CUDA devices found: %d\n", deviceCount);


    if(deviceCount == 0)
    {
        printf("No CUDA GPU detected\n");
        return -1;
    }


    helloFromGPU<<<1,10>>>();


    err = cudaGetLastError();

    if(err != cudaSuccess)
    {
        printf("Kernel launch error: %s\n",
               cudaGetErrorString(err));
        return -1;
    }


    err = cudaDeviceSynchronize();

    if(err != cudaSuccess)
    {
        printf("Kernel execution error: %s\n",
               cudaGetErrorString(err));
        return -1;
    }


    cudaDeviceReset();

    return 0;
}