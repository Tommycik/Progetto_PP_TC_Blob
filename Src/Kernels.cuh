#ifndef KERNELS_CUH
#define KERNELS_CUH

__global__ void gaussianBlurHorizontalKernel(const float* sourceImage, float* destinationImage, int imageWidth, int imageHeight, float sigma);
__global__ void gaussianBlurVerticalKernel(const float* sourceImage, float* destinationImage, int imageWidth, int imageHeight, float sigma);
__global__ void computeDoGKernel(const float* spaceGaussians, float* spaceDogs, int layerIndex, int imageWidth, int imageHeight);
__global__ void findExtremaKernel(const float* spaceDogs, float* outputMap, int imageWidth, int imageHeight, float threshold, int numDogs);
__global__ void nmsKernel(const float* extremaMap, float* finalOutput, int imageWidth, int imageHeight, int radius);

#endif