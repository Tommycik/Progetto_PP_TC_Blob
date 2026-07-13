#ifndef GPU_PIPELINES_CUH
#define GPU_PIPELINES_CUH

#include "Common.h"
#include "Kernels.cuh"

std::vector<float> runCudaSingleTest(const std::vector<float>& hostLuminance, int imageWidth, int imageHeight, float threshold, int blockSizeX, int blockSizeY) {
    size_t totalPixels = (size_t)imageWidth * imageHeight;
    size_t layerBytes = totalPixels * sizeof(float);
    float *deviceLuminance, *deviceTemporary, *deviceGaussians, *deviceDogs, *deviceOutput;
    
    CUDA_CHECK(cudaMalloc(&deviceLuminance, layerBytes)); 
    CUDA_CHECK(cudaMalloc(&deviceTemporary, layerBytes));
    CUDA_CHECK(cudaMalloc(&deviceOutput, layerBytes)); 
    CUDA_CHECK(cudaMemset(deviceOutput, 0, layerBytes));
    CUDA_CHECK(cudaMemcpy(deviceLuminance, hostLuminance.data(), layerBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&deviceGaussians, layerBytes * NUM_SCALES)); 
    CUDA_CHECK(cudaMalloc(&deviceDogs, layerBytes * NUM_DOGS));

    dim3 blockSize(blockSizeX, blockSizeY); 
    dim3 gridSize((imageWidth + blockSizeX - 1) / blockSizeX, (imageHeight + blockSizeY - 1) / blockSizeY);

    for (int i = 0; i < NUM_SCALES; ++i) {
        float sigma = SIGMA_BASE * powf(K_FACTOR, i); 
        float* currentGaussTarget = deviceGaussians + ((size_t)i * totalPixels);
        gaussianBlurHorizontalKernel<<<gridSize, blockSize>>>(i == 0 ? deviceLuminance : (deviceGaussians + ((size_t)(i - 1) * totalPixels)), deviceTemporary, imageWidth, imageHeight, sigma);
        gaussianBlurVerticalKernel<<<gridSize, blockSize>>>(deviceTemporary, currentGaussTarget, imageWidth, imageHeight, sigma);
    }
    for (int i = 0; i < NUM_DOGS; ++i) computeDoGKernel<<<gridSize, blockSize>>>(deviceGaussians, deviceDogs, i, imageWidth, imageHeight);
    findExtremaKernel<<<gridSize, blockSize>>>(deviceDogs, deviceOutput, imageWidth, imageHeight, threshold);
    
    std::vector<float> cudaOutput(totalPixels, 0.0f);
    CUDA_CHECK(cudaMemcpy(cudaOutput.data(), deviceOutput, layerBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(deviceLuminance); cudaFree(deviceTemporary); cudaFree(deviceOutput); 
    cudaFree(deviceGaussians); cudaFree(deviceDogs);
    return cudaOutput;
}

#endif