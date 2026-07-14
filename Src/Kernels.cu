#include "Kernels.cuh"
#include <cmath>

__global__ void gaussianBlurHorizontalKernel(const float* sourceImage, float* destinationImage, int imageWidth, int imageHeight, float sigma) {
    int positionX = blockIdx.x * blockDim.x + threadIdx.x;
    int positionY = blockIdx.y * blockDim.y + threadIdx.y;
    if (positionX >= imageWidth || positionY >= imageHeight) return;
    int kernelRadius = ceilf(3.0f * sigma);
    float pixelSum = 0.0f, totalWeight = 0.0f;
    size_t currentRowOffset = (size_t)positionY * imageWidth;
    for (int deltaX = -kernelRadius; deltaX <= kernelRadius; ++deltaX) {
        int neighborX = positionX + deltaX;
        if (neighborX < 0) neighborX = 0; 
        if (neighborX >= imageWidth) neighborX = imageWidth - 1;
        float weight = expf(-(deltaX * deltaX) / (2.0f * sigma * sigma));
        pixelSum += sourceImage[currentRowOffset + neighborX] * weight;
        totalWeight += weight;
    }
    destinationImage[currentRowOffset + positionX] = pixelSum / totalWeight;
}

__global__ void gaussianBlurVerticalKernel(const float* sourceImage, float* destinationImage, int imageWidth, int imageHeight, float sigma) {
    int positionX = blockIdx.x * blockDim.x + threadIdx.x;
    int positionY = blockIdx.y * blockDim.y + threadIdx.y;
    if (positionX >= imageWidth || positionY >= imageHeight) return;
    int kernelRadius = ceilf(3.0f * sigma);
    float pixelSum = 0.0f, totalWeight = 0.0f;
    for (int deltaY = -kernelRadius; deltaY <= kernelRadius; ++deltaY) {
        int neighborY = positionY + deltaY;
        if (neighborY < 0) neighborY = 0; 
        if (neighborY >= imageHeight) neighborY = imageHeight - 1;
        float weight = expf(-(deltaY * deltaY) / (2.0f * sigma * sigma));
        pixelSum += sourceImage[(size_t)neighborY * imageWidth + positionX] * weight;
        totalWeight += weight;
    }
    destinationImage[(size_t)positionY * imageWidth + positionX] = pixelSum / totalWeight;
}

__global__ void computeDoGKernel(const float* spaceGaussians, float* spaceDogs, int layerIndex, int imageWidth, int imageHeight) {
    int positionX = blockIdx.x * blockDim.x + threadIdx.x;
    int positionY = blockIdx.y * blockDim.y + threadIdx.y;
    if (positionX >= imageWidth || positionY >= imageHeight) return;
    size_t layerStride = (size_t)imageWidth * imageHeight;
    size_t pixelIndex = (size_t)positionY * imageWidth + positionX;
    float globalGaussian1 = spaceGaussians[(size_t)layerIndex * layerStride + pixelIndex];
    float globalGaussian2 = spaceGaussians[(size_t)(layerIndex + 1) * layerStride + pixelIndex];
    spaceDogs[(size_t)layerIndex * layerStride + pixelIndex] = globalGaussian2 - globalGaussian1;
}

__global__ void findExtremaKernel(const float* spaceDogs, float* outputMap, int imageWidth, int imageHeight, float threshold, int numDogs) {
    int positionX = blockIdx.x * blockDim.x + threadIdx.x;
    int positionY = blockIdx.y * blockDim.y + threadIdx.y;
    if (positionX <= 0 || positionX >= imageWidth - 1 || positionY <= 0 || positionY >= imageHeight - 1) return;
    size_t layerStride = (size_t)imageWidth * imageHeight;
    size_t pixelIndex = (size_t)positionY * imageWidth + positionX;

    for (int dogIdx = 1; dogIdx < numDogs - 1; ++dogIdx) {
        float centerValue = spaceDogs[(size_t)dogIdx * layerStride + pixelIndex];
        if (fabsf(centerValue) < threshold) continue;
        bool isMaximum = true, isMinimum = true;
        for (int scale = dogIdx - 1; scale <= dogIdx + 1; ++scale) {
            size_t layerOffset = (size_t)scale * layerStride;
            for (int deltaY = -1; deltaY <= 1; ++deltaY) {
                for (int deltaX = -1; deltaX <= 1; ++deltaX) {
                    if (scale == dogIdx && deltaX == 0 && deltaY == 0) continue;
                    int neighborX = positionX + deltaX;
                    int neighborY = positionY + deltaY;
                    float neighborValue = spaceDogs[layerOffset + (size_t)neighborY * imageWidth + neighborX];
                    if (neighborValue >= centerValue) isMaximum = false;
                    if (neighborValue <= centerValue) isMinimum = false;
                }
            }
        }
        if (isMaximum || isMinimum) {
            outputMap[pixelIndex] = 1.0f;
            break;
        }
    }
}