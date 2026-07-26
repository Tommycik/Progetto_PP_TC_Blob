#ifndef GPU_PIPELINES_CUH
#define GPU_PIPELINES_CUH

#include "Common.hpp"
#include "Kernels.cuh"
#include <cmath>
// Usa la gpu per trovare i blob
std::vector<float> runCudaSingleTest(const std::vector<float>& hostLuminance, int imageWidth, int imageHeight, float threshold, int blockSizeX, int blockSizeY) {
    //calcola numero di pixel dell'immagine e il peso in byte
    size_t totalPixels = (size_t)imageWidth * imageHeight;
    size_t layerBytes = totalPixels * sizeof(float);
    //alloca memoria sul device e copia i dati
    float *deviceLuminance, *deviceTemporary, *deviceGaussians, *deviceDogs, *deviceOutput, *deviceExtremaMap;

    cudaMalloc(&deviceLuminance, layerBytes);
    cudaMalloc(&deviceTemporary, layerBytes);
    cudaMalloc(&deviceOutput, layerBytes);
    cudaMalloc(&deviceExtremaMap, layerBytes);
    cudaMemset(deviceOutput, 0, layerBytes);
    cudaMemset(deviceExtremaMap, 0, layerBytes);
    cudaMemcpy(deviceLuminance, hostLuminance.data(), layerBytes, cudaMemcpyHostToDevice);
    cudaMalloc(&deviceGaussians, layerBytes * NUM_SCALES);
    cudaMalloc(&deviceDogs, layerBytes * NUM_DOGS);
    // calcola dimensione blocco e griglia dei thread
    dim3 blockSize(blockSizeX, blockSizeY);
    dim3 gridSize((imageWidth + blockSizeX - 1) / blockSizeX, (imageHeight + blockSizeY - 1) / blockSizeY);
    // calcola i gaussiani
    for (int i = 0; i < NUM_SCALES; ++i) {
        //raggio di sfocatura
        float sigma = SIGMA_BASE * std::pow(K_FACTOR, i);
        //calcola offset in memoria
        float* currentGaussTarget = deviceGaussians + ((size_t)i * totalPixels);
        // calcola i gaussiani in orizzontale e verticale
        gaussianBlurHorizontalKernel<<<gridSize, blockSize>>>(deviceLuminance, deviceTemporary, imageWidth, imageHeight, sigma);
        gaussianBlurVerticalKernel<<<gridSize, blockSize>>>(deviceTemporary, currentGaussTarget, imageWidth, imageHeight, sigma);
    }
    //calcola i Dogs
    for (int i = 0; i < NUM_DOGS; ++i) computeDoGKernel<<<gridSize, blockSize>>>(deviceGaussians, deviceDogs, i, imageWidth, imageHeight);
    //trova i massimi
    findExtremaKernel<<<gridSize, blockSize>>>(deviceDogs, deviceExtremaMap, imageWidth, imageHeight, threshold, NUM_DOGS);
    //esegue il Non-Maximum Suppression finestra 13 X 13
    int nmsRadius = 6;
    nmsKernel<<<gridSize, blockSize>>>(deviceExtremaMap, deviceOutput, imageWidth, imageHeight, nmsRadius);
    // alloca memoria per i dati di output
    std::vector<float> cudaOutput(totalPixels, 0.0f);
    // attende il completamento di tutti i kernel prima di leggere il risultato
    cudaDeviceSynchronize();
    // copia i risultati dal device all'host
    cudaMemcpy(cudaOutput.data(), deviceOutput, layerBytes, cudaMemcpyDeviceToHost);
    // libera memoria sul device
    cudaFree(deviceLuminance);
    cudaFree(deviceTemporary);
    cudaFree(deviceOutput);
    cudaFree(deviceGaussians);
    cudaFree(deviceExtremaMap);
    cudaFree(deviceDogs);
    return cudaOutput;
}

#endif