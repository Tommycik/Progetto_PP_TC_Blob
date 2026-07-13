#ifndef CPU_PIPELINES_H
#define CPU_PIPELINES_H

#include "Common.hpp"
#include <cmath>
#include <omp.h>

//sfocatura gausiana
void cpuGaussianBlur(const std::vector<float>& sourceImage, std::vector<float>& destinationImage, int imageWidth, int imageHeight, float sigma) {
    //calcola raggio e alloca memoria per immagine temporanea
    int kernelRadius = ceilf(3.0f * sigma);
    std::vector<float> temporaryBuffer((size_t)imageWidth * imageHeight);
    //sfocatura orizzontale
    for (int positionY = 0; positionY < imageHeight; ++positionY) {
        for (int positionX = 0; positionX < imageWidth; ++positionX) {
            float pixelSum = 0.0f, totalWeight = 0.0f;
            //ciclo per considerare i pixel adiacenti in orizzontale
            for (int deltaX = -kernelRadius; deltaX <= kernelRadius; ++deltaX) {
                //calcola posizione del vicino
                int neighborX = std::min(std::max(positionX + deltaX, 0), imageWidth - 1);
                //calcola peso in base alla distanza dal pixel centrale
                float weight = std::exp(-(deltaX * deltaX) / (2.0f * sigma * sigma));
                //calcola la somma pesata dei pixel adiacenti e la somma dei pesi
                pixelSum += sourceImage[(size_t)positionY * imageWidth + neighborX] * weight; 
                totalWeight += weight;
            }
            //calcola il valore finale del pixel e lo salva nella destinazione temporanea
            temporaryBuffer[(size_t)positionY * imageWidth + positionX] = pixelSum / totalWeight;
        }
    }
    //sfocatura verticale
    for (int positionY = 0; positionY < imageHeight; ++positionY) {
        for (int positionX = 0; positionX < imageWidth; ++positionX) {
            float pixelSum = 0.0f, totalWeight = 0.0f;
            //ciclo per considerare i pixel adiacenti in verticale
            for (int deltaY = -kernelRadius; deltaY <= kernelRadius; ++deltaY) {
                //calcola posizione del vicino
                int neighborY = std::min(std::max(positionY + deltaY, 0), imageHeight - 1);
                //calcola peso in base alla distanza dal pixel centrale
                float weight = std::exp(-(deltaY * deltaY) / (2.0f * sigma * sigma));
                //calcola la somma pesata dei pixel adiacenti e la somma dei pesi
                pixelSum += temporaryBuffer[(size_t)neighborY * imageWidth + positionX] * weight; 
                totalWeight += weight;
            }
            //calcola il valore finale del pixel e lo salva nella destinazione finale
            destinationImage[(size_t)positionY * imageWidth + positionX] = pixelSum / totalWeight;
        }
    }
}

std::vector<float> runCPUImplementation(const std::vector<float>& hostLuminance, int imageWidth, int imageHeight, float threshold) {
    size_t totalPixels = (size_t)imageWidth * imageHeight;
    std::vector<std::vector<float>> scaleGaussians(NUM_SCALES, std::vector<float>(totalPixels));
    std::vector<std::vector<float>> scaleDogs(NUM_DOGS, std::vector<float>(totalPixels));
    std::vector<float> referenceOutput(totalPixels, 0.0f);
    for (int i = 0; i < NUM_SCALES; ++i) {
        float sigma = SIGMA_BASE * std::pow(K_FACTOR, i);
        cpuGaussianBlur(hostLuminance, scaleGaussians[i], imageWidth, imageHeight, sigma);
    }
    for (int i = 0; i < NUM_DOGS; ++i) {
        for (size_t j = 0; j < totalPixels; ++j) scaleDogs[i][j] = scaleGaussians[i + 1][j] - scaleGaussians[i][j];
    }
    for (int positionY = 1; positionY < imageHeight - 1; ++positionY) {
        for (int positionX = 1; positionX < imageWidth - 1; ++positionX) {
            size_t pixelIndex = (size_t)positionY * imageWidth + positionX;
            float centerValue = scaleDogs[1][pixelIndex]; 
            if (std::abs(centerValue) < threshold) continue;
            bool isMaximum = true, isMinimum = true;
            for (int scale = 0; scale < 3; ++scale) {
                for (int deltaY = -1; deltaY <= 1; ++deltaY) {
                    for (int deltaX = -1; deltaX <= 1; ++deltaX) {
                        if (scale == 1 && deltaX == 0 && deltaY == 0) continue;
                        float neighborValue = scaleDogs[scale][(size_t)(positionY + deltaY) * imageWidth + (positionX + deltaX)];
                        if (neighborValue >= centerValue) isMaximum = false; 
                        if (neighborValue <= centerValue) isMinimum = false;
                    }
                }
            }
            if (isMaximum || isMinimum) referenceOutput[pixelIndex] = 1.0f;
        }
    }
    return referenceOutput;
}

void ompGaussianBlur(const std::vector<float>& sourceImage, std::vector<float>& destinationImage, int imageWidth, int imageHeight, float sigma) {
    int kernelRadius = ceilf(3.0f * sigma);
    std::vector<float> temporaryBuffer((size_t)imageWidth * imageHeight);
    
    #pragma omp parallel for simd collapse(2)
    for (int positionY = 0; positionY < imageHeight; ++positionY) {
        for (int positionX = 0; positionX < imageWidth; ++positionX) {
            float pixelSum = 0.0f, totalWeight = 0.0f;
            for (int deltaX = -kernelRadius; deltaX <= kernelRadius; ++deltaX) {
                int neighborX = std::min(std::max(positionX + deltaX, 0), imageWidth - 1);
                float weight = std::exp(-(deltaX * deltaX) / (2.0f * sigma * sigma));
                pixelSum += sourceImage[(size_t)positionY * imageWidth + neighborX] * weight; 
                totalWeight += weight;
            }
            temporaryBuffer[(size_t)positionY * imageWidth + positionX] = pixelSum / totalWeight;
        }
    }
    
    #pragma omp parallel for simd collapse(2)
    for (int positionY = 0; positionY < imageHeight; ++positionY) {
        for (int positionX = 0; positionX < imageWidth; ++positionX) {
            float pixelSum = 0.0f, totalWeight = 0.0f;
            for (int deltaY = -kernelRadius; deltaY <= kernelRadius; ++deltaY) {
                int neighborY = std::min(std::max(positionY + deltaY, 0), imageHeight - 1);
                float weight = std::exp(-(deltaY * deltaY) / (2.0f * sigma * sigma));
                pixelSum += temporaryBuffer[(size_t)neighborY * imageWidth + positionX] * weight; 
                totalWeight += weight;
            }
            destinationImage[(size_t)positionY * imageWidth + positionX] = pixelSum / totalWeight;
        }
    }
}

std::vector<float> runOpenMPImplementation(const std::vector<float>& hostLuminance, int imageWidth, int imageHeight, float threshold, int threads) {
    omp_set_num_threads(threads);
    size_t totalPixels = (size_t)imageWidth * imageHeight;
    std::vector<std::vector<float>> scaleGaussians(NUM_SCALES, std::vector<float>(totalPixels));
    std::vector<std::vector<float>> scaleDogs(NUM_DOGS, std::vector<float>(totalPixels));
    std::vector<float> openmpOutput(totalPixels, 0.0f);
    for (int i = 0; i < NUM_SCALES; ++i) {
        float sigma = SIGMA_BASE * std::pow(K_FACTOR, i);
        ompGaussianBlur(hostLuminance, scaleGaussians[i], imageWidth, imageHeight, sigma);
    }
    
    #pragma omp parallel for
    for (int i = 0; i < NUM_DOGS; ++i) {
        #pragma omp simd
        for (size_t j = 0; j < totalPixels; ++j) scaleDogs[i][j] = scaleGaussians[i + 1][j] - scaleGaussians[i][j];
    }
    
    #pragma omp parallel for simd collapse(2)
    for (int positionY = 1; positionY < imageHeight - 1; ++positionY) {
        for (int positionX = 1; positionX < imageWidth - 1; ++positionX) {
            size_t pixelIndex = (size_t)positionY * imageWidth + positionX;
            float centerValue = scaleDogs[1][pixelIndex]; 
            if (std::abs(centerValue) < threshold) continue;
            bool isMaximum = true, isMinimum = true;
            for (int scale = 0; scale < 3; ++scale) {
                for (int deltaY = -1; deltaY <= 1; ++deltaY) {
                    for (int deltaX = -1; deltaX <= 1; ++deltaX) {
                        if (scale == 1 && deltaX == 0 && deltaY == 0) continue;
                        float neighborValue = scaleDogs[scale][(size_t)(positionY + deltaY) * imageWidth + (positionX + deltaX)];
                        if (neighborValue >= centerValue) isMaximum = false; 
                        if (neighborValue <= centerValue) isMinimum = false;
                    }
                }
            }
            if (isMaximum || isMinimum) openmpOutput[pixelIndex] = 1.0f;
        }
    }
    return openmpOutput;
}

#endif