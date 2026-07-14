#include "Kernels.cuh"
#include <cmath>
// sfocatura orizzontale sul device
__global__ void gaussianBlurHorizontalKernel(const float* sourceImage, float* destinationImage, int imageWidth, int imageHeight, float sigma) {
    // calcola posizione del pixel e evita accessi illegali alla memoria
    int positionX = blockIdx.x * blockDim.x + threadIdx.x;
    int positionY = blockIdx.y * blockDim.y + threadIdx.y;
    if (positionX >= imageWidth || positionY >= imageHeight) return;
    // calcola raggio e inizializza variabili
    int kernelRadius = ceilf(3.0f * sigma);
    float pixelSum = 0.0f, totalWeight = 0.0f;
    size_t currentRowOffset = (size_t)positionY * imageWidth;
    // calcola la somma pesata dei pixel adiacenti e la somma dei pesi
    for (int deltaX = -kernelRadius; deltaX <= kernelRadius; ++deltaX) {
        int neighborX = positionX + deltaX;
        if (neighborX < 0) neighborX = 0; 
        if (neighborX >= imageWidth) neighborX = imageWidth - 1;
        float weight = expf(-(deltaX * deltaX) / (2.0f * sigma * sigma));
        pixelSum += sourceImage[currentRowOffset + neighborX] * weight;
        totalWeight += weight;
    }
    // salva il risultato nella destinazione
    destinationImage[currentRowOffset + positionX] = pixelSum / totalWeight;
}
// sfocatura verticale sul device
__global__ void gaussianBlurVerticalKernel(const float* sourceImage, float* destinationImage, int imageWidth, int imageHeight, float sigma) {
    // calcola posizione del pixel e evita accessi illegali alla memoria
    int positionX = blockIdx.x * blockDim.x + threadIdx.x;
    int positionY = blockIdx.y * blockDim.y + threadIdx.y;
    if (positionX >= imageWidth || positionY >= imageHeight) return;
    // calcola raggio e inizializza variabili
    int kernelRadius = ceilf(3.0f * sigma);
    float pixelSum = 0.0f, totalWeight = 0.0f;
    // calcola la somma pesata dei pixel adiacenti e la somma dei pesi
    for (int deltaY = -kernelRadius; deltaY <= kernelRadius; ++deltaY) {
        int neighborY = positionY + deltaY;
        if (neighborY < 0) neighborY = 0; 
        if (neighborY >= imageHeight) neighborY = imageHeight - 1;
        float weight = expf(-(deltaY * deltaY) / (2.0f * sigma * sigma));
        pixelSum += sourceImage[(size_t)neighborY * imageWidth + positionX] * weight;
        totalWeight += weight;
    }
    // salva il risultato nella destinazione
    destinationImage[(size_t)positionY * imageWidth + positionX] = pixelSum / totalWeight;
}
// calcola i DoGs
__global__ void computeDoGKernel(const float* spaceGaussians, float* spaceDogs, int layerIndex, int imageWidth, int imageHeight) {
    // calcola posizione del pixel e evita accessi illegali alla memoria
    int positionX = blockIdx.x * blockDim.x + threadIdx.x;
    int positionY = blockIdx.y * blockDim.y + threadIdx.y;
    if (positionX >= imageWidth || positionY >= imageHeight) return;
    // calcola l'indice del pixel e la dimensione in pixel dell'immagine
    size_t layerStride = (size_t)imageWidth * imageHeight;
    size_t pixelIndex = (size_t)positionY * imageWidth + positionX;
    // calcola il valore della differenza tra i gaussiani adiacenti
    float globalGaussian1 = spaceGaussians[(size_t)layerIndex * layerStride + pixelIndex];
    float globalGaussian2 = spaceGaussians[(size_t)(layerIndex + 1) * layerStride + pixelIndex];
    spaceDogs[(size_t)layerIndex * layerStride + pixelIndex] = globalGaussian2 - globalGaussian1;
}
// trova i massimi e i minimi
__global__ void findExtremaKernel(const float* spaceDogs, float* outputMap, int imageWidth, int imageHeight, float threshold, int numDogs) {
    // calcola posizione del pixel e evita accessi illegali alla memoria
    int positionX = blockIdx.x * blockDim.x + threadIdx.x;
    int positionY = blockIdx.y * blockDim.y + threadIdx.y;
    if (positionX <= 0 || positionX >= imageWidth - 1 || positionY <= 0 || positionY >= imageHeight - 1) return;
    // calcola l'indice del pixel e la dimensione in pixel dell'immagine
    size_t layerStride = (size_t)imageWidth * imageHeight;
    size_t pixelIndex = (size_t)positionY * imageWidth + positionX;
    // cicla sui livelli validi
    for (int dogIdx = 1; dogIdx < numDogs - 1; ++dogIdx) {
        // ottiene il valore del pixel
        float centerValue = spaceDogs[(size_t)dogIdx * layerStride + pixelIndex];
        // se il valore assoluto è inferiore al threshold non considera il pixel
        if (fabsf(centerValue) < threshold) continue;
        bool isMaximum = true, isMinimum = true;
        // controlla i 26 pixel vicini
        for (int scale = dogIdx - 1; scale <= dogIdx + 1; ++scale) {
            size_t layerOffset = (size_t)scale * layerStride;
            for (int deltaY = -1; deltaY <= 1; ++deltaY) {
                for (int deltaX = -1; deltaX <= 1; ++deltaX) {
                    // salta il pixel centrale
                    if (scale == dogIdx && deltaX == 0 && deltaY == 0) continue;
                    int neighborX = positionX + deltaX;
                    int neighborY = positionY + deltaY;
                    float neighborValue = spaceDogs[layerOffset + (size_t)neighborY * imageWidth + neighborX];
                    if (neighborValue > centerValue) isMaximum = false;
                    if (neighborValue < centerValue) isMinimum = false;
                }
            }
        }
        // se è massimo o minimo, lo salva nella destinazione
        if (isMaximum || isMinimum) {
            outputMap[pixelIndex] = 1.0f;
            break;
        }
    }
}

// Non-Maximum Suppression Spaziale
__global__ void nmsKernel(const float* extremaMap, float* finalOutput, int imageWidth, int imageHeight, int radius) {
    // calcola posizione del pixel e evita accessi illegali alla memoria
    int positionX = blockIdx.x * blockDim.x + threadIdx.x;
    int positionY = blockIdx.y * blockDim.y + threadIdx.y;
    if (positionX >= imageWidth || positionY >= imageHeight) return;

    size_t pixelIndex = (size_t)positionY * imageWidth + positionX;
    float currentStrength = extremaMap[pixelIndex];

    // Se non era un candidato blob lascia 0 ed esce
    if (currentStrength == 0.0f) {
        finalOutput[pixelIndex] = 0.0f;
        return;
    }

    bool isLocalMax = true;

    // controlla la finestra spaziale attorno al pixel
    for (int deltaY = -radius; deltaY <= radius; ++deltaY) {
        int neighborY = positionY + deltaY;
        if (neighborY < 0 || neighborY >= imageHeight) continue;

        for (int deltaX = -radius; deltaX <= radius; ++deltaX) {
            int neighborX = positionX + deltaX;
            if (neighborX < 0 || neighborX >= imageWidth) continue;
            if (deltaX == 0 && deltaY == 0) continue;

            float neighborStrength = extremaMap[(size_t)neighborY * imageWidth + neighborX];

            // se un vicino ha una risposta DoG più forte sopprime questo blob
            if (neighborStrength > currentStrength) {
                isLocalMax = false;
                break;
            }
        }
        if (!isLocalMax) break;
    }

    // Se è il massimo locale della finestra diventa un blob definitivo (1.0f)
    finalOutput[pixelIndex] = isLocalMax ? 1.0f : 0.0f;
}