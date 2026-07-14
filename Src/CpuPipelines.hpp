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
// Trova i blob usando la cpu in modo sequenziale
std::vector<float> runCPUImplementation(const std::vector<float>& hostLuminance, int imageWidth, int imageHeight, float threshold) {
    //calcola numero di pixel dell'immagine
    size_t totalPixels = (size_t)imageWidth * imageHeight;
    //alloca memoria per i gaussiani e i dogs
    std::vector<std::vector<float>> scaleGaussians(NUM_SCALES, std::vector<float>(totalPixels));
    std::vector<std::vector<float>> scaleDogs(NUM_DOGS, std::vector<float>(totalPixels));
    //alloca memoria per i risultati
    std::vector<float> referenceOutput(totalPixels, 0.0f);
    // genera le immagini sfocate
    for (int i = 0; i < NUM_SCALES; ++i) {
        //raggio di sfocatura
        float sigma = SIGMA_BASE * std::pow(K_FACTOR, i);
        cpuGaussianBlur(hostLuminance, scaleGaussians[i], imageWidth, imageHeight, sigma);
    }
    // genera i DoGs
    for (int i = 0; i < NUM_DOGS; ++i) {
        for (size_t j = 0; j < totalPixels; ++j) scaleDogs[i][j] = scaleGaussians[i + 1][j] - scaleGaussians[i][j];
    }
    // trova i blob cercando gli estremi
    for (int positionY = 1; positionY < imageHeight - 1; ++positionY) {
        for (int positionX = 1; positionX < imageWidth - 1; ++positionX) {
            //calcola l'indice del pixel
            size_t pixelIndex = (size_t)positionY * imageWidth + positionX;

            for (int dogIdx = 1; dogIdx < NUM_DOGS - 1; ++dogIdx) {
                //ottiene il valore del pixel
                float centerValue = scaleDogs[dogIdx][pixelIndex];
                //se il valore assoluto è inferiore al threshold, non considera il pixel
                if (std::abs(centerValue) < threshold) continue;
                bool isMaximum = true, isMinimum = true;
                //ciclo per trovare i massimi e i minimi nei 26 pixel vicini
                for (int scale = dogIdx - 1; scale <= dogIdx + 1; ++scale) {
                    for (int deltaY = -1; deltaY <= 1; ++deltaY) {
                        for (int deltaX = -1; deltaX <= 1; ++deltaX) {
                            //ignora il pixel centrale
                            if (scale == dogIdx && deltaX == 0 && deltaY == 0) continue;
                            //ottiene il valore del vicino
                            float neighborValue = scaleDogs[scale][(size_t)(positionY + deltaY) * imageWidth + (positionX + deltaX)];
                            //se il vicino è maggiore o minore del pixel centrale, non è un massimo o minimo
                            if (neighborValue >= centerValue) isMaximum = false;
                            if (neighborValue <= centerValue) isMinimum = false;
                        }
                    }
                }
                //se è un massimo o minimo, lo salva in output
                if (isMaximum || isMinimum) {
                    referenceOutput[pixelIndex] = 1.0f;
                    break;
                }
            }
        }
    }
    return referenceOutput;
}
// sfocatura gaussiana usando OpenMP
void ompGaussianBlur(const std::vector<float>& sourceImage, std::vector<float>& destinationImage, int imageWidth, int imageHeight, float sigma) {
    // calcola il raggio del kernel
    int kernelRadius = ceilf(3.0f * sigma);
    // alloca memoria per il buffer temporaneo
    std::vector<float> temporaryBuffer((size_t)imageWidth * imageHeight);
    //regione parallela
    #pragma omp parallel
    // parallelizza due for annidati e usa simd
        #pragma omp for simd collapse(2)
        // sfocatura orizzontale
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
        // parallelizza due for annidati e usa simd
        #pragma omp for simd collapse(2)
        // sfocatura verticale
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
// trova i blob in modo parallelo
std::vector<float> runOpenMPImplementation(const std::vector<float>& hostLuminance, int imageWidth, int imageHeight, float threshold, int threads) {
    // setta il numero di thread
    omp_set_num_threads(threads);
    // calcola il numero di pixel e alloca memoria per i gaussiani,i Dogs e l'output
    size_t totalPixels = (size_t)imageWidth * imageHeight;
    std::vector<std::vector<float>> scaleGaussians(NUM_SCALES, std::vector<float>(totalPixels));
    std::vector<std::vector<float>> scaleDogs(NUM_DOGS, std::vector<float>(totalPixels));
    std::vector<float> openmpOutput(totalPixels, 0.0f);
    // calcola i gaussiani
    for (int i = 0; i < NUM_SCALES; ++i) {
        // raggio di sfocatura
        float sigma = SIGMA_BASE * std::pow(K_FACTOR, i);
        ompGaussianBlur(hostLuminance, scaleGaussians[i], imageWidth, imageHeight, sigma);
    }
    // calcola i Dogs in modo parallelo
    #pragma omp parallel for simd collapse(2)
    for (int i = 0; i < NUM_DOGS; ++i) {
        for (size_t j = 0; j < totalPixels; ++j) {
            scaleDogs[i][j] = scaleGaussians[i + 1][j] - scaleGaussians[i][j];
        }
    }
    // trova i blob in modo parallelo
    #pragma omp parallel for collapse(2)
    for (int positionY = 1; positionY < imageHeight - 1; ++positionY) {
        for (int positionX = 1; positionX < imageWidth - 1; ++positionX) {
            // calcola l'indice del pixel
            size_t pixelIndex = (size_t)positionY * imageWidth + positionX;

            for (int dogIdx = 1; dogIdx < NUM_DOGS - 1; ++dogIdx) {
                // ottiene il valore del pixel
                float centerValue = scaleDogs[dogIdx][pixelIndex];
                // se il valore del pixel è inferiore al threshold, salta il pixel
                if (std::abs(centerValue) < threshold) continue;
                bool isMaximum = true, isMinimum = true;
                // trova i massimi e i minimi nei 26 vicini
                for (int scale = dogIdx - 1; scale <= dogIdx + 1; ++scale) {
                    for (int deltaY = -1; deltaY <= 1; ++deltaY) {
                        for (int deltaX = -1; deltaX <= 1; ++deltaX) {
                            // salta il pixel centrale
                            if (scale == dogIdx && deltaX == 0 && deltaY == 0) continue;
                            // ottiene il valore del vicino
                            float neighborValue = scaleDogs[scale][(size_t)(positionY + deltaY) * imageWidth + (positionX + deltaX)];
                            if (neighborValue >= centerValue) isMaximum = false;
                            if (neighborValue <= centerValue) isMinimum = false;
                        }
                    }
                }
                // se il pixel è un massimo o un minimo lo aggiunge all'output
                if (isMaximum || isMinimum) {
                    openmpOutput[pixelIndex] = 1.0f;
                    break;
                }
            }
        }
    }
    return openmpOutput;
}

#endif