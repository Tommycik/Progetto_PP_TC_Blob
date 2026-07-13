#include "Common.h"
#include "CpuPipelines.h"
#include "GpuPipelines.cuh"
#include <chrono>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <SFML/Graphics.hpp>

void runBenchmark() {
    sf::Image inputImage; if (!inputImage.loadFromFile("../input.png")) return;
    sf::Vector2u size = inputImage.getSize(); size_t totalPixels = (size_t)size.x * size.y;
    std::vector<float> hostLuminance(totalPixels);
    for (unsigned int y = 0; y < size.y; ++y) {
        for (unsigned int x = 0; x < size.x; ++x) {
            sf::Color color = inputImage.getPixel(x, y);
            hostLuminance[(size_t)y * size.x + x] = (0.299f * color.r + 0.587f * color.g + 0.114f * color.b) / 255.0f;
        }
    }
    float threshold = 0.0005f;
    const int RUNS = 5;

    std::cout << "\nAVVIO PIPELINE DI BENCHMARK" << std::endl;
    std::ofstream csvFile("benchmark_results.csv");
    if (csvFile.is_open()) {
        csvFile << "Backend,Parametro,TempoMedio_ms,TempoMin_ms,TempoMax_ms,Speedup,Verificato\n";
    }

    double sequentialTotalTime = 0.0;
    std::vector<float> referenceOutput;
    for (int r = 0; r < RUNS; ++r) {
        auto startTiming = std::chrono::high_resolution_clock::now();
        referenceOutput = runCPUImplementation(hostLuminance, size.x, size.y, threshold);
        auto endTiming = std::chrono::high_resolution_clock::now();
        sequentialTotalTime += std::chrono::duration<double, std::milli>(endTiming - startTiming).count();
    }
    sequentialTotalTime /= RUNS;

    std::cout << "\nBaseline Sequenziale CPU: " << std::fixed << std::setprecision(2) << sequentialTotalTime << " ms" << std::endl;
    if (csvFile.is_open()) {
        csvFile << "CPU_Sequenziale,1," << sequentialTotalTime << "," << sequentialTotalTime << "," << sequentialTotalTime << ",1.0,SI\n";
    }

    auto checkMatch = [&](const std::vector<float>& targetBuffer) -> std::string {
        for (size_t i = 0; i < totalPixels; ++i) {
            if (std::abs(referenceOutput[i] - targetBuffer[i]) > 0.01f) return "NO";
        }
        return "SI";
    };

    std::vector<int> threadCounts = {12};
    std::cout << "\nPerformance Multi Core OpenMP e SIMD" << std::endl;
    std::cout << std::setw(10) << "Threads" << std::setw(15) << "Medio ms" << std::setw(12) << "Min ms" << std::setw(12) << "Max ms" << std::setw(12) << "Speedup" << std::setw(12) << "Match" << std::endl;
    std::cout << "----------------------------------------------------------------------------" << std::endl;

    for (int threads : threadCounts) {
        double openmpTotalTime = 0.0, openmpMinTime = 999999.0, openmpMaxTime = 0.0;
        std::vector<float> openmpOutput;
        for (int r = 0; r < RUNS; ++r) {
            auto startTiming = std::chrono::high_resolution_clock::now();
            openmpOutput = runOpenMPImplementation(hostLuminance, size.x, size.y, threshold, threads);
            auto endTiming = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double, std::milli>(endTiming - startTiming).count();
            openmpTotalTime += duration;
            openmpMinTime = std::min(openmpMinTime, duration);
            openmpMaxTime = std::max(openmpMaxTime, duration);
        }
        double averageDuration = openmpTotalTime / RUNS;
        std::string identityMatch = checkMatch(openmpOutput);
        std::cout << std::setw(10) << threads << std::setw(15) << averageDuration << std::setw(12) << openmpMinTime << std::setw(12) << openmpMaxTime << std::setw(11) << (sequentialTotalTime / averageDuration) << "x" << std::setw(12) << identityMatch << std::endl;
        if (csvFile.is_open()) {
            csvFile << "CPU_OpenMP," << threads << "," << averageDuration << "," << openmpMinTime << "," << openmpMaxTime << "," << (sequentialTotalTime / averageDuration) << "," << identityMatch << "\n";
        }
    }

    std::vector<std::pair<int, int>> blockSizes = {
        {16, 16}, {32, 32}, {32, 8}, {8, 32}, {64, 4}
    };
    std::cout << "\nPerformance Scheda Video CUDA" << std::endl;
    std::cout << std::setw(10) << "Blocco" << std::setw(15) << "Medio ms" << std::setw(12) << "Min ms" << std::setw(12) << "Max ms" << std::setw(12) << "Speedup" << std::setw(12) << "Match" << std::endl;
    std::cout << "----------------------------------------------------------------------------" << std::endl;

    for (auto executionBlock : blockSizes) {
        double cudaTotalTime = 0.0, cudaMinTime = 999999.0, cudaMaxTime = 0.0;
        std::vector<float> cudaOutput;
        for (int r = 0; r < RUNS; ++r) {
            auto startTiming = std::chrono::high_resolution_clock::now();
            cudaOutput = runCudaSingleTest(hostLuminance, size.x, size.y, threshold, executionBlock.first, executionBlock.second);
            auto endTiming = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double, std::milli>(endTiming - startTiming).count();
            cudaTotalTime += duration;
            cudaMinTime = std::min(cudaMinTime, duration);
            cudaMaxTime = std::max(cudaMaxTime, duration);
        }
        double averageDuration = cudaTotalTime / RUNS;
        std::string identityMatch = checkMatch(cudaOutput);
        std::string blockLabel = std::to_string(executionBlock.first) + "x" + std::to_string(executionBlock.second);
        std::cout << std::setw(10) << blockLabel << std::setw(15) << averageDuration << std::setw(12) << cudaMinTime << std::setw(12) << cudaMaxTime << std::setw(11) << (sequentialTotalTime / averageDuration) << "x" << std::setw(12) << identityMatch << std::endl;
        if (csvFile.is_open()) {
            csvFile << "CUDA_SchedaVideo," << blockLabel << "," << averageDuration << "," << cudaMinTime << "," << cudaMaxTime << "," << (sequentialTotalTime / averageDuration) << "," << identityMatch << "\n";
        }
    }

    if (csvFile.is_open()) {
        csvFile.close();
        std::cout << "\nPipeline terminata. Dati salvati" << std::endl;
    }
}

void runGUI() {
    int blobRadius = 10;

    sf::Image inputImage; if (!inputImage.loadFromFile("../input.png")) return;
    sf::Vector2u size = inputImage.getSize(); size_t totalPixels = (size_t)size.x * size.y;

    std::vector<float> hostLuminance(totalPixels);
    for (unsigned int y = 0; y < size.y; ++y) {
        for (unsigned int x = 0; x < size.x; ++x) {
            sf::Color color = inputImage.getPixel(x, y);
            hostLuminance[(size_t)y * size.x + x] = (0.299f * color.r + 0.587f * color.g + 0.114f * color.b) / 255.0f;
        }
    }

    float threshold = 0.0005f;
    std::vector<float> blobMap = runCudaSingleTest(hostLuminance, size.x, size.y, threshold, 16, 16);

    sf::Image resultImage; resultImage.create(size.x, size.y);
    for (unsigned int y = 0; y < size.y; ++y) {
        for (unsigned int x = 0; x < size.x; ++x) {
            sf::Color originalColor = inputImage.getPixel(x, y);
            resultImage.setPixel(x, y, sf::Color(originalColor.r * 0.6f, originalColor.g * 0.6f, originalColor.b * 0.6f, 255));
        }
    }

    for (unsigned int y = 0; y < size.y; ++y) {
        for (unsigned int x = 0; x < size.x; ++x) {
            size_t pixelIndex = (size_t)y * size.x + x;
            if (blobMap[pixelIndex] > 0.5f) {
                for (int deltaY = -blobRadius; deltaY <= blobRadius; ++deltaY) {
                    for (int deltaX = -blobRadius; deltaX <= blobRadius; ++deltaX) {
                        int neighborX = (int)x + deltaX;
                        int neighborY = (int)y + deltaY;
                        if (neighborX >= 0 && neighborX < (int)size.x && neighborY >= 0 && neighborY < (int)size.y) {
                            float exactDistance = sqrtf(deltaX * deltaX + deltaY * deltaY);
                            if (fabsf(exactDistance - blobRadius) < 0.4f) {
                                resultImage.setPixel(neighborX, neighborY, sf::Color(255, 0, 0, 255));
                            }
                        }
                    }
                }
            }
        }
    }

    resultImage.saveToFile("risultato_blobs.png");
    sf::RenderWindow window(sf::VideoMode(size.x, size.y), "Risultato");
    sf::Texture texture; texture.loadFromImage(resultImage); sf::Sprite sprite(texture);
    while (window.isOpen()) {
        sf::Event event; while (window.pollEvent(event)) { if (event.type == sf::Event::Closed) window.close(); }
        window.clear(); window.draw(sprite); window.display();
    }
}

int main() {
    int choice; std::cout << "1. Pipeline Benchmark\n2. Finestra Grafica\nScelta: "; std::cin >> choice;
    if (choice == 1) runBenchmark(); else runGUI();
    return 0;
}