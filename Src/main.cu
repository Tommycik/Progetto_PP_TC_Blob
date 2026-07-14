#include "Common.hpp"
#include "CpuPipelines.hpp"
#include "GpuPipelines.cuh"
#include <chrono>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <SFML/Graphics.hpp>

// benchmark
void runBenchmark() {
    // carica l'immagine da processare e calcola la luminanza e le dimensioni in pixel
    sf::Image inputImage; if (!inputImage.loadFromFile("../input.png")) return;
    sf::Vector2u size = inputImage.getSize(); size_t totalPixels = (size_t)size.x * size.y;
    std::vector<float> hostLuminance(totalPixels);
    for (unsigned int y = 0; y < size.y; ++y) {
        for (unsigned int x = 0; x < size.x; ++x) {
            sf::Color color = inputImage.getPixel(x, y);
            // calcola la luminanza del pixel estraendo i canali di colore
            hostLuminance[(size_t)y * size.x + x] = (0.299f * color.r + 0.587f * color.g + 0.114f * color.b) / 255.0f;
        }
    }
    float threshold = 0.0005f;
    const int RUNS = 5;
    // avvia il benchmark e apre o crea un file CSV
    std::cout << "\nAVVIO PIPELINE DI BENCHMARK" << std::endl;
    std::ofstream csvFile("benchmark_results.csv");
    if (csvFile.is_open()) {
        csvFile << "Backend,Parametro,TempoMedio_ms,TempoMin_ms,TempoMax_ms,Speedup,Verificato\n";
    }
    //esegue pipeline sequenziale 5 volte e fa la media dei tempi
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
    // scrive i risultati della baseline sul file CSV
    if (csvFile.is_open()) {
        csvFile << "CPU_Sequenziale,1," << sequentialTotalTime << "," << sequentialTotalTime << "," << sequentialTotalTime << ",1.0,SI\n";
    }
    // funzione di verifica dell'output con i risultati di riferimento
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
    // esegue pipeline OpenMP e calcola i tempi
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
        // verifica l'output con i risultati di riferimento
        std::string identityMatch = checkMatch(openmpOutput);
        std::cout << std::setw(10) << threads << std::setw(15) << averageDuration << std::setw(12) << openmpMinTime << std::setw(12) << openmpMaxTime << std::setw(11) << (sequentialTotalTime / averageDuration) << "x" << std::setw(12) << identityMatch << std::endl;
        // scrive i risultati su file CSV
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
    // esegue pipeline CUDA con le varie configurazioni e calcola i tempi
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
        // verifica l'output con i risultati di riferimento
        std::string identityMatch = checkMatch(cudaOutput);
        std::string blockLabel = std::to_string(executionBlock.first) + "x" + std::to_string(executionBlock.second);
        std::cout << std::setw(10) << blockLabel << std::setw(15) << averageDuration << std::setw(12) << cudaMinTime << std::setw(12) << cudaMaxTime << std::setw(11) << (sequentialTotalTime / averageDuration) << "x" << std::setw(12) << identityMatch << std::endl;
        // scrive i risultati su file CSV
        if (csvFile.is_open()) {
            csvFile << "CUDA_SchedaVideo," << blockLabel << "," << averageDuration << "," << cudaMinTime << "," << cudaMaxTime << "," << (sequentialTotalTime / averageDuration) << "," << identityMatch << "\n";
        }
    }
    // chiude il file CSV
    if (csvFile.is_open()) {
        csvFile.close();
        std::cout << "\nPipeline terminata. Dati salvati" << std::endl;
    }
}
// esegue la pipeline GUI
void runGUI() {
    // parametri iniziali
    int blobRadius = 10;
    // carica l'immagine e ne calcola la luminosità e dimensione in pixel
    sf::Image inputImage; if (!inputImage.loadFromFile("../input.png")) return;
    sf::Vector2u size = inputImage.getSize(); size_t totalPixels = (size_t)size.x * size.y;

    std::vector<float> hostLuminance(totalPixels);
    for (unsigned int y = 0; y < size.y; ++y) {
        for (unsigned int x = 0; x < size.x; ++x) {
            sf::Color color = inputImage.getPixel(x, y);
            hostLuminance[(size_t)y * size.x + x] = (0.299f * color.r + 0.587f * color.g + 0.114f * color.b) / 255.0f;
        }
    }

    float threshold = 0.0001f;
    // esegue la pipeline CUDA
    std::vector<float> blobMap = runCudaSingleTest(hostLuminance, size.x, size.y, threshold, 16, 16);
    // crea l'immagine finale da quella inziiale ma riducendo la luminosità per dare contrasto ai blob
    sf::Image resultImage; resultImage.create(size.x, size.y);
    for (unsigned int y = 0; y < size.y; ++y) {
        for (unsigned int x = 0; x < size.x; ++x) {
            sf::Color originalColor = inputImage.getPixel(x, y);
            resultImage.setPixel(x, y, sf::Color(originalColor.r * 0.6f, originalColor.g * 0.6f, originalColor.b * 0.6f, 255));
        }
    }
    // evidenzia i blob con dei cerchi rossi
    for (unsigned int y = 0; y < size.y; ++y) {
        for (unsigned int x = 0; x < size.x; ++x) {
            size_t pixelIndex = (size_t)y * size.x + x;
            if (blobMap[pixelIndex] > 0.5f) {
                for (int deltaY = -blobRadius; deltaY <= blobRadius; ++deltaY) {
                    for (int deltaX = -blobRadius; deltaX <= blobRadius; ++deltaX) {
                        int neighborX = (int)x + deltaX;
                        int neighborY = (int)y + deltaY;
                        if (neighborX >= 0 && neighborX < (int)size.x && neighborY >= 0 && neighborY < (int)size.y) {
                            // crea il cerchio
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
    //salva l'immagine finale
    resultImage.saveToFile("risultato_blobs.png");
    // crea la finestra e la texture
    sf::RenderWindow window(sf::VideoMode(size.x, size.y), "Risultato");
    sf::Texture texture; texture.loadFromImage(resultImage); sf::Sprite sprite(texture);
    while (window.isOpen()) {
        sf::Event event; while (window.pollEvent(event)) { if (event.type == sf::Event::Closed) window.close(); }
        window.clear(); window.draw(sprite); window.display();
    }
}
//main
int main() {
    int choice;
    std::cout << "1. Pipeline Benchmark\n2. Finestra Grafica\nScelta: ";
    std::cin >> choice;
    if (choice == 1) {
        runBenchmark();
    }else if (choice == 2) {
        runGUI();
    }else {
        std::cout << "Scelta non valida!\n";
    }
    return 0;
}