#include "Common.hpp"
#include "CpuPipelines.hpp"
#include "GpuPipelines.cuh"
#include <chrono>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <SFML/Graphics.hpp>
#include <numeric>

// calcola la deviazione standard dei tempi di esecuzione per valutare la stabilità
double calculateStdDev(const std::vector<double>& times, double mean) {
    double sum = 0.0;
    for (double t : times) {
        sum += (t - mean) * (t - mean);
    }
    return std::sqrt(sum / times.size());
}
// controlla se esistono le immagini a risoluzione crescente e, se manca qualcosa, le genera a partire da quella 512x512
void checkAndGenerateImages() {
    sf::Image sourceImage;
    // prova a caricare l'immagine base da 512x512 pixel
    if (!sourceImage.loadFromFile("../images/input_512.png")) {
        std::cout << "Errore: Immagine base '../images/input_512.png' non trovata! Scaricala prima di avviare." << std::endl;
        return;
    }

    std::vector<unsigned int> targetSizes = {1024, 2048, 4096};

    // scorre le dimensioni target per verificare se i rispettivi file esistono già
    for (unsigned int targetSize : targetSizes) {
        std::string targetPath = "../images/input_" + std::to_string(targetSize) + ".png";
        sf::Image checkImg;

        // se l'immagine non esiste, la genera effettuando il ridimensionamento
        if (!checkImg.loadFromFile(targetPath)) {
            std::cout << "Generazione: Creazione immagine in corso: " << targetPath << std::endl;

            // mappa i pixel per scalare l'immagine con una interpolazione bilineare
            sf::Image resizedImage;
            resizedImage.create(targetSize, targetSize);

            for (unsigned int y = 0; y < targetSize; ++y) {
                for (unsigned int x = 0; x < targetSize; ++x) {
                    // calcola le coordinate corrispondenti sull'immagine sorgente originale
                    float sourceX = (static_cast<float>(x) / targetSize) * 512.0f;
                    float sourceY = (static_cast<float>(y) / targetSize) * 512.0f;

                    unsigned int x0 = std::min(static_cast<unsigned int>(std::floor(sourceX)), 511u);
                    unsigned int y0 = std::min(static_cast<unsigned int>(std::floor(sourceY)), 511u);

                    // assegna il pixel campionato all'immagine a risoluzione maggiore
                    resizedImage.setPixel(x, y, sourceImage.getPixel(x0, y0));
                }
            }
            // salva il file generato su disco nella cartella prestabilita
            resizedImage.saveToFile(targetPath);
        }
    }
}
// benchmark
void runBenchmark() {
    const int RUNS = 5;
    // definisce i percorsi delle immagini con risoluzioni crescenti da testare
    std::vector<std::string> imagePaths = {
        "../images/input_512.png",
        "../images/input_1024.png",
        "../images/input_2048.png",
        "../images/input_4096.png"
    };
    // definisce le diverse soglie da testare come parametri addizionali
    std::vector<float> thresholds = { 0.0005f, 0.005f };
    // definisce le configurazioni dei blocchi per la scheda video
    std::vector<std::pair<int, int>> blockSizes = {
        {16, 16}, {32, 32}, {32, 8}, {8, 32}, {64, 4}
    };
    std::vector<int> threadCounts = {12};

    // avvia il benchmark e apre o crea un file CSV
    std::cout << "\nAVVIO PIPELINE DI BENCHMARK AVANZATA" << std::endl;
    std::ofstream csvFile("benchmark_results.csv");
    if (csvFile.is_open()) {
        // scrive l'intestazione del file CSV includendo risoluzione e deviazione standard
        csvFile << "Risoluzione,Backend,Parametro,Soglia,TempoMedio_ms,TempoMin_ms,TempoMax_ms,DevStd_ms,Speedup,Verificato\n";
    }

    // esegue il ciclo principale sulle diverse risoluzioni delle immagini
    for (const auto& imgPath : imagePaths) {
        // carica l'immagine corrente da processare e calcola la luminanza e le dimensioni in pixel
        sf::Image inputImage;
        if (!inputImage.loadFromFile(imgPath)) {
            std::cout << "Immagine non trovata, salto il test per: " << imgPath << std::endl;
            continue;
        }
        sf::Vector2u size = inputImage.getSize();
        size_t totalPixels = (size_t)size.x * size.y;
        std::string resLabel = std::to_string(size.x) + "x" + std::to_string(size.y);

        std::cout << "\n========================================" << std::endl;
        std::cout << "ELABORAZIONE RISOLUZIONE: " << resLabel << std::endl;
        std::cout << "========================================" << std::endl;

        std::vector<float> hostLuminance(totalPixels);
        for (unsigned int y = 0; y < size.y; ++y) {
            for (unsigned int x = 0; x < size.x; ++x) {
                sf::Color color = inputImage.getPixel(x, y);
                // calcola la luminanza del pixel estraendo i canali di colore
                hostLuminance[(size_t)y * size.x + x] = (0.299f * color.r + 0.587f * color.g + 0.114f * color.b) / 255.0f;
            }
        }

        // esegue il ciclo per testare le diverse configurazioni di soglia
        for (float threshold : thresholds) {
            std::cout << "\n--- Configurazione con Soglia: " << threshold << " ---" << std::endl;

            // esegue pipeline sequenziale 5 volte e fa la media dei tempi
            std::vector<double> seqTimes(RUNS);
            std::vector<float> referenceOutput;
            for (int r = 0; r < RUNS; ++r) {
                auto startTiming = std::chrono::high_resolution_clock::now();
                referenceOutput = runCPUImplementation(hostLuminance, size.x, size.y, threshold);
                auto endTiming = std::chrono::high_resolution_clock::now();
                seqTimes[r] = std::chrono::duration<double, std::milli>(endTiming - startTiming).count();
            }
            // calcola i tempi massimi, minimi, medi e la deviazione standard per la baseline
            double seqMin = *std::min_element(seqTimes.begin(), seqTimes.end());
            double seqMax = *std::max_element(seqTimes.begin(), seqTimes.end());
            double seqAvg = std::accumulate(seqTimes.begin(), seqTimes.end(), 0.0) / RUNS;
            double seqStdDev = calculateStdDev(seqTimes, seqAvg);

            std::cout << "\nBaseline Sequenziale CPU: " << std::fixed << std::setprecision(2)
                      << seqAvg << " ms (±" << seqStdDev << " ms)" << std::endl;
            // scrive i risultati della baseline sul file CSV
            if (csvFile.is_open()) {
                csvFile << resLabel << ",CPU_Sequenziale,1," << threshold << "," << seqAvg << ","
                        << seqMin << "," << seqMax << "," << seqStdDev << ",1.0,SI\n";
            }

            // funzione di verifica dell'output con i risultati di riferimento
            auto checkMatch = [&](const std::vector<float>& targetBuffer) -> std::string {
                for (size_t i = 0; i < totalPixels; ++i) {
                    if (std::abs(referenceOutput[i] - targetBuffer[i]) > 0.01f) return "NO";
                }
                return "SI";
            };

            std::cout << "\nPerformance Multi Core OpenMP" << std::endl;
            std::cout << std::setw(10) << "Threads" << std::setw(12) << "Medio ms" << std::setw(10) << "Min ms"
                      << std::setw(10) << "Max ms" << std::setw(10) << "DevStd" << std::setw(10) << "Speedup" << std::setw(8) << "Match" << std::endl;
            std::cout << "----------------------------------------------------------------------------------------" << std::endl;
            // esegue pipeline OpenMP e calcola i tempi
            for (int threads : threadCounts) {
                std::vector<double> ompTimes(RUNS);
                std::vector<float> openmpOutput;
                for (int r = 0; r < RUNS; ++r) {
                    auto startTiming = std::chrono::high_resolution_clock::now();
                    openmpOutput = runOpenMPImplementation(hostLuminance, size.x, size.y, threshold, threads);
                    auto endTiming = std::chrono::high_resolution_clock::now();
                    ompTimes[r] = std::chrono::duration<double, std::milli>(endTiming - startTiming).count();
                }
                // calcola i parametri statistici e lo speedup per OpenMP
                double ompMin = *std::min_element(ompTimes.begin(), ompTimes.end());
                double ompMax = *std::max_element(ompTimes.begin(), ompTimes.end());
                double ompAvg = std::accumulate(ompTimes.begin(), ompTimes.end(), 0.0) / RUNS;
                double ompStdDev = calculateStdDev(ompTimes, ompAvg);
                double ompSpeedup = seqAvg / ompAvg;
                // verifica l'output con i risultati di riferimento
                std::string identityMatch = checkMatch(openmpOutput);

                std::cout << std::setw(10) << threads << std::setw(12) << ompAvg << std::setw(10) << ompMin
                          << std::setw(10) << ompMax << std::setw(10) << ompStdDev << std::setw(9) << ompSpeedup << "x" << std::setw(8) << identityMatch << std::endl;
                // scrive i risultati su file CSV
                if (csvFile.is_open()) {
                    csvFile << resLabel << ",CPU_OpenMP," << threads << "," << threshold << "," << ompAvg << ","
                            << ompMin << "," << ompMax << "," << ompStdDev << "," << ompSpeedup << "," << identityMatch << "\n";
                }
            }

            std::cout << "\nPerformance Scheda Video CUDA" << std::endl;
            std::cout << std::setw(10) << "Blocco" << std::setw(12) << "Medio ms" << std::setw(10) << "Min ms"
                      << std::setw(10) << "Max ms" << std::setw(10) << "DevStd" << std::setw(10) << "Speedup" << std::setw(8) << "Match" << std::endl;
            std::cout << "----------------------------------------------------------------------------------------" << std::endl;
            // esegue pipeline CUDA con le varie configurazioni e calcola i tempi
            for (auto executionBlock : blockSizes) {
                std::vector<double> cudaTimes(RUNS);
                std::vector<float> cudaOutput;
                for (int r = 0; r < RUNS; ++r) {
                    auto startTiming = std::chrono::high_resolution_clock::now();
                    cudaOutput = runCudaSingleTest(hostLuminance, size.x, size.y, threshold, executionBlock.first, executionBlock.second);
                    auto endTiming = std::chrono::high_resolution_clock::now();
                    cudaTimes[r] = std::chrono::duration<double, std::milli>(endTiming - startTiming).count();
                }
                // calcola i parametri statistici e lo speedup per CUDA
                double cudaMin = *std::min_element(cudaTimes.begin(), cudaTimes.end());
                double cudaMax = *std::max_element(cudaTimes.begin(), cudaTimes.end());
                double cudaAvg = std::accumulate(cudaTimes.begin(), cudaTimes.end(), 0.0) / RUNS;
                double cudaStdDev = calculateStdDev(cudaTimes, cudaAvg);
                double cudaSpeedup = seqAvg / cudaAvg;
                // verifica l'output con i risultati di riferimento
                std::string identityMatch = checkMatch(cudaOutput);
                std::string blockLabel = std::to_string(executionBlock.first) + "x" + std::to_string(executionBlock.second);

                std::cout << std::setw(10) << blockLabel << std::setw(12) << cudaAvg << std::setw(10) << cudaMin
                          << std::setw(10) << cudaMax << std::setw(10) << cudaStdDev << std::setw(9) << cudaSpeedup << "x" << std::setw(8) << identityMatch << std::endl;
                // scrive i risultati su file CSV
                if (csvFile.is_open()) {
                    csvFile << resLabel << ",CUDA_SchedaVideo," << blockLabel << "," << threshold << "," << cudaAvg << ","
                            << cudaMin << "," << cudaMax << "," << cudaStdDev << "," << cudaSpeedup << "," << identityMatch << "\n";
                }
            }
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
    int blobRadius = 4;
    // carica la prima immagine della serie e ne calcola la luminosità e dimensione in pixel
    sf::Image inputImage;
    if (!inputImage.loadFromFile("../images/input_512.png")) return;
    sf::Vector2u size = inputImage.getSize();
    size_t totalPixels = (size_t)size.x * size.y;

    std::vector<float> hostLuminance(totalPixels);
    for (unsigned int y = 0; y < size.y; ++y) {
        for (unsigned int x = 0; x < size.x; ++x) {
            sf::Color color = inputImage.getPixel(x, y);
            // calcola la luminanza del pixel estraendo i canali di colore
            hostLuminance[(size_t)y * size.x + x] = (0.299f * color.r + 0.587f * color.g + 0.114f * color.b) / 255.0f;
        }
    }

    float threshold = 0.017f;
    // esegue la pipeline CUDA
    std::vector<float> blobMap = runCudaSingleTest(hostLuminance, size.x, size.y, threshold, 16, 16);
    // crea l'immagine finale da quella iniziale ma riducendo la luminosità per dare contrasto ai blob
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
    // salva l'immagine finale
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
    // esegue il controllo preliminare e genera le risoluzioni mancanti prima di iniziare i test
    checkAndGenerateImages();
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