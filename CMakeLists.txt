cmake_minimum_required(VERSION 3.24)

# Il nome del progetto e i linguaggi abilitati
project(cuda_hello LANGUAGES CUDA CXX)

# Standard C++ e CUDA impostati a C++17
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Flag per evitare blocchi se il compilatore host (es. Visual Studio 2026)
# è più recente del toolkit CUDA 13.3.1 rilevato
set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler"
)

# Cerca il toolkit CUDA installato sul sistema in modo dinamico
find_package(CUDAToolkit REQUIRED)

# Definizione dell'eseguibile
add_executable(cuda_hello
        main.cu
)

# La RTX 2070 Super è basata su architettura Turing (Compute Capability 7.5)
set_target_properties(cuda_hello PROPERTIES
        CUDA_ARCHITECTURES 75
)

# Link della libreria CUDA Runtime
target_link_libraries(cuda_hello
        CUDA::cudart
)