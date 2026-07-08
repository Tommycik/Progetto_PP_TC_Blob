cmake_minimum_required(VERSION 3.18)
project(BlobDetectionCUDA CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# pacchetti necessari
find_package(CUDAToolkit REQUIRED)
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# file sorgente ( .cu per includere kernel CUDA e C++ normale)
add_executable(BlobDetection main.cu)

# librerie
target_link_libraries(BlobDetection
        PRIVATE
        sfml-graphics
        sfml-window
        sfml-system
        CUDA::cudart
)

# proprietà CUDA
set_target_properties(BlobDetection PROPERTIES CUDA_SEPARABLE_COMPILATION ON)