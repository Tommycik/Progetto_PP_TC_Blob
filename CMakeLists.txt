cmake_minimum_required(VERSION 3.24)

project(BlobDetectionCUDA LANGUAGES CUDA CXX)

# standard C++ e CUDA impostati a C++17
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# forza la compilazione in modalità Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Scegli il tipo di build." FORCE)
endif()

# cerca il toolkit CUDA installato sul sistema
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

# SFML
include(FetchContent)
set(SFML_BUILD_AUDIO FALSE CACHE BOOL "" FORCE)
set(SFML_BUILD_NETWORK FALSE CACHE BOOL "" FORCE)
set(SFML_BUILD_EXAMPLES FALSE CACHE BOOL "" FORCE)
set(SFML_BUILD_DOCS FALSE CACHE BOOL "" FORCE)

FetchContent_Declare(
        SFML
        GIT_REPOSITORY https://github.com/SFML/SFML.git
        GIT_TAG 2.6.1
)
FetchContent_MakeAvailable(SFML)

add_executable(BlobDetection Src/main.cu
        Src/Kernels.cuh
        Src/Kernels.cu
        Src/CpuPipelines.hpp
        Src/Common.hpp
        Src/GpuPipelines.cuh
)
# Indica a CMake di cercare gli header sia nella cartella radice che dentro Src
target_include_directories(BlobDetection PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/Src
)

# proprietà del target
set_target_properties(BlobDetection PROPERTIES
        CUDA_ARCHITECTURES 75
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_LINK_WITH_NVCC ON
)
# flag OpenMP e ottimizzazioni per host e device solo in modalità Release
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(BlobDetection PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-march=native>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-fopenmp>
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-Xcompiler=-fopenmp,-O3,-march=native>
    )
elseif(MSVC)
    target_compile_options(BlobDetection PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/openmp>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/Oi>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/Ot>
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-Xcompiler=/openmp,/O2,/Oi,/Ot>
    )
endif()
# link delle librerie
target_link_libraries(BlobDetection
        PRIVATE
        sfml-graphics
        sfml-window
        sfml-system
        CUDA::cudart
        OpenMP::OpenMP_CXX
)

# copia automatica delle dll
add_custom_command(TARGET BlobDetection POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_RUNTIME_DLLS:BlobDetection>
        $<TARGET_FILE_DIR:BlobDetection>
        COMMAND_EXPAND_LISTS
)