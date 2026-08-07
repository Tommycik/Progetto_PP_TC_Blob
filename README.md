# Multi-Scale Blob Detection with CPU, OpenMP and CUDA

## Project overview

This project implements a multi-scale blob detector and compares three execution paths. The first path is sequential CPU. The second path parallelizes the CPU stages with OpenMP. The third path executes the same main pipeline on a CUDA-capable GPU.

The detector builds a Gaussian scale space and calculates Difference of Gaussian maps. Candidate pixels are selected by comparing each response with the neighbours in the current, previous and next scale. A threshold removes weak responses before the complete neighbourhood test. Non-maximum suppression is then used to keep only the strongest spatial detections.

The project is designed both as an image-processing application and as a parallel-computing benchmark. It measures the effect of image resolution, threshold, OpenMP execution and CUDA block geometry.

## Processing pipeline

The main stages are:

1. Load the input image and convert RGB values to normalized luminance.
2. Generate several Gaussian-blurred images with increasing sigma values.
3. Subtract consecutive Gaussian images to build the Difference of Gaussian maps.
4. Search the three-dimensional neighborhood formed by position and scale.
5. Reject responses whose absolute value is below the selected threshold.
6. Store the strength of local extrema.
7. Apply spatial non-maximum suppression.
8. Return a binary map containing the final blob locations.

The sequential, OpenMP and CUDA paths use the same input luminance and the same threshold. Their final maps are compared with the sequential result.

## Implementations

### Sequential CPU

The sequential implementation is the reference used for timing and correctness. It executes Gaussian filtering, Difference of Gaussian generation, extrema detection and non-maximum suppression on one CPU thread.

### OpenMP CPU

The OpenMP implementation parallelizes the independent image positions. It uses 12 threads in the benchmark. The Gaussian passes, Difference of Gaussian construction, extrema search and non-maximum suppression are executed with OpenMP directives where the data dependencies allow it.

### CUDA GPU

The CUDA implementation uses custom kernels for the main image-processing stages. The benchmark changes the CUDA block shape while keeping the algorithm and input parameters fixed. This makes it possible to study both the number of threads per block and the orientation of the block in row-major image memory.

## Input images

The base image must be stored as:

```text
images/input_512.png
```

The executable is normally started from the CLion build directory. The source code therefore accesses the image through `../images/input_512.png`.

At startup the program checks the following files:

```text
images/input_512.png
images/input_1024.png
images/input_2048.png
images/input_4096.png
```

If the larger files are missing,  they are generated from the 512 by 512 base image. The generated images are used to study resolution scaling with the same source content.

## Graphical mode

The graphical mode processes the 512 by 512 image with CUDA. It uses threshold `0.017` and a `16 x 16` CUDA block. Detected points are drawn as red circles over a darker copy of the original image.

The result is saved as:

```text
risultato_blobs.png
```

An SFML window then displays the generated image.

## Benchmark

The benchmark repeats every configuration five times. It calculates the mean, minimum, maximum and population standard deviation of the measured times.

The tested resolutions are:

- 512 by 512;
- 1024 by 1024;
- 2048 by 2048;
- 4096 by 4096.

The tested thresholds are:

- `0.0005`;
- `0.005`;
- `0.017`;
- `0.05`.

The OpenMP benchmark uses 12 threads. CUDA evaluates these block layouts:

```text
8x8
16x16
32x32
64x16
16x64
128x8
8x128
256x4
4x256
1024x1
1x1024
```

The layouts include symmetric, horizontal and vertical configurations. This is useful because adjacent threads should preferably access adjacent elements of the row-major image.

## Verification

The sequential output is used as the reference. Every OpenMP and CUDA result is compared pixel by pixel. The CSV records whether the output matches, the number of different pixels and the maximum absolute difference.

A difference must not be interpreted only from the final boolean field. CPU and GPU floating-point operations can produce small changes close to the threshold or to an equality decision. The number of different pixels and the maximum difference provide the information required to distinguish a sparse numerical variation from a systematic algorithmic error.

## CSV output

The benchmark creates:

```text
benchmark_results.csv
```

The file records:

- resolution;
- backend;
- thread count or CUDA block layout;
- threshold;
- mean time;
- minimum time;
- maximum time;
- standard deviation;
- speedup against the sequential CPU baseline;
- verification result;
- number of different pixels;
- maximum difference.

The CSV is created in the program working directory. With the standard CLion configuration this is normally the Release build directory.

## Project structure

- `Src/main.cu` contains the menu, image preparation, benchmark and graphical mode.
- `Src/CpuPipelines.hpp` contains the sequential and OpenMP implementations.
- `Src/GpuPipelines.cuh` contains the CUDA pipeline wrapper and device memory management.
- `Src/Kernels.cu` contains the CUDA kernels.
- `Src/Kernels.cuh` declares the CUDA kernels.
- `Src/Common.hpp` contains the shared constants.
- `CMakeLists.txt` configures C++17, CUDA, OpenMP and SFML.

## Requirements

The project requires:

- a C++17 compiler supported by the installed CUDA toolkit;
- CMake 3.24 or newer;
- the NVIDIA CUDA toolkit;
- an NVIDIA GPU compatible with the configured CUDA architecture;
- OpenMP support;
- an internet connection during the first configuration because SFML 2.6.1 is downloaded through `FetchContent`.

The current CMake file uses CUDA architecture 75. Change `CUDA_ARCHITECTURES` only when the target GPU requires another architecture.

## Use from CLion

1. Open the project directory in CLion.
2. Select the compiler and CUDA toolkit used by the project.
3. Use the Release configuration.
4. Wait for CMake to configure SFML, OpenMP and CUDA.
5. Select and run the `BlobDetection` target.

No command-line arguments are required. The application displays:

```text
1. Pipeline Benchmark
2. Finestra Grafica
Scelta:
```

Choose option 1 to generate the CSV. Choose option 2 to create and display `risultato_blobs.png`.

## Result interpretation

The benchmark contains two different comparisons. OpenMP speedup shows the benefit of CPU thread parallelism. CUDA speedup includes the effect of GPU execution and the timing scope implemented by the project. Block-shape results must be compared at the same resolution and threshold.

A faster block layout at one resolution is not automatically the best layout for every workload. Image dimensions, occupancy, memory access direction and fixed launch overhead can change the preferred configuration.
