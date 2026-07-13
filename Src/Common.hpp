#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const int NUM_SCALES = 4;
const int NUM_DOGS = NUM_SCALES - 1;
const float SIGMA_BASE = 1.0f;
const float K_FACTOR = 1.6f;

#define CUDA_CHECK(call) \
do { \
cudaError_t err = call; \
if (err != cudaSuccess) { \
std::cerr << "\nErrore hardware alla riga: " << __LINE__ << std::endl; \
exit(1); \
} \
} while (0)

#endif