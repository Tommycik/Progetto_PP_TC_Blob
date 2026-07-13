#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const int NUM_SCALES = 6;
const int NUM_DOGS = NUM_SCALES - 1;
const float SIGMA_BASE = 1.05f;
const float K_FACTOR = 1.45f;

#endif
