#pragma once

#include <cuda.h>

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <stdio.h>
#include <cuda.h>
#include <cmath>

#include <glm/glm.hpp>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

//#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
///**
//* Check for CUDA errors; print and exit if there was a problem.
//*/
//void checkCUDAError(const char* msg, int line = -1) {
//    cudaError_t err = cudaGetLastError();
//    if (cudaSuccess != err) {
//        if (line >= 0) {
//            fprintf(stderr, "Line %d: ", line);
//        }
//        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
//        exit(EXIT_FAILURE);
//    }
//}

glm::vec2 inline ThreadBlockInfo(int blockSize, int numThreads)
{
    return glm::vec2(int(numThreads / blockSize) + 1, blockSize > numThreads ? numThreads : blockSize);
}