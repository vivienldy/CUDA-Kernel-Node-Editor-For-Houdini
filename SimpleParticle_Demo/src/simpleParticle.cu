#include "simpleParticle.h"
#include "simpleParticle.cuh"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char* msg, int line = -1) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        if (line >= 0) {
            fprintf(stderr, "Line %d: ", line);
        }
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

glm::vec2 ThreadBlockInfo(int blockSize, int numThreads)
{
    return glm::vec2(int(numThreads / blockSize) + 1, blockSize > numThreads ? numThreads : blockSize);
}

__global__ void CodeGenerator::CUDAKernel::simpleParticle(glm::vec3* geo1_simpleParticle_geometryvopglobal1_Pbuffer, glm::vec3* geo1_simpleParticle_geometryvopglobal1_vbuffer, float geo1_simpleParticle_geometryvopglobal1_TimeInc,  int numThreads)
{
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if(index > numThreads)
    return;

  CodeGenerator::GenericCode::simpleParticle(geo1_simpleParticle_geometryvopglobal1_Pbuffer, geo1_simpleParticle_geometryvopglobal1_vbuffer, geo1_simpleParticle_geometryvopglobal1_TimeInc,  index);
}

void CodeGenerator::CUDA::simpleParticle (
    CGBuffer<glm::vec3>* geo1_simpleParticle_geometryvopglobal1_Pbuffer, CGBuffer<glm::vec3>* geo1_simpleParticle_geometryvopglobal1_vbuffer, float geo1_simpleParticle_geometryvopglobal1_TimeInc, 
    int blockSize)
{
    // Buffer malloc
    geo1_simpleParticle_geometryvopglobal1_Pbuffer->malloc();
geo1_simpleParticle_geometryvopglobal1_Pbuffer->loadHostToDevice();

geo1_simpleParticle_geometryvopglobal1_vbuffer->malloc();
geo1_simpleParticle_geometryvopglobal1_vbuffer->loadHostToDevice();



    // Compute threads num
    int numOfThreads = geo1_simpleParticle_geometryvopglobal1_Pbuffer->getSize();
    // Compute blocks num
    auto num_blocks_threads = ThreadBlockInfo(blockSize, numOfThreads);
    
    // Kernel launch
    CodeGenerator::CUDAKernel::simpleParticle<<<num_blocks_threads.x, num_blocks_threads.y>>>(
        geo1_simpleParticle_geometryvopglobal1_Pbuffer->getDevicePointer(), geo1_simpleParticle_geometryvopglobal1_vbuffer->getDevicePointer(), geo1_simpleParticle_geometryvopglobal1_TimeInc,  numOfThreads);

    checkCUDAErrorWithLine("simpleParticle error");

    cudaDeviceSynchronize();
}