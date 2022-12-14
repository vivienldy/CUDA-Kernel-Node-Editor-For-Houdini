#include "ParticleAdvect.h"
#include "ParticleAdvect.cuh"

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

__global__ void CodeGenerator::CUDAKernel::ParticleAdvect(float geo1_ParticleAdvect_parameter_time, float geo1_ParticleAdvect_input3_input3, float geo1_ParticleAdvect_input2_input2, glm::vec3* geo1_ParticleAdvect_geometryvopglobal1_Pbuffer, float* geo1_ParticleAdvect_geometryvopglobal1_agebuffer, glm::vec3* geo1_ParticleAdvect_geometryvopglobal1_Cdbuffer, float geo1_ParticleAdvect_geometryvopglobal1_TimeInc, CGGeometry::RAWData geo1_ParticleAdvect_geometryvopglobal1_OpInput1, CGGeometry::RAWData geo1_ParticleAdvect_geometryvopglobal1_OpInput2, float* __geo1_ParticleAdvect_add1_sum_debug_buffer,  int numThreads)
{
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if(index > numThreads)
    return;

  CodeGenerator::GenericCode::ParticleAdvect(geo1_ParticleAdvect_parameter_time, geo1_ParticleAdvect_input3_input3, geo1_ParticleAdvect_input2_input2, geo1_ParticleAdvect_geometryvopglobal1_Pbuffer, geo1_ParticleAdvect_geometryvopglobal1_agebuffer, geo1_ParticleAdvect_geometryvopglobal1_Cdbuffer, geo1_ParticleAdvect_geometryvopglobal1_TimeInc, geo1_ParticleAdvect_geometryvopglobal1_OpInput1, geo1_ParticleAdvect_geometryvopglobal1_OpInput2, __geo1_ParticleAdvect_add1_sum_debug_buffer,  index);
}

void CodeGenerator::CUDA::ParticleAdvect (
    float geo1_ParticleAdvect_parameter_time, float geo1_ParticleAdvect_input3_input3, float geo1_ParticleAdvect_input2_input2, CGBuffer<glm::vec3>* geo1_ParticleAdvect_geometryvopglobal1_Pbuffer, CGBuffer<float>* geo1_ParticleAdvect_geometryvopglobal1_agebuffer, CGBuffer<glm::vec3>* geo1_ParticleAdvect_geometryvopglobal1_Cdbuffer, float geo1_ParticleAdvect_geometryvopglobal1_TimeInc, CGGeometry* geo1_ParticleAdvect_geometryvopglobal1_OpInput1, CGGeometry* geo1_ParticleAdvect_geometryvopglobal1_OpInput2, CGBuffer<float>* __geo1_ParticleAdvect_add1_sum_debug_buffer, 
    int blockSize)
{
    // Buffer malloc
    geo1_ParticleAdvect_geometryvopglobal1_Pbuffer->malloc();
geo1_ParticleAdvect_geometryvopglobal1_Pbuffer->loadHostToDevice();

geo1_ParticleAdvect_geometryvopglobal1_agebuffer->malloc();
geo1_ParticleAdvect_geometryvopglobal1_agebuffer->loadHostToDevice();

geo1_ParticleAdvect_geometryvopglobal1_Cdbuffer->malloc();
geo1_ParticleAdvect_geometryvopglobal1_Cdbuffer->loadHostToDevice();

__geo1_ParticleAdvect_add1_sum_debug_buffer->malloc();
__geo1_ParticleAdvect_add1_sum_debug_buffer->loadHostToDevice();

geo1_ParticleAdvect_geometryvopglobal1_OpInput1->DeviceMalloc();
geo1_ParticleAdvect_geometryvopglobal1_OpInput1->LoadToDevice();

geo1_ParticleAdvect_geometryvopglobal1_OpInput2->DeviceMalloc();
geo1_ParticleAdvect_geometryvopglobal1_OpInput2->LoadToDevice();



    // Compute threads num
    int numOfThreads = geo1_ParticleAdvect_geometryvopglobal1_Pbuffer->getSize();
    // Compute blocks num
    auto num_blocks_threads = ThreadBlockInfo(blockSize, numOfThreads);
    
    // Kernel launch
    CodeGenerator::CUDAKernel::ParticleAdvect<<<num_blocks_threads.x, num_blocks_threads.y>>>(
        geo1_ParticleAdvect_parameter_time, geo1_ParticleAdvect_input3_input3, geo1_ParticleAdvect_input2_input2, geo1_ParticleAdvect_geometryvopglobal1_Pbuffer->getDevicePointer(), geo1_ParticleAdvect_geometryvopglobal1_agebuffer->getDevicePointer(), geo1_ParticleAdvect_geometryvopglobal1_Cdbuffer->getDevicePointer(), geo1_ParticleAdvect_geometryvopglobal1_TimeInc, geo1_ParticleAdvect_geometryvopglobal1_OpInput1->GetGeometryRawDataDevice(), geo1_ParticleAdvect_geometryvopglobal1_OpInput2->GetGeometryRawDataDevice(), __geo1_ParticleAdvect_add1_sum_debug_buffer->getDevicePointer(),  numOfThreads);

    checkCUDAErrorWithLine("ParticleAdvect error");

    cudaDeviceSynchronize();
}