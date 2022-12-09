#include "volumevop1.h"
#include "volumevop1.cuh"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
/**
 * Check for CUDA errors; print and exit if there was a problem.
 */
void checkCUDAError(const char *msg, int line = -1)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        if (line >= 0)
        {
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

__global__ void CodeGenerator::CUDAKernel::volumevop1(float geo1_volumevop1_input8_input5, float geo1_volumevop1_input7_input4, float geo1_volumevop1_input5_input3, float geo1_volumevop1_input3_input2, CGGeometry::RawData geo1_volumevop1_volumevopglobal1_OpInput1, CGGeometry::RawData geo1_volumevop1_volumevopglobal1_OpInput2, int numThreads)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index > numThreads)
        return;

    CodeGenerator::GenericCode::volumevop1(
        geo1_volumevop1_input8_input5, 
        geo1_volumevop1_input7_input4,
        geo1_volumevop1_input5_input3, 
        geo1_volumevop1_input3_input2, 
        geo1_volumevop1_volumevopglobal1_OpInput1, 
        geo1_volumevop1_volumevopglobal1_OpInput2, index);
}

void CodeGenerator::CUDA::volumevop1(
    float geo1_volumevop1_input8_input5, 
    float geo1_volumevop1_input7_input4, 
    float geo1_volumevop1_input5_input3, 
    float geo1_volumevop1_input3_input2, 
    CGGeometry *geo1_volumevop1_volumevopglobal1_OpInput1, 
    CGGeometry *geo1_volumevop1_volumevopglobal1_OpInput2,
    int blockSize)
{
    // Buffer malloc

    // Compute threads num
    int numOfThreads = geo1_volumevop1_volumevopglobal1_OpInput1->getSize();
    // Compute blocks num
    auto num_blocks_threads = ThreadBlockInfo(blockSize, numOfThreads);

    // Kernel launch
    CodeGenerator::CUDAKernel::volumevop1<<<num_blocks_threads.x, num_blocks_threads.y>>>(
        geo1_volumevop1_input8_input5, 
        geo1_volumevop1_input7_input4, 
        geo1_volumevop1_input5_input3,
        geo1_volumevop1_input3_input2, 
        geo1_volumevop1_volumevopglobal1_OpInput1->GetGeometryRawDataDevice(), 
        geo1_volumevop1_volumevopglobal1_OpInput2->GetGeometryRawDataDevice(),
        numOfThreads);

    checkCUDAErrorWithLine("volumevop1 error");

    cudaDeviceSynchronize();
}