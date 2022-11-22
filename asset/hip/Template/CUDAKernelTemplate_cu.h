#include "#OUT_FILE_NAME#.h"
#include "#OUT_FILE_NAME#.cuh"

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

__global__ void CodeGenerator::CUDAKernel::#KERNEL_NAME#(#KERNEL_PARM_DECLARE_LIST#, int numThreads)
{
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if(index > numThreads)
    return;

  CodeGenerator::GenericCode::#FUNC_NAME#(#PARM_LIST#, index);
}

void CodeGenerator::CUDA::#KERNEL_LAUNCH_NAME# (
    #KERNEL_LAUNCH_PARM_DECLARE_LIST#,
    int blockSize)
{
    // Buffer malloc
    #BUFFER_MALLOC#

    // Compute threads num
    int numOfThreads = #COMP_BLK_NUM#
    // Compute blocks num
    auto num_blocks_threads = ThreadBlockInfo(blockSize, numOfThreads);
    
    // Kernel launch
    CodeGenerator::CUDAKernel::#KERNEL_NAME#<<<num_blocks_threads.x, num_blocks_threads.y>>>(
        #KERNEL_PARM_INPUT_LIST#, numOfThreads);

    checkCUDAErrorWithLine(#KERNEL_LAUNCH_ERROR_MSG#);

    cudaDeviceSynchronize();
}