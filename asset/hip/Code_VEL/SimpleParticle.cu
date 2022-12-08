#include "@OUT_FILE_NAME@.h"
#include "@OUT_FILE_NAME@.cuh"

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

__global__ void CodeGenerator::CUDAKernel::particle_advect(int geo1_solver1_d_s_particle_advect_DEBUG_particle_advect_turb_turb, float geo1_solver1_d_s_particle_advect_DEBUG_particle_advect_amp_amp, glm::vec3 geo1_solver1_d_s_particle_advect_DEBUG_particle_advect_freq_freq, glm::vec3* geo1_solver1_d_s_particle_advect_DEBUG_particle_advect_geometryvopglobal1_Pbuffer, glm::vec3* geo1_solver1_d_s_particle_advect_DEBUG_particle_advect_geometryvopglobal1_vbuffer, float geo1_solver1_d_s_particle_advect_DEBUG_particle_advect_geometryvopglobal1_TimeInc, glm::vec3* __geo1_solver1_d_s_particle_advect_DEBUG_particle_advect_add2_sum_debug_buffer, glm::vec3* __geo1_solver1_d_s_particle_advect_DEBUG_particle_advect_multiply2_product_debug_buffer, glm::vec3* __geo1_solver1_d_s_particle_advect_DEBUG_particle_advect_add1_sum_debug_buffer, glm::vec3* __geo1_solver1_d_s_particle_advect_DEBUG_particle_advect_multiply1_product_debug_buffer, glm::vec3* __geo1_solver1_d_s_particle_advect_DEBUG_particle_advect_fit1_shift_debug_buffer, glm::vec3* __geo1_solver1_d_s_particle_advect_DEBUG_particle_advect_multiply3_product_debug_buffer,  int numThreads)
{
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if(index > numThreads)
    return;

  CodeGenerator::GenericCode::@FUNC_NAME@(@SHARE_CODE_PARM_INPUT_LIST@ index);
}

void CodeGenerator::CUDA::@KERNEL_LAUNCH_NAME@ (
    @KERNEL_LAUNCH_PARM_DECLARE_LIST@
    int blockSize)
{
    // Buffer malloc
    @BUFFER_MALLOC@

    // Compute threads num
    int numOfThreads = @REF_BUFFER_NAME@->getSize();
    // Compute blocks num
    auto num_blocks_threads = ThreadBlockInfo(blockSize, numOfThreads);
    
    // Kernel launch
    CodeGenerator::CUDAKernel::particle_advect<<<num_blocks_threads.x, num_blocks_threads.y>>>(
        @KERNEL_PARM_INPUT_LIST@ numOfThreads);

    checkCUDAErrorWithLine("@KERNEL_LAUNCH_ERROR_MSG@");

    cudaDeviceSynchronize();
}