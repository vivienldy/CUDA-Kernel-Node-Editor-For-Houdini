#include "SimpleParticle.h"
#include "SimpleParticle.cuh"

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

__global__ void CodeGenerator::CUDAKernel::particle_advect(int geo1_solver1_d_s_particle_advect_turb_turb, float geo1_solver1_d_s_particle_advect_amp_amp, glm::vec3 geo1_solver1_d_s_particle_advect_freq_freq, glm::vec3* geo1_solver1_d_s_particle_advect_geometryvopglobal1_Pbuffer, glm::vec3* geo1_solver1_d_s_particle_advect_geometryvopglobal1_vbuffer, float geo1_solver1_d_s_particle_advect_geometryvopglobal1_TimeInc, glm::vec3* __geo1_solver1_d_s_particle_advect_add2_sum_debug_buffer, glm::vec3* __geo1_solver1_d_s_particle_advect_multiply2_product_debug_buffer, glm::vec3* __geo1_solver1_d_s_particle_advect_add1_sum_debug_buffer, glm::vec3* __geo1_solver1_d_s_particle_advect_multiply1_product_debug_buffer, glm::vec3* __geo1_solver1_d_s_particle_advect_fit1_shift_debug_buffer, glm::vec3* __geo1_solver1_d_s_particle_advect_multiply3_product_debug_buffer, glm::vec3* geo1_solver1_d_s_particle_advect_geometryvopoutput1_Pbuffer,  int numThreads)
{
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if(index > numThreads)
    return;

  CodeGenerator::GenericCode::particle_advect(geo1_solver1_d_s_particle_advect_turb_turb, geo1_solver1_d_s_particle_advect_amp_amp, geo1_solver1_d_s_particle_advect_freq_freq, geo1_solver1_d_s_particle_advect_geometryvopglobal1_Pbuffer, geo1_solver1_d_s_particle_advect_geometryvopglobal1_vbuffer, geo1_solver1_d_s_particle_advect_geometryvopglobal1_TimeInc, __geo1_solver1_d_s_particle_advect_add2_sum_debug_buffer, __geo1_solver1_d_s_particle_advect_multiply2_product_debug_buffer, __geo1_solver1_d_s_particle_advect_add1_sum_debug_buffer, __geo1_solver1_d_s_particle_advect_multiply1_product_debug_buffer, __geo1_solver1_d_s_particle_advect_fit1_shift_debug_buffer, __geo1_solver1_d_s_particle_advect_multiply3_product_debug_buffer, geo1_solver1_d_s_particle_advect_geometryvopoutput1_Pbuffer,  index);
}

void CodeGenerator::CUDA::particle_advect (
    int geo1_solver1_d_s_particle_advect_turb_turb, float geo1_solver1_d_s_particle_advect_amp_amp, glm::vec3 geo1_solver1_d_s_particle_advect_freq_freq, CGBuffer<glm::vec3>* geo1_solver1_d_s_particle_advect_geometryvopglobal1_Pbuffer, CGBuffer<glm::vec3>* geo1_solver1_d_s_particle_advect_geometryvopglobal1_vbuffer, float geo1_solver1_d_s_particle_advect_geometryvopglobal1_TimeInc, CGBuffer<glm::vec3>* __geo1_solver1_d_s_particle_advect_add2_sum_debug_buffer, CGBuffer<glm::vec3>* __geo1_solver1_d_s_particle_advect_multiply2_product_debug_buffer, CGBuffer<glm::vec3>* __geo1_solver1_d_s_particle_advect_add1_sum_debug_buffer, CGBuffer<glm::vec3>* __geo1_solver1_d_s_particle_advect_multiply1_product_debug_buffer, CGBuffer<glm::vec3>* __geo1_solver1_d_s_particle_advect_fit1_shift_debug_buffer, CGBuffer<glm::vec3>* __geo1_solver1_d_s_particle_advect_multiply3_product_debug_buffer, CGBuffer<glm::vec3>* geo1_solver1_d_s_particle_advect_geometryvopoutput1_Pbuffer, 
    int blockSize)
{
    // Buffer malloc
    geo1_solver1_d_s_particle_advect_geometryvopglobal1_Pbuffer->malloc();
geo1_solver1_d_s_particle_advect_geometryvopglobal1_Pbuffer->loadHostToDevice();

geo1_solver1_d_s_particle_advect_geometryvopglobal1_vbuffer->malloc();
geo1_solver1_d_s_particle_advect_geometryvopglobal1_vbuffer->loadHostToDevice();

__geo1_solver1_d_s_particle_advect_add2_sum_debug_buffer->malloc();
__geo1_solver1_d_s_particle_advect_add2_sum_debug_buffer->loadHostToDevice();

__geo1_solver1_d_s_particle_advect_multiply2_product_debug_buffer->malloc();
__geo1_solver1_d_s_particle_advect_multiply2_product_debug_buffer->loadHostToDevice();

__geo1_solver1_d_s_particle_advect_add1_sum_debug_buffer->malloc();
__geo1_solver1_d_s_particle_advect_add1_sum_debug_buffer->loadHostToDevice();

__geo1_solver1_d_s_particle_advect_multiply1_product_debug_buffer->malloc();
__geo1_solver1_d_s_particle_advect_multiply1_product_debug_buffer->loadHostToDevice();

__geo1_solver1_d_s_particle_advect_fit1_shift_debug_buffer->malloc();
__geo1_solver1_d_s_particle_advect_fit1_shift_debug_buffer->loadHostToDevice();

__geo1_solver1_d_s_particle_advect_multiply3_product_debug_buffer->malloc();
__geo1_solver1_d_s_particle_advect_multiply3_product_debug_buffer->loadHostToDevice();

geo1_solver1_d_s_particle_advect_geometryvopoutput1_Pbuffer->malloc();
geo1_solver1_d_s_particle_advect_geometryvopoutput1_Pbuffer->loadHostToDevice();



    // Compute threads num
    int numOfThreads = Pbuffer->getSize();
    // Compute blocks num
    auto num_blocks_threads = ThreadBlockInfo(blockSize, numOfThreads);
    
    // Kernel launch
    CodeGenerator::CUDAKernel::particle_advect<<<num_blocks_threads.x, num_blocks_threads.y>>>(
        geo1_solver1_d_s_particle_advect_turb_turb, geo1_solver1_d_s_particle_advect_amp_amp, geo1_solver1_d_s_particle_advect_freq_freq, geo1_solver1_d_s_particle_advect_geometryvopglobal1_Pbuffer->getDevicePointer(), geo1_solver1_d_s_particle_advect_geometryvopglobal1_vbuffer->getDevicePointer(), geo1_solver1_d_s_particle_advect_geometryvopglobal1_TimeInc, __geo1_solver1_d_s_particle_advect_add2_sum_debug_buffer->getDevicePointer(), __geo1_solver1_d_s_particle_advect_multiply2_product_debug_buffer->getDevicePointer(), __geo1_solver1_d_s_particle_advect_add1_sum_debug_buffer->getDevicePointer(), __geo1_solver1_d_s_particle_advect_multiply1_product_debug_buffer->getDevicePointer(), __geo1_solver1_d_s_particle_advect_fit1_shift_debug_buffer->getDevicePointer(), __geo1_solver1_d_s_particle_advect_multiply3_product_debug_buffer->getDevicePointer(), geo1_solver1_d_s_particle_advect_geometryvopoutput1_Pbuffer->getDevicePointer(),  numOfThreads);

    checkCUDAErrorWithLine("simpleparticle advect error");

    cudaDeviceSynchronize();
}