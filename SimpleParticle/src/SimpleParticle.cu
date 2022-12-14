// 实现 kernel __global__ void
// CodeGenerator::CUDAKernel::particleAdvect()
// 实现 kernel launch <<< >>>
// CodeGenerator::CUDA::ParticleAdvect()

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

__global__ void CodeGenerator::CUDAKernel::particleAdvect (
    int geo1_solver1_d_s_pointvop2__DEBUG_turb_turb,
    float geo1_solver1_d_s_pointvop2__DEBUG_amp_amp,
    glm::vec3 geo1_solver1_d_s_pointvop2__DEBUG_freq_freq,
    glm::vec3* geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer,
    glm::vec3* geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_vbuffer,
    float TimeInc,
    glm::vec3* __geo1_solver1_d_s_pointvop2__DEBUG_add2_sum_debug_buffer,
    glm::vec3* __geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug_buffer,
    glm::vec3* __geo1_solver1_d_s_pointvop2__DEBUG_add1_sum_debug_buffer,
    glm::vec3* __geo1_solver1_d_s_pointvop2__DEBUG_multiply1_product_debug_buffer,
    glm::vec3* __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer,
    int numThreads)
{
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if(index > numThreads)
    return;
  CodeGenerator::GenericCode::particle_advect(
      geo1_solver1_d_s_pointvop2__DEBUG_turb_turb,
      geo1_solver1_d_s_pointvop2__DEBUG_amp_amp,
      geo1_solver1_d_s_pointvop2__DEBUG_freq_freq,
      geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer,
      geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_vbuffer,
      TimeInc,
      __geo1_solver1_d_s_pointvop2__DEBUG_add2_sum_debug_buffer,
      __geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug_buffer,
      __geo1_solver1_d_s_pointvop2__DEBUG_add1_sum_debug_buffer,
      __geo1_solver1_d_s_pointvop2__DEBUG_multiply1_product_debug_buffer,
      __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer,
      index);
}

void CodeGenerator::CUDA::ParticleAdvect (
    int geo1_solver1_d_s_pointvop2__DEBUG_turb_turb,
    float geo1_solver1_d_s_pointvop2__DEBUG_amp_amp,
    glm::vec3 geo1_solver1_d_s_pointvop2__DEBUG_freq_freq,
    CGBuffer<glm::vec3>* geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer,
    CGBuffer<glm::vec3>* geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_vbuffer,
    float TimeInc,
    CGBuffer<glm::vec3>* __geo1_solver1_d_s_pointvop2__DEBUG_add2_sum_debug_buffer,
    CGBuffer<glm::vec3>* __geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug_buffer,
    CGBuffer<glm::vec3>* __geo1_solver1_d_s_pointvop2__DEBUG_add1_sum_debug_buffer,
    CGBuffer<glm::vec3>* __geo1_solver1_d_s_pointvop2__DEBUG_multiply1_product_debug_buffer,
    CGBuffer<glm::vec3>* __geo1_solver1_d_s_pointvop2__DEBUG_curlnoise1_noise_debug_buffer,
    CGBuffer<glm::vec3>* __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer,
    int blockSize)
{
    /*geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer->malloc();
    geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer->loadHostToDevice();

    geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_vbuffer->malloc();
    geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_vbuffer->loadHostToDevice();

    __geo1_solver1_d_s_pointvop2__DEBUG_add2_sum_debug_buffer->malloc();
    __geo1_solver1_d_s_pointvop2__DEBUG_add2_sum_debug_buffer->loadHostToDevice();

    __geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug_buffer->malloc();
    __geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug_buffer->loadHostToDevice();

    __geo1_solver1_d_s_pointvop2__DEBUG_add1_sum_debug_buffer->malloc();
    __geo1_solver1_d_s_pointvop2__DEBUG_add1_sum_debug_buffer->loadHostToDevice();

    __geo1_solver1_d_s_pointvop2__DEBUG_multiply1_product_debug_buffer->malloc();
    __geo1_solver1_d_s_pointvop2__DEBUG_multiply1_product_debug_buffer->loadHostToDevice();

    __geo1_solver1_d_s_pointvop2__DEBUG_curlnoise1_noise_debug_buffer->malloc();
    __geo1_solver1_d_s_pointvop2__DEBUG_curlnoise1_noise_debug_buffer->loadHostToDevice();

    __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer->malloc();
    __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer->loadHostToDevice();

    __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer->malloc();
    __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer->loadHostToDevice();*/

    int numOfThreads = geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer->getSize();
    auto num_blocks_threads = ThreadBlockInfo(blockSize, numOfThreads);
  
      CodeGenerator::CUDAKernel::particleAdvect<<<num_blocks_threads.x, num_blocks_threads.y>>>(
          geo1_solver1_d_s_pointvop2__DEBUG_turb_turb,
          geo1_solver1_d_s_pointvop2__DEBUG_amp_amp,
          geo1_solver1_d_s_pointvop2__DEBUG_freq_freq,
          geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer->getDevicePointer(),
          geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_vbuffer->getDevicePointer(),
          TimeInc,
          __geo1_solver1_d_s_pointvop2__DEBUG_add2_sum_debug_buffer->getDevicePointer(),
          __geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug_buffer->getDevicePointer(),
          __geo1_solver1_d_s_pointvop2__DEBUG_add1_sum_debug_buffer->getDevicePointer(),
          __geo1_solver1_d_s_pointvop2__DEBUG_multiply1_product_debug_buffer->getDevicePointer(),
          __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer->getDevicePointer(),
          numOfThreads);

      checkCUDAErrorWithLine("simpleparticle advect");

      cudaDeviceSynchronize();
     
    //// because now input buffer and output buffer are different
    //// need to pingpong buffer on device
    //cudaMemcpy(
    //    geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer->getDevicePointer(),
    //    geo1_solver1_d_s_pointvop2__DEBUG_geometryvopoutput1_Pbuffer->getDevicePointer(),
    //    sizeof(glm::vec3) * numOfThreads, cudaMemcpyDeviceToDevice);
}
