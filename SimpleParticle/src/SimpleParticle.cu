// 实现 kernel __global__ void
// CodeGenerator::CUDAKernel::particleAdvect()
// 实现 kernel launch <<< >>>
// CodeGenerator::CUDA::ParticleAdvect()

#include "SimpleParticle.h"
#include "SimpleParticle.cuh"


glm::vec2 ThreadBlockInfo(int blockSize, int numThreads)
{
    return glm::vec2(int(numThreads / blockSize) + 1, blockSize > numThreads ? numThreads : blockSize);
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
    glm::vec3* __geo1_solver1_d_s_pointvop2__DEBUG_curlnoise1_noise_debug_buffer,
    glm::vec3* __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer,
    glm::vec3* geo1_solver1_d_s_pointvop2__DEBUG_geometryvopoutput1_Pbuffer,
    int numThreads)
{
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if(index > numThreads)
    return;
  CodeGenerator::GenericCode::particleAdvect(
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
      __geo1_solver1_d_s_pointvop2__DEBUG_curlnoise1_noise_debug_buffer,
      __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer,
      geo1_solver1_d_s_pointvop2__DEBUG_geometryvopoutput1_Pbuffer,
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
    CGBuffer<glm::vec3>* geo1_solver1_d_s_pointvop2__DEBUG_geometryvopoutput1_Pbuffer,
    int blockSize = 512)
{
  int numOfThreads = geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer->size();
  auto num_blocks_threads = ThreadBlockInfo(blockSize, numOfThreads);
  
  CodeGenerator::CUDAKernel::particleAdvect<<<num_blocks_threads.x, num_blocks_threads.y>>>(
    posBuffer->GetDevicePtr(),
    velBuffer->GetDevicePtr(),
    dt,
      num_blocks_threads);
}
