// 声明 CodeGenerator::ParticleAdvect() // CPU FORLOOP
// 声明 CodeGenerator::CUDA::ParticleAdvect() // GPU Kernel Launch
// 实现main函数

#include "../../include/CGBuffer.h"
#include "SimpleParticle.GenericCode.h"

#define CPU_VERSION 0
#define GPU_VERSION 1

namespace CodeGenerator
{
  void ParticleAdvect (
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
      CGBuffer<glm::vec3>* __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer);

  namespace CUDA 
  {
    void ParticleAdvect (
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
        int blockSize = 512);
  }
}



