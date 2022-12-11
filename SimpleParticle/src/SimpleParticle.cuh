// 声明 kernel __global__ void 

#include <cuda.h> 
#include <glm/glm.hpp>
#include <device_launch_parameters.h>

namespace CodeGenerator
{
  namespace CUDAKernel
  {
      __global__ void particleAdvect(
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
          int numThreads);
  }
}
