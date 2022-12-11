#include <cuda.h> 
#include <glm/glm.hpp>
#include <device_launch_parameters.h>

namespace CodeGenerator
{
  namespace CUDAKernel
  {
      __global__ void particle_advect(int geo1_solver1_d_s_particle_advect_turb_turb, float geo1_solver1_d_s_particle_advect_amp_amp, glm::vec3 geo1_solver1_d_s_particle_advect_freq_freq, glm::vec3* geo1_solver1_d_s_particle_advect_geometryvopglobal1_Pbuffer, glm::vec3* geo1_solver1_d_s_particle_advect_geometryvopglobal1_vbuffer, float geo1_solver1_d_s_particle_advect_geometryvopglobal1_TimeInc, glm::vec3* __geo1_solver1_d_s_particle_advect_add2_sum_debug_buffer, glm::vec3* __geo1_solver1_d_s_particle_advect_multiply2_product_debug_buffer, glm::vec3* __geo1_solver1_d_s_particle_advect_add1_sum_debug_buffer, glm::vec3* __geo1_solver1_d_s_particle_advect_multiply1_product_debug_buffer, glm::vec3* __geo1_solver1_d_s_particle_advect_multiply3_product_debug_buffer,  int numThreads);
  }
}