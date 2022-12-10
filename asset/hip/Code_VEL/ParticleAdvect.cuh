<<<<<<< HEAD
#include <cuda.h> 
=======
#include <cuda.h>
>>>>>>> c6697ae8c39c9fb59de35b574ac2e713d9e084dd
#include <glm/glm.hpp>
#include <device_launch_parameters.h>

namespace CodeGenerator
{
  namespace CUDAKernel
  {
<<<<<<< HEAD
      __global__ void ParticleAdvect(glm::vec3 geo1_ParticleAdvect_offset_offset, float geo1_ParticleAdvect_input3_input3, float geo1_ParticleAdvect_input2_input2, glm::vec3* geo1_ParticleAdvect_geometryvopglobal1_Pbuffer, float geo1_ParticleAdvect_geometryvopglobal1_age, glm::vec3* geo1_ParticleAdvect_geometryvopglobal1_Cdbuffer, float geo1_ParticleAdvect_geometryvopglobal1_TimeInc, CGGeometry::RAWData geo1_ParticleAdvect_geometryvopglobal1_OpInput1, CGGeometry::RAWData geo1_ParticleAdvect_geometryvopglobal1_OpInput2, float* __geo1_ParticleAdvect_add1_sum_debug_buffer,  int numThreads);
=======
    __global__ void ParticleAdvect(
      glm::vec3 geo1_ParticleAdvect_offset_offset, 
      float geo1_ParticleAdvect_input3_input3, 
      float geo1_ParticleAdvect_input2_input2, 
      glm::vec3 *geo1_ParticleAdvect_geometryvopglobal1_Pbuffer, 
      float geo1_ParticleAdvect_geometryvopglobal1_age, 
      glm::vec3 *geo1_ParticleAdvect_geometryvopglobal1_Cdbuffer, 
      float geo1_ParticleAdvect_geometryvopglobal1_TimeInc, 
      CGGeometry::RAWData geo1_ParticleAdvect_geometryvopglobal1_OpInput1, 
      CGGeometry::RAWData geo1_ParticleAdvect_geometryvopglobal1_OpInput2, 
      float *__geo1_ParticleAdvect_add1_sum_debug_buffer, 
      int numThreads);
>>>>>>> c6697ae8c39c9fb59de35b574ac2e713d9e084dd
  }
}