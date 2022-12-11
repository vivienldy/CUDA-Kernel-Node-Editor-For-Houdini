#include <cuda.h> 
#include <glm/glm.hpp>
#include <device_launch_parameters.h>

namespace CodeGenerator
{
  namespace CUDAKernel
  {
      __global__ void simpleParticle(glm::vec3 geo1_simpleParticle_parm2_force, float geo1_simpleParticle_parm1_time, glm::vec3* geo1_simpleParticle_geometryvopglobal1_Pbuffer, glm::vec3* geo1_simpleParticle_geometryvopglobal1_vbuffer, float geo1_simpleParticle_geometryvopglobal1_TimeInc, CGGeometry::RAWData geo1_simpleParticle_geometryvopglobal1_OpInput1,  int numThreads);
  }
}