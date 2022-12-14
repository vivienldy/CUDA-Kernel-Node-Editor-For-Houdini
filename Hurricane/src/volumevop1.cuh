#include <cuda.h>
#include <glm/glm.hpp>
#include <device_launch_parameters.h>

namespace CodeGenerator
{
  namespace CUDAKernel
  {
    __global__ void volumevop1(
      float geo1_volumevop1_input8_input5, 
      float geo1_volumevop1_input7_input4, 
      float geo1_volumevop1_input5_input3, 
      float geo1_volumevop1_input3_input2, 
      CGGeometry::RAWData geo1_volumevop1_volumevopglobal1_OpInput1, 
      CGGeometry::RAWData geo1_volumevop1_volumevopglobal1_OpInput2,
      int numThreads);
  }
}