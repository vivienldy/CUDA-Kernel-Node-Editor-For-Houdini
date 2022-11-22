// 声明 kernel __global__ void 

#include <cuda.h> 
#include <glm/glm.hpp>
#include <device_launch_parameters.h>

namespace CodeGenerator
{
  namespace CUDAKernel
  {
      __global__ void #KERNEL_NAME#(#KERNEL_PARM_DECLARE_LIST#, int numThreads);
  }
}