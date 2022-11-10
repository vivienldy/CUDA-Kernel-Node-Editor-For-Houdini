// 声明 kernel __global__ void 

#include <cuda.h> // ？ 在这里include cuda 合适么

namespace CodeGenerator
{
  namespace CUDAKernel
  {
    __global__ void particleAdvect(
      glm::vec3* posBuffer,
      glm::vec3* velBuffer,
      float dt,
      int numThreads);
  }
}
