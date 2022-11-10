// 实现 kernel __global__ void
// CodeGenerator::CUDAKernel::particleAdvect()
// 实现 kernel launch <<< >>>
// CodeGenerator::CUDA::ParticleAdvect()

#include "ShowParticle.h"
#include "ShowParticle.cuh"

__global__ void CodeGenerator::CUDAKernel::particleAdvect (
  glm::vec3* posBuffer,
  glm::vec3* velBuffer,
  float dt,
  int numThreads)
{
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if(index > numThreads)
    return;
  CodeGenerator::GenericCode::particleAdvect(
    posBuffer,
    velBuffer,
    dt,
    index);
}

void CodeGenerator::CUDA::ParticleAdvect (
    CGBuffer* posBuffer,
    CGBuffer* velBuffer,
    float dt,
    int blockSize = 512)
{
  auto num_blocks_threads = posBuffer->GetNBlocksAndThreads(blockSize);

  CodeGenerator::CUDAKernel::particleAdvect<<<num_blocks_threads.x, num_blocks_threads.y>>>(
    posBuffer->GetDevicePtr(),
    velBuffer->GetDevicePtr(),
    dt,
    posBuffer->Size());
}
