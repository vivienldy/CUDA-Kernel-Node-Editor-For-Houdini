// 声明 CodeGenerator::ParticleAdvect() // CPU FORLOOP
// 声明 CodeGenerator::CUDA::ParticleAdvect() // GPU Kernel Launch
// 实现main函数

#include "CGBuffer.h"
#include "ShowParticle.GenericCode.h"

#define CPU_VERSION 1
#define GPU_VERSION 0

namespace CodeGenerator
{
  void ParticleAdvect (
    CGBuffer* posBuffer,
    CGBuffer* velBuffer,
    float dt);

  namespace CUDA 
  {
    void ParticleAdvect (
      CGBuffer* posBuffer,
      CGBuffer* velBuffer,
      float dt,
      int blockSize = 512);
  }
}


int main() // ? 还是main函数应该是现在单独的main.cpp 里
{
  // 所有CGBuffer相关功能需要育嘉再确定
  // load buffer from file
  auto posBuffer = CGBuffer::Load(/*file path*/); 
  
  // create and initialize vel buffer
  auto velBuffer = new CGBuffer("v",0); 

  int startFrame = 0;
  int endFrame = 10;
  float FPS = 24.f;
  int blockSize = 128;

  for(int i = startFrame; i < endFrame; ++i){
    //hard code var block
    float Time = i/FPS;
    float Frame = i;
    float TimeInc = 1.0/FPS;
    int NumPoints = posBuffer->Size(); // ? 目前在这里需要么

#if CPU_VERSION
  CodeGenerator::ParticleAdvect(
    posBuffer,
    velBuffer,
    TimeInc);

#elif GPU_VERSION
  CodeGenerator::CUDA::ParticleAdvect(
    posBuffer,
    velBuffer,
    TimeInc,
    blockSize);
#endif

  // [TODO] SAFE BUFFER AS OBJ
  }  
}



