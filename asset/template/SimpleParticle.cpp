// 实现 CodeGenerator::ParticleAdvect() // CPU FORLOOP

#include "ShowParticle.h"

void CodeGenerator::ParticleAdvect (
CGBuffer* posBuffer,
CGBuffer* velBuffer,
float dt)
{
     auto posBufferRaw =  posBuffer->GetDataRaw();
     auto velBufferRaw =  velBuffer->GetDataRaw();
     int numThreads = posBuffer->Size();

     for(int i = 0; i < numThreads; ++i){
          CodeGenerator::GenericCode::particleAdvect(
               posBufferRaw,
               velBufferRaw,
               dt,
               i);
     }
}
