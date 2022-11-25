#include "SimpleParticle.h"

#include "CGField.h"

void CodeGenerator::particle_advect(CGBuffer<glm::vec3>* pbuffer, CGBuffer<glm::vec3>* vbuffer, float dt)
{
    int numThreads = pbuffer->getSize();;

    for(int index = 0; index < numThreads; ++index){
        CodeGenerator::GenericCode::particle_advect(pbuffer->getRawData(), vbuffer->getRawData(), dt, index);
    }
}
