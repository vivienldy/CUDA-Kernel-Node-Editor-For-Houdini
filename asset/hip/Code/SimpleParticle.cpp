#include "SimpleParticle.h"

#include "CGField.h"

void CodeGenerator::particle_advect(CGBuffer<glm::vec3>* pbuffer, CGBuffer<glm::vec3>* vbuffer, float dt, CGBuffer<glm::vec3>* add2_sum_debugbuffer, CGBuffer<glm::vec3>* multiply2_product_debugbuffer, CGBuffer<glm::vec3>* add1_sum_debugbuffer, int turb_turb, float amp_amp)
{
    int numThreads = pbuffer->getSize();;

    for(int index = 0; index < numThreads; ++index){
        CodeGenerator::GenericCode::particle_advect(pbuffer->getRawData(), vbuffer->getRawData(), dt, add2_sum_debugbuffer->getRawData(), multiply2_product_debugbuffer->getRawData(), add1_sum_debugbuffer->getRawData(), turb_turb, amp_amp, index);
    }
}
