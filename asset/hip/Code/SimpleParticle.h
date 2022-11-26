#include <cuda.h> 
#include <glm/glm.hpp>

#include "CGBuffer.h"
#include "SimpleParticle.GenericCode.h"


namespace CodeGenerator
{
    void particle_advect(CGBuffer<glm::vec3>* pbuffer, CGBuffer<glm::vec3>* vbuffer, float dt, CGBuffer<glm::vec3>* add2_sum_debugbuffer, CGBuffer<glm::vec3>* multiply2_product_debugbuffer, CGBuffer<glm::vec3>* add1_sum_debugbuffer, int turb_turb, float amp_amp);
    namespace CUDA 
    {
        void particle_advect(CGBuffer<glm::vec3>* pbuffer, CGBuffer<glm::vec3>* vbuffer, float dt, CGBuffer<glm::vec3>* add2_sum_debugbuffer, CGBuffer<glm::vec3>* multiply2_product_debugbuffer, CGBuffer<glm::vec3>* add1_sum_debugbuffer, int turb_turb, float amp_amp, int blockSize=512);
    }
}
