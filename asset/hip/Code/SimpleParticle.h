#include <cuda.h> 
#include <glm/glm.hpp>

#include "CGBuffer.h"
#include "SimpleParticle.GenericCode.h"


namespace CodeGenerator
{
    void particle_advect(CGBuffer<glm::vec3>* pbuffer, CGBuffer<glm::vec3>* vbuffer, float dt);
    namespace CUDA 
    {
        void particle_advect(CGBuffer<glm::vec3>* pbuffer, CGBuffer<glm::vec3>* vbuffer, float dt, int blockSize=512);
    }
}
