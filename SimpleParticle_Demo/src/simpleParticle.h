#include "CGBuffer.h"
#include "simpleParticle.GenericCode.h"

namespace CodeGenerator
{
    void simpleParticle(glm::vec3 geo1_simpleParticle_parm2_force, float geo1_simpleParticle_parm1_time, CGBuffer<glm::vec3>* geo1_simpleParticle_geometryvopglobal1_Pbuffer, CGBuffer<glm::vec3>* geo1_simpleParticle_geometryvopglobal1_vbuffer, float geo1_simpleParticle_geometryvopglobal1_TimeInc, CGGeometry* geo1_simpleParticle_geometryvopglobal1_OpInput1);
    namespace CUDA 
    {
        void simpleParticle(glm::vec3 geo1_simpleParticle_parm2_force, float geo1_simpleParticle_parm1_time, CGBuffer<glm::vec3>* geo1_simpleParticle_geometryvopglobal1_Pbuffer, CGBuffer<glm::vec3>* geo1_simpleParticle_geometryvopglobal1_vbuffer, float geo1_simpleParticle_geometryvopglobal1_TimeInc, CGGeometry* geo1_simpleParticle_geometryvopglobal1_OpInput1, int blockSize = 512);
    }
}
