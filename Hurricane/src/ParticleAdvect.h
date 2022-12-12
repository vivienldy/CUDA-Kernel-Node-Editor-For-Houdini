#include "CGBuffer.h"
#include "ParticleAdvect.GenericCode.h"

namespace CodeGenerator
{
    void ParticleAdvect(float geo1_ParticleAdvect_parameter_time, float geo1_ParticleAdvect_input3_input3, float geo1_ParticleAdvect_input2_input2, CGBuffer<glm::vec3>* geo1_ParticleAdvect_geometryvopglobal1_Pbuffer, CGBuffer<float>* geo1_ParticleAdvect_geometryvopglobal1_agebuffer, CGBuffer<glm::vec3>* geo1_ParticleAdvect_geometryvopglobal1_Cdbuffer, float geo1_ParticleAdvect_geometryvopglobal1_TimeInc, CGGeometry* geo1_ParticleAdvect_geometryvopglobal1_OpInput1, CGGeometry* geo1_ParticleAdvect_geometryvopglobal1_OpInput2, CGBuffer<float>* __geo1_ParticleAdvect_add1_sum_debug_buffer);
    namespace CUDA 
    {
        void ParticleAdvect(
            float geo1_ParticleAdvect_parameter_time, 
            float geo1_ParticleAdvect_input3_input3, 
            float geo1_ParticleAdvect_input2_input2, 
            CGBuffer<glm::vec3>* geo1_ParticleAdvect_geometryvopglobal1_Pbuffer, 
            CGBuffer<float>* geo1_ParticleAdvect_geometryvopglobal1_agebuffer,
            CGBuffer<glm::vec3>* geo1_ParticleAdvect_geometryvopglobal1_Cdbuffer, 
            float geo1_ParticleAdvect_geometryvopglobal1_TimeInc, 
            CGGeometry* geo1_ParticleAdvect_geometryvopglobal1_OpInput1, 
            CGGeometry* geo1_ParticleAdvect_geometryvopglobal1_OpInput2, 
            CGBuffer<float>* __geo1_ParticleAdvect_add1_sum_debug_buffer, 
            int blockSize = 512);
    }
}
