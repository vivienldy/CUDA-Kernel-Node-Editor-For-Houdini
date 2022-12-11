#include "simpleParticle.h"
#include "CGField.h"

void CodeGenerator::simpleParticle(glm::vec3 geo1_simpleParticle_parm2_force, float geo1_simpleParticle_parm1_time, CGBuffer<glm::vec3>* geo1_simpleParticle_geometryvopglobal1_Pbuffer, CGBuffer<glm::vec3>* geo1_simpleParticle_geometryvopglobal1_vbuffer, float geo1_simpleParticle_geometryvopglobal1_TimeInc, CGGeometry* geo1_simpleParticle_geometryvopglobal1_OpInput1)
{
    int numThreads = geo1_simpleParticle_geometryvopglobal1_Pbuffer->getSize();

    for(int index = 0; index < numThreads; ++index){
        CodeGenerator::GenericCode::simpleParticle(geo1_simpleParticle_parm2_force, geo1_simpleParticle_parm1_time, geo1_simpleParticle_geometryvopglobal1_Pbuffer->getRawData(), geo1_simpleParticle_geometryvopglobal1_vbuffer->getRawData(), geo1_simpleParticle_geometryvopglobal1_TimeInc, geo1_simpleParticle_geometryvopglobal1_OpInput1->GetGeometryRawData(), index);
    }
}
