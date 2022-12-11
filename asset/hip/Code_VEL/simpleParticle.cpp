#include "simpleParticle.h"
#include "CGField.h"

void CodeGenerator::simpleParticle(CGBuffer<glm::vec3>* geo1_simpleParticle_geometryvopglobal1_Pbuffer, CGBuffer<glm::vec3>* geo1_simpleParticle_geometryvopglobal1_vbuffer, float geo1_simpleParticle_geometryvopglobal1_TimeInc)
{
    int numThreads = geo1_simpleParticle_geometryvopglobal1_Pbuffer->getSize();

    for(int index = 0; index < numThreads; ++index){
        CodeGenerator::GenericCode::simpleParticle(geo1_simpleParticle_geometryvopglobal1_Pbuffer->getRawData(), geo1_simpleParticle_geometryvopglobal1_vbuffer->getRawData(), geo1_simpleParticle_geometryvopglobal1_TimeInc, index);
    }
}
