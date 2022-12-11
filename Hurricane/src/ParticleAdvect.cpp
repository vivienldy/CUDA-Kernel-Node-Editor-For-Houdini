#include "ParticleAdvect.h"
#include "CGField.h"

void CodeGenerator::ParticleAdvect(glm::vec3 geo1_ParticleAdvect_offset_offset, float geo1_ParticleAdvect_input3_input3, float geo1_ParticleAdvect_input2_input2, CGBuffer<glm::vec3>* geo1_ParticleAdvect_geometryvopglobal1_Pbuffer, CGBuffer<float>* geo1_ParticleAdvect_geometryvopglobal1_agebuffer, CGBuffer<glm::vec3>* geo1_ParticleAdvect_geometryvopglobal1_Cdbuffer, float geo1_ParticleAdvect_geometryvopglobal1_TimeInc, CGGeometry* geo1_ParticleAdvect_geometryvopglobal1_OpInput1, CGGeometry* geo1_ParticleAdvect_geometryvopglobal1_OpInput2, CGBuffer<float>* __geo1_ParticleAdvect_add1_sum_debug_buffer)
{
    int numThreads = geo1_ParticleAdvect_geometryvopglobal1_Pbuffer->getSize();

   

    for(int index = 0; index < numThreads; ++index){

        //if (geo1_ParticleAdvect_geometryvopglobal1_OpInput2->m_VelField->m_FieldX->GetVoxelBufferPtr()->getSize() == 1026206379)
        //{
        //    std::cout << "---" << std::endl;
        //}       
        CodeGenerator::GenericCode::ParticleAdvect(geo1_ParticleAdvect_offset_offset, 
            geo1_ParticleAdvect_input3_input3, 
            geo1_ParticleAdvect_input2_input2, 
            geo1_ParticleAdvect_geometryvopglobal1_Pbuffer->getRawData(), 
            geo1_ParticleAdvect_geometryvopglobal1_agebuffer->getRawData(), 
            geo1_ParticleAdvect_geometryvopglobal1_Cdbuffer->getRawData(),
            geo1_ParticleAdvect_geometryvopglobal1_TimeInc,
            geo1_ParticleAdvect_geometryvopglobal1_OpInput1->GetGeometryRawData(), 
            geo1_ParticleAdvect_geometryvopglobal1_OpInput2->GetGeometryRawData(),
            __geo1_ParticleAdvect_add1_sum_debug_buffer->getRawData(), 
            index);
        //auto opinput2 = geo1_ParticleAdvect_geometryvopglobal1_OpInput2->GetGeometryRawData();
        //std::cout << *(opinput2.velFieldRAWData.VoxelDataX) << std::endl;
        //if (index == 782) {
        //    std::cout << geo1_ParticleAdvect_geometryvopglobal1_OpInput2->m_VelField->m_FieldX->GetVoxelBufferPtr()->getSize() << std::endl;
        //}
    }
}
