#include "volumevop1.h"
#include "CGField.h"

void CodeGenerator::volumevop1(
    float geo1_volumevop1_input8_input5, 
    float geo1_volumevop1_input7_input4, 
    float geo1_volumevop1_input5_input3, 
    float geo1_volumevop1_input3_input2, 
    CGGeometry* geo1_volumevop1_volumevopglobal1_OpInput1, 
    CGGeometry* geo1_volumevop1_volumevopglobal1_OpInput2)
{
    int numThreads = geo1_volumevop1_volumevopglobal1_OpInput1->m_VelField->GetNumVoxels();

    for (int index = 0; index < numThreads; ++index)
    {
        CodeGenerator::GenericCode::volumevop1(
            geo1_volumevop1_input8_input5, 
            geo1_volumevop1_input7_input4, 
            geo1_volumevop1_input5_input3, 
            geo1_volumevop1_input3_input2, 
            geo1_volumevop1_volumevopglobal1_OpInput1->GetGeometryRawData(), 
            geo1_volumevop1_volumevopglobal1_OpInput2->GetGeometryRawData(), 
            index);
    }
}
