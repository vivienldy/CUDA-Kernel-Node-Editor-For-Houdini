#include "CGBuffer.h"
#include "volumevop1.GenericCode.h"

namespace CodeGenerator
{
    void volumevop1(
        float geo1_volumevop1_input8_input5, 
        float geo1_volumevop1_input7_input4, 
        float geo1_volumevop1_input5_input3, 
        float geo1_volumevop1_input3_input2, 
        CGGeometry *geo1_volumevop1_volumevopglobal1_OpInput1, 
        CGGeometry *geo1_volumevop1_volumevopglobal1_OpInput2);
    
    namespace CUDA
    {
        void volumevop1(
            float geo1_volumevop1_input8_input5, 
            float geo1_volumevop1_input7_input4, 
            float geo1_volumevop1_input5_input3, 
            float geo1_volumevop1_input3_input2, 
            CGGeometry *geo1_volumevop1_volumevopglobal1_OpInput1, 
            CGGeometry *geo1_volumevop1_volumevopglobal1_OpInput2, 
            int blockSize = 512);
    }
}
