#include "CGBuffer.h"
#include "volumevop1.GenericCode.h"

#define CPU_VERSION 0
#define GPU_VERSION 1

namespace CodeGenerator
{
    void volumevop1(float geo1_volumevop1_input8_input5, float geo1_volumevop1_input7_input4, float geo1_volumevop1_input5_input3, float geo1_volumevop1_input3_input2, struct geo1_volumevop1_volumevopglobal1_OpInput1, struct geo1_volumevop1_volumevopglobal1_OpInput2);
    namespace CUDA
    {
        void volumevop1(float geo1_volumevop1_input8_input5, float geo1_volumevop1_input7_input4, float geo1_volumevop1_input5_input3, float geo1_volumevop1_input3_input2, struct geo1_volumevop1_volumevopglobal1_OpInput1, struct geo1_volumevop1_volumevopglobal1_OpInput2, int blockSize = 512);
    }
}
