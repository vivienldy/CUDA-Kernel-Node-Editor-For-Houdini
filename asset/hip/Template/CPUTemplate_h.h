#include "CGBuffer.h"
#include "@PROJ_NAME@.GenericCode.h"

#define CPU_VERSION 0
#define GPU_VERSION 1

namespace CodeGenerator
{
    void @FUNC_NAME@(@FUNC_DECLARE_LIST@);
    namespace CUDA 
    {
        void @FUNC_NAME@(@FUNC_DECLARE_LIST@, int blockSize = 512);
    }
}
