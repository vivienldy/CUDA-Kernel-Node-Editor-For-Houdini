#include <cuda.h> 
#include <glm/glm.hpp>

#include "CGBuffer.h"
#include "#PROJ_NAME#.GenericCode.h"


namespace CodeGenerator
{
    void #FUNC_NAME#(#FUNC_DECLARE_LIST#);
    namespace CUDA 
    {
        void #FUNC_NAME#(#FUNC_DECLARE_LIST#, int blockSize=512);
    }
}
