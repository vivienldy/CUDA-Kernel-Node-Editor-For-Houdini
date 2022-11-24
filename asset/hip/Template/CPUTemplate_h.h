#include <cuda.h> 
#include <glm/glm.hpp>

#include "CGBuffer.h"
#include "#OUT_FILE_NAME#.GenericCode.h"


namespace CodeGenerator
{
    void #FUNC_NAME#(#FUNC_DECLARE_LIST#);
    namespace CUDA 
    {
        void #FUNC_NAME#(#FUNC_DELCARE_LIST#, int blockSize=512);
    }
}
