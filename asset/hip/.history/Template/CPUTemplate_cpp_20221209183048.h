#include "@PROJ_NAME@.h"
#include "CGField.h"

void CodeGenerator::@FUNC_NAME@(@FUNC_DECLARE_LIST@)
{
    int numThreads = @GET_NUM_THREAD@;

    for(int index = 0; index < numThreads; ++index){
        CodeGenerator::GenericCode::@FUNC_NAME@(@SHARE_CODE_PARAM@, index);
    }
}
