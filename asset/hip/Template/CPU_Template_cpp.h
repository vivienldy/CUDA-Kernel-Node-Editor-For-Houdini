#include "#FILE_NAME#.h"

#include "CGField.h"

void CodeGenerator::#FUNC_NAME#(#FUNC_DECLARE_LIST#)
{
    #GET_RAWDATA#

    int numThreads = #GET_DATA_SIZE#;

    for(int index = 0; index < numThreads; ++index){
        CodeGenerator::GenericCode::#FUNC_NAME#(#RAWDATA_LIST#, index);
    }
}
