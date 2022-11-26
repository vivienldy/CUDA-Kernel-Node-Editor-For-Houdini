#include <glm/glm.hpp>

namespace CodeGenerator 
{ 
	namespace GenericCode 
 	{ 
        __host__ __device__ inline void @FUNC_NAME@(@PARM_LIST@)
        {
            // Data Load 
            @DATA_LOAD@

            // Compute graph
            @COMPUTE_GRAPH@

            // Write bacl
            @WRITE_BACK@
        }
	} 
} 