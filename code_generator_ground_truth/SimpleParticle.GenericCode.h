// 实现generic code
// CodeGenerator::GenericCode::particleAdvect()

#include <glm/glm.hpp> // 在这里includ glm 合适么？

namespace CodeGenerator 
{ 
    namespace GenericCode
    {
        void particleAdvect (
            glm::vec3* posBuffer,
            glm::vec3* velBuffer,
            float dt,
            int idx) 
        {
            // input variable analyse
            glm::vec3 global_input_p = posBuffer[idx];
            glm::vec3 global_input_v = velBuffer[idx];
            float global_input_t = dt;

            // compute graph
            // generate by multiply1
            glm::vec3 product = global_input_v * dt;
            // generate by add1
            glm::vec3 add = global_input_p + product;

            // write back
            glm::vec3 global_output_p = add;
            posBuffer[idx] = global_output_p;
        }
    }
} 