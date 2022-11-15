// 实现generic code
// CodeGenerator::GenericCode::particleAdvect()

#include <glm/glm.hpp>

namespace CodeGenerator 
{ 
    namespace GenericCode
    {
        glm::vec3 curlnoise(glm::vec3 pos,  glm::vec3 freq, float amp, int turb) {
            return glm::vec3(1.f);
        }

        void particleAdvect(
            int geo1_solver1_d_s_pointvop2__DEBUG_turb_turb, 
            float geo1_solver1_d_s_pointvop2__DEBUG_amp_amp, 
            glm::vec3 geo1_solver1_d_s_pointvop2__DEBUG_freq_freq, 
            glm::vec3* geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer, 
            glm::vec3* geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_vbuffer,
            float geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_TimeInc, 
            glm::vec3* __geo1_solver1_d_s_pointvop2__DEBUG_add2_sum_debug_buffer,
            glm::vec3* __geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug_buffer, 
            glm::vec3* __geo1_solver1_d_s_pointvop2__DEBUG_add1_sum_debug_buffer,
            glm::vec3* __geo1_solver1_d_s_pointvop2__DEBUG_multiply1_product_debug_buffer,
            glm::vec3* __geo1_solver1_d_s_pointvop2__DEBUG_curlnoise1_noise_debug_buffer,
            glm::vec3* __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer, 
            glm::vec3* geo1_solver1_d_s_pointvop2__DEBUG_geometryvopoutput1_Pbuffer, 
            int idx) {

            // Data Load 
            // Geometry Global Input
            glm::vec3 geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_P = geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer[idx];
            glm::vec3 geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_v = geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_vbuffer[idx];

            // Compute graph
            float geo1_solver1_d_s_pointvop2__DEBUG_Node_Value = 0.9;
            glm::vec3 geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product = geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_P * geo1_solver1_d_s_pointvop2__DEBUG_Node_Value;
            glm::vec3 geo1_solver1_d_s_pointvop2__DEBUG_curlnoise1_noise = curlnoise(
                geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product, 
                geo1_solver1_d_s_pointvop2__DEBUG_freq_freq, 
                geo1_solver1_d_s_pointvop2__DEBUG_amp_amp, 
                geo1_solver1_d_s_pointvop2__DEBUG_turb_turb);
            glm::vec3 geo1_solver1_d_s_pointvop2__DEBUG_multiply1_product = geo1_solver1_d_s_pointvop2__DEBUG_curlnoise1_noise * geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_TimeInc;
            glm::vec3 geo1_solver1_d_s_pointvop2__DEBUG_add1_sum = geo1_solver1_d_s_pointvop2__DEBUG_multiply1_product + geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_v;
            glm::vec3 geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product = geo1_solver1_d_s_pointvop2__DEBUG_add1_sum * geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_TimeInc;
            glm::vec3 geo1_solver1_d_s_pointvop2__DEBUG_add2_sum = geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product + geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_P;

            // Write back 
            glm::vec3 __geo1_solver1_d_s_pointvop2__DEBUG_add2_sum_debug = geo1_solver1_d_s_pointvop2__DEBUG_add2_sum;
            __geo1_solver1_d_s_pointvop2__DEBUG_add2_sum_debug_buffer[idx] = __geo1_solver1_d_s_pointvop2__DEBUG_add2_sum_debug;

            glm::vec3 __geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug = geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product;
            __geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug_buffer[idx] = __geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug;

            glm::vec3 __geo1_solver1_d_s_pointvop2__DEBUG_add1_sum_debug = geo1_solver1_d_s_pointvop2__DEBUG_add1_sum;
            __geo1_solver1_d_s_pointvop2__DEBUG_add1_sum_debug_buffer[idx] = __geo1_solver1_d_s_pointvop2__DEBUG_add1_sum_debug;

            glm::vec3 __geo1_solver1_d_s_pointvop2__DEBUG_multiply1_product_debug = geo1_solver1_d_s_pointvop2__DEBUG_multiply1_product;
            __geo1_solver1_d_s_pointvop2__DEBUG_multiply1_product_debug_buffer[idx] = __geo1_solver1_d_s_pointvop2__DEBUG_multiply1_product_debug;

            glm::vec3 __geo1_solver1_d_s_pointvop2__DEBUG_curlnoise1_noise_debug = geo1_solver1_d_s_pointvop2__DEBUG_curlnoise1_noise;
            __geo1_solver1_d_s_pointvop2__DEBUG_curlnoise1_noise_debug_buffer[idx] = __geo1_solver1_d_s_pointvop2__DEBUG_curlnoise1_noise_debug;

            glm::vec3 __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug = geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product;
            __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer[idx] = __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug;

            glm::vec3 global_output_geo1_solver1_d_s_pointvop2__DEBUG_geometryvopoutput1_P = geo1_solver1_d_s_pointvop2__DEBUG_add2_sum;
            geo1_solver1_d_s_pointvop2__DEBUG_geometryvopoutput1_Pbuffer[idx] = global_output_geo1_solver1_d_s_pointvop2__DEBUG_geometryvopoutput1_P;
        }

    }
} 