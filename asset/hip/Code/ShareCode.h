#include <glm/glm.hpp>

namespace CodeGenerator 
{ 
	namespace GenericCode 
 	{ 
        __host__ __device__ inline void particle_advect(int geo1_solver1_d_s_particle_advect_turb_turb, float geo1_solver1_d_s_particle_advect_amp_amp, glm::vec3 geo1_solver1_d_s_particle_advect_freq_freq, glm::vec3* geo1_solver1_d_s_particle_advect_geometryvopglobal1_Pbuffer, glm::vec3* geo1_solver1_d_s_particle_advect_geometryvopglobal1_vbuffer, float geo1_solver1_d_s_particle_advect_geometryvopglobal1_TimeInc, glm::vec3* __geo1_solver1_d_s_particle_advect_add2_sum_debug_buffer, glm::vec3* __geo1_solver1_d_s_particle_advect_multiply2_product_debug_buffer, glm::vec3* __geo1_solver1_d_s_particle_advect_add1_sum_debug_buffer, glm::vec3* __geo1_solver1_d_s_particle_advect_multiply1_product_debug_buffer, glm::vec3* __geo1_solver1_d_s_particle_advect_fit1_shift_debug_buffer, glm::vec3* __geo1_solver1_d_s_particle_advect_multiply3_product_debug_buffer, glm::vec3* geo1_solver1_d_s_particle_advect_geometryvopoutput1_Pbuffer, int idx)
        {
            // Data Load 
            // Geometry Global Input
glm::vec3 geo1_solver1_d_s_particle_advect_geometryvopglobal1_P = geo1_solver1_d_s_particle_advect_geometryvopglobal1_Pbuffer[idx];
glm::vec3 geo1_solver1_d_s_particle_advect_geometryvopglobal1_v = geo1_solver1_d_s_particle_advect_geometryvopglobal1_vbuffer[idx];


            // Compute graph
            float geo1_solver1_d_s_particle_advect_Node_Value = 0.9;
float geo1_solver1_d_s_particle_advect_fit2_shift = fit(geo1_solver1_d_s_particle_advect_geometryvopglobal1_TimeInc, CG_NONE, CG_NONE, CG_NONE, CG_NONE);
glm::vec3 geo1_solver1_d_s_particle_advect_multiply3_product = geo1_solver1_d_s_particle_advect_geometryvopglobal1_P * geo1_solver1_d_s_particle_advect_Node_Value;
glm::vec3 geo1_solver1_d_s_particle_advect_curlnoise1_noise = curlnoise(CG_NONE, geo1_solver1_d_s_particle_advect_multiply3_product, geo1_solver1_d_s_particle_advect_freq_freq, CG_NONE, geo1_solver1_d_s_particle_advect_amp_amp, CG_NONE, CG_NONE, geo1_solver1_d_s_particle_advect_turb_turb, CG_NONE, CG_NONE, CG_NONE, CG_NONE, CG_NONE, CG_NONE);
glm::vec3 geo1_solver1_d_s_particle_advect_multiply1_product = geo1_solver1_d_s_particle_advect_curlnoise1_noise * geo1_solver1_d_s_particle_advect_fit2_shift;
glm::vec3 geo1_solver1_d_s_particle_advect_add1_sum = geo1_solver1_d_s_particle_advect_multiply1_product + geo1_solver1_d_s_particle_advect_geometryvopglobal1_v;
glm::vec3 geo1_solver1_d_s_particle_advect_multiply2_product = geo1_solver1_d_s_particle_advect_add1_sum * geo1_solver1_d_s_particle_advect_geometryvopglobal1_TimeInc;
glm::vec3 geo1_solver1_d_s_particle_advect_add2_sum = geo1_solver1_d_s_particle_advect_multiply2_product + geo1_solver1_d_s_particle_advect_geometryvopglobal1_P;
glm::vec3 geo1_solver1_d_s_particle_advect_fit1_shift = fit(geo1_solver1_d_s_particle_advect_curlnoise1_noise, CG_NONE, CG_NONE, CG_NONE, CG_NONE);


            // Write bacl
            glm::vec3 __geo1_solver1_d_s_particle_advect_add2_sum_debug = geo1_solver1_d_s_particle_advect_add2_sum;
__geo1_solver1_d_s_particle_advect_add2_sum_debug_buffer[idx] = __geo1_solver1_d_s_particle_advect_add2_sum_debug;

glm::vec3 __geo1_solver1_d_s_particle_advect_multiply2_product_debug = geo1_solver1_d_s_particle_advect_multiply2_product;
__geo1_solver1_d_s_particle_advect_multiply2_product_debug_buffer[idx] = __geo1_solver1_d_s_particle_advect_multiply2_product_debug;

glm::vec3 __geo1_solver1_d_s_particle_advect_add1_sum_debug = geo1_solver1_d_s_particle_advect_add1_sum;
__geo1_solver1_d_s_particle_advect_add1_sum_debug_buffer[idx] = __geo1_solver1_d_s_particle_advect_add1_sum_debug;

glm::vec3 __geo1_solver1_d_s_particle_advect_multiply1_product_debug = geo1_solver1_d_s_particle_advect_multiply1_product;
__geo1_solver1_d_s_particle_advect_multiply1_product_debug_buffer[idx] = __geo1_solver1_d_s_particle_advect_multiply1_product_debug;

glm::vec3 __geo1_solver1_d_s_particle_advect_fit1_shift_debug = geo1_solver1_d_s_particle_advect_fit1_shift;
__geo1_solver1_d_s_particle_advect_fit1_shift_debug_buffer[idx] = __geo1_solver1_d_s_particle_advect_fit1_shift_debug;

glm::vec3 __geo1_solver1_d_s_particle_advect_multiply3_product_debug = geo1_solver1_d_s_particle_advect_multiply3_product;
__geo1_solver1_d_s_particle_advect_multiply3_product_debug_buffer[idx] = __geo1_solver1_d_s_particle_advect_multiply3_product_debug;

glm::vec3 global_output_geo1_solver1_d_s_particle_advect_geometryvopoutput1_P = geo1_solver1_d_s_particle_advect_add2_sum;
geo1_solver1_d_s_particle_advect_geometryvopoutput1_Pbuffer[idx] = global_output_geo1_solver1_d_s_particle_advect_geometryvopoutput1_P;


        }
	} 
} 