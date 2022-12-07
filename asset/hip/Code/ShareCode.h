#include <glm/glm.hpp>

namespace CodeGenerator 
{ 
	namespace GenericCode 
 	{ 
        __host__ __device__ inline void particle_advect(int geo1_solver1_d_s_particle_advect_turb_turb, float geo1_solver1_d_s_particle_advect_amp_amp, glm::vec3 geo1_solver1_d_s_particle_advect_freq_freq, glm::vec3* geo1_solver1_d_s_particle_advect_geometryvopglobal1_Pbuffer, glm::vec3* geo1_solver1_d_s_particle_advect_geometryvopglobal1_vbuffer, float geo1_solver1_d_s_particle_advect_geometryvopglobal1_TimeInc, int idx)
        {
            // Data Load 
            // Geometry Global Input
glm::vec3 geo1_solver1_d_s_particle_advect_geometryvopglobal1_P = geo1_solver1_d_s_particle_advect_geometryvopglobal1_Pbuffer[idx];
glm::vec3 geo1_solver1_d_s_particle_advect_geometryvopglobal1_v = geo1_solver1_d_s_particle_advect_geometryvopglobal1_vbuffer[idx];


            // Compute graph
            float geo1_solver1_d_s_particle_advect_Node_Value = 0.9f;
glm::vec3 geo1_solver1_d_s_particle_advect_multiply3_product = geo1_solver1_d_s_particle_advect_geometryvopglobal1_P * geo1_solver1_d_s_particle_advect_Node_Value;
glm::vec3 geo1_solver1_d_s_particle_advect_curlnoise1_noise = curlnoise(char(pnoise), geo1_solver1_d_s_particle_advect_multiply3_product, geo1_solver1_d_s_particle_advect_freq_freq, glm::vec3(0.0f,0.0f,0.0f), geo1_solver1_d_s_particle_advect_amp_amp, float(0.5f), float(1.0f), geo1_solver1_d_s_particle_advect_turb_turb, float(0.0001f), float(1.0f), float(1.0f), glm::vec3(0.0f,0.0f,0.0f), char(), int(0));
glm::vec3 geo1_solver1_d_s_particle_advect_multiply1_product = geo1_solver1_d_s_particle_advect_curlnoise1_noise * geo1_solver1_d_s_particle_advect_geometryvopglobal1_TimeInc;
glm::vec3 geo1_solver1_d_s_particle_advect_add1_sum = geo1_solver1_d_s_particle_advect_multiply1_product + geo1_solver1_d_s_particle_advect_geometryvopglobal1_v;
glm::vec3 geo1_solver1_d_s_particle_advect_multiply2_product = geo1_solver1_d_s_particle_advect_add1_sum * geo1_solver1_d_s_particle_advect_geometryvopglobal1_TimeInc;
glm::vec3 geo1_solver1_d_s_particle_advect_add2_sum = geo1_solver1_d_s_particle_advect_multiply2_product + geo1_solver1_d_s_particle_advect_geometryvopglobal1_P;


            // Write bacl
            glm::vec3 global_output_geo1_solver1_d_s_particle_advect_geometryvopoutput1_P = geo1_solver1_d_s_particle_advect_add2_sum;
geo1_solver1_d_s_particle_advect_geometryvopglobal1_Pbuffer[idx] = global_output_geo1_solver1_d_s_particle_advect_geometryvopoutput1_P;


        }
	} 
} 