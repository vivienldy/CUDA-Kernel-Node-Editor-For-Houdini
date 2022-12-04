#include "CGBuffer.h"
#include "SimpleParticle.GenericCode.h"

#define CPU_VERSION 0
#define GPU_VERSION 1

namespace CodeGenerator
{
    void particle_advect(int geo1_solver1_d_s_particle_advect_turb_turb, float geo1_solver1_d_s_particle_advect_amp_amp, glm::vec3 geo1_solver1_d_s_particle_advect_freq_freq, CGBuffer<glm::vec3>* geo1_solver1_d_s_particle_advect_geometryvopglobal1_Pbuffer, CGBuffer<glm::vec3>* geo1_solver1_d_s_particle_advect_geometryvopglobal1_vbuffer, float geo1_solver1_d_s_particle_advect_geometryvopglobal1_TimeInc, CGBuffer<float>* __geo1_solver1_d_s_particle_advect_add2_sum_debug_buffer, CGBuffer<float>* __geo1_solver1_d_s_particle_advect_multiply2_product_debug_buffer, CGBuffer<float>* __geo1_solver1_d_s_particle_advect_add1_sum_debug_buffer, CGBuffer<float>* __geo1_solver1_d_s_particle_advect_multiply1_product_debug_buffer, CGBuffer<float>* __geo1_solver1_d_s_particle_advect_fit1_shift_debug_buffer, CGBuffer<float>* __geo1_solver1_d_s_particle_advect_multiply3_product_debug_buffer);
    namespace CUDA 
    {
        void particle_advect(int geo1_solver1_d_s_particle_advect_turb_turb, float geo1_solver1_d_s_particle_advect_amp_amp, glm::vec3 geo1_solver1_d_s_particle_advect_freq_freq, CGBuffer<glm::vec3>* geo1_solver1_d_s_particle_advect_geometryvopglobal1_Pbuffer, CGBuffer<glm::vec3>* geo1_solver1_d_s_particle_advect_geometryvopglobal1_vbuffer, float geo1_solver1_d_s_particle_advect_geometryvopglobal1_TimeInc, CGBuffer<float>* __geo1_solver1_d_s_particle_advect_add2_sum_debug_buffer, CGBuffer<float>* __geo1_solver1_d_s_particle_advect_multiply2_product_debug_buffer, CGBuffer<float>* __geo1_solver1_d_s_particle_advect_add1_sum_debug_buffer, CGBuffer<float>* __geo1_solver1_d_s_particle_advect_multiply1_product_debug_buffer, CGBuffer<float>* __geo1_solver1_d_s_particle_advect_fit1_shift_debug_buffer, CGBuffer<float>* __geo1_solver1_d_s_particle_advect_multiply3_product_debug_buffer, int blockSize = 512);
    }
}
