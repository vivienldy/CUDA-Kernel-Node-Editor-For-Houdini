#include "SimpleParticle.h"
#include "CGField.h"

void CodeGenerator::particle_advect(int geo1_solver1_d_s_particle_advect_turb_turb, float geo1_solver1_d_s_particle_advect_amp_amp, glm::vec3 geo1_solver1_d_s_particle_advect_freq_freq, CGBuffer<glm::vec3>* geo1_solver1_d_s_particle_advect_geometryvopglobal1_Pbuffer, CGBuffer<glm::vec3>* geo1_solver1_d_s_particle_advect_geometryvopglobal1_vbuffer, float geo1_solver1_d_s_particle_advect_geometryvopglobal1_TimeInc, CGBuffer<float>* __geo1_solver1_d_s_particle_advect_add2_sum_debug_buffer, CGBuffer<float>* __geo1_solver1_d_s_particle_advect_multiply2_product_debug_buffer, CGBuffer<float>* __geo1_solver1_d_s_particle_advect_add1_sum_debug_buffer, CGBuffer<float>* __geo1_solver1_d_s_particle_advect_multiply1_product_debug_buffer, CGBuffer<float>* __geo1_solver1_d_s_particle_advect_fit1_shift_debug_buffer, CGBuffer<float>* __geo1_solver1_d_s_particle_advect_multiply3_product_debug_buffer)
{
    int numThreads = geo1_solver1_d_s_particle_advect_geometryvopglobal1_Pbuffer->getSize();

    for(int index = 0; index < numThreads; ++index){
        CodeGenerator::GenericCode::particle_advect(geo1_solver1_d_s_particle_advect_turb_turb, geo1_solver1_d_s_particle_advect_amp_amp, geo1_solver1_d_s_particle_advect_freq_freq, geo1_solver1_d_s_particle_advect_geometryvopglobal1_Pbuffer->getRawData(), geo1_solver1_d_s_particle_advect_geometryvopglobal1_vbuffer->getRawData(), geo1_solver1_d_s_particle_advect_geometryvopglobal1_TimeInc, __geo1_solver1_d_s_particle_advect_add2_sum_debug_buffer->getRawData(), __geo1_solver1_d_s_particle_advect_multiply2_product_debug_buffer->getRawData(), __geo1_solver1_d_s_particle_advect_add1_sum_debug_buffer->getRawData(), __geo1_solver1_d_s_particle_advect_multiply1_product_debug_buffer->getRawData(), __geo1_solver1_d_s_particle_advect_fit1_shift_debug_buffer->getRawData(), __geo1_solver1_d_s_particle_advect_multiply3_product_debug_buffer->getRawData(), index);
    }
}
