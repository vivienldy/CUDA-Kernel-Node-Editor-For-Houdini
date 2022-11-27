#include "SimpleParticle.h"
#include "CGField.h"

void CodeGenerator::ParticleAdvect(
    int geo1_solver1_d_s_pointvop2__DEBUG_turb_turb,
    float geo1_solver1_d_s_pointvop2__DEBUG_amp_amp,
    glm::vec3 geo1_solver1_d_s_pointvop2__DEBUG_freq_freq,
    CGBuffer<glm::vec3>* geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer,
    CGBuffer<glm::vec3>* geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_vbuffer,
    float TimeInc,
    CGBuffer<glm::vec3>* __geo1_solver1_d_s_pointvop2__DEBUG_add2_sum_debug_buffer,
    CGBuffer<glm::vec3>* __geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug_buffer,
    CGBuffer<glm::vec3>* __geo1_solver1_d_s_pointvop2__DEBUG_add1_sum_debug_buffer,
    CGBuffer<glm::vec3>* __geo1_solver1_d_s_pointvop2__DEBUG_multiply1_product_debug_buffer,
    CGBuffer<glm::vec3>* __geo1_solver1_d_s_pointvop2__DEBUG_curlnoise1_noise_debug_buffer,
    CGBuffer<glm::vec3>* __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer)
{
     auto geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer_raw = geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer->getRawData();
     auto geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_vbuffer_raw = geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_vbuffer->getRawData();
     
     auto __geo1_solver1_d_s_pointvop2__DEBUG_add2_sum_debug_buffer_raw = __geo1_solver1_d_s_pointvop2__DEBUG_add2_sum_debug_buffer->getRawData();
     auto __geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug_buffer_raw = __geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug_buffer->getRawData();
     auto __geo1_solver1_d_s_pointvop2__DEBUG_add1_sum_debug_buffer_raw = __geo1_solver1_d_s_pointvop2__DEBUG_add1_sum_debug_buffer->getRawData();
     auto __geo1_solver1_d_s_pointvop2__DEBUG_multiply1_product_debug_buffer_raw = __geo1_solver1_d_s_pointvop2__DEBUG_multiply1_product_debug_buffer->getRawData();
     auto __geo1_solver1_d_s_pointvop2__DEBUG_curlnoise1_noise_debug_buffer_raw = __geo1_solver1_d_s_pointvop2__DEBUG_curlnoise1_noise_debug_buffer->getRawData();
     auto __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer_raw = __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer->getRawData();
     
     int numThreads = geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer->getSize();

     for(int i = 0; i < numThreads; ++i){
          CodeGenerator::GenericCode::particleAdvect(
              geo1_solver1_d_s_pointvop2__DEBUG_turb_turb,
              geo1_solver1_d_s_pointvop2__DEBUG_amp_amp,
              geo1_solver1_d_s_pointvop2__DEBUG_freq_freq,
              geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer_raw,
              geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_vbuffer_raw,
              TimeInc,
              __geo1_solver1_d_s_pointvop2__DEBUG_add2_sum_debug_buffer_raw,
              __geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug_buffer_raw,
              __geo1_solver1_d_s_pointvop2__DEBUG_add1_sum_debug_buffer_raw,
              __geo1_solver1_d_s_pointvop2__DEBUG_multiply1_product_debug_buffer_raw,
              __geo1_solver1_d_s_pointvop2__DEBUG_curlnoise1_noise_debug_buffer_raw,
              __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer_raw,
              i);
     }
}


