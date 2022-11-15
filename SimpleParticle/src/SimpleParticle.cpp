// 实现 CodeGenerator::ParticleAdvect() // CPU FORLOOP

#include "SimpleParticle.h"

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
    CGBuffer<glm::vec3>* __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer,
    CGBuffer<glm::vec3>* geo1_solver1_d_s_pointvop2__DEBUG_geometryvopoutput1_Pbuffer)
{
     auto geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer_raw = geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer->getRawData();
     auto geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_vbuffer_raw = geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_vbuffer->getRawData();
     
     auto __geo1_solver1_d_s_pointvop2__DEBUG_add2_sum_debug_buffer_raw = __geo1_solver1_d_s_pointvop2__DEBUG_add2_sum_debug_buffer->getRawData();
     auto __geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug_buffer_raw = __geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug_buffer->getRawData();
     auto __geo1_solver1_d_s_pointvop2__DEBUG_add1_sum_debug_buffer_raw = __geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug_buffer->getRawData();
     auto __geo1_solver1_d_s_pointvop2__DEBUG_multiply1_product_debug_buffer_raw = __geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug_buffer->getRawData();
     auto __geo1_solver1_d_s_pointvop2__DEBUG_curlnoise1_noise_debug_buffer_raw = __geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug_buffer->getRawData();
     auto __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer_raw = __geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug_buffer->getRawData();
     
     auto geo1_solver1_d_s_pointvop2__DEBUG_geometryvopoutput1_Pbuffer_raw = __geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug_buffer->getRawData();

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
              geo1_solver1_d_s_pointvop2__DEBUG_geometryvopoutput1_Pbuffer_raw,
              i);
     }
}

int main() // ? 还是main函数应该是现在单独的main.cpp 里
{
    // ===== create input buffer
    // pbuffer, vbuffer, timeInc is defined inside for loop
    // load pbuffer from file
    std::string filename = "../userInputData/posRawBufferData.txt";
    auto geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer =
        dynamic_cast<CGBuffer<glm::vec3>*>(CGBuffer<float>::loadFromFile(filename));

    int numPoints = geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer->getSize();

    // dynamically create and initialize vbuffer
    auto geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_vbuffer = new CGBuffer<glm::vec3>(
        "vel",
        numPoints,
        glm::vec3(0.f));

    // ===== create vop user custom parameter
    // load from another json ???
    int geo1_solver1_d_s_pointvop2__DEBUG_turb_turb = 3;
    float geo1_solver1_d_s_pointvop2__DEBUG_amp_amp = 3.f;
    glm::vec3 geo1_solver1_d_s_pointvop2__DEBUG_freq_freq = glm::vec3(1.f, 2.f, 1.f);

    // dynamically create and initalize debug buffer
    auto __geo1_solver1_d_s_pointvop2__DEBUG_add2_sum_debug_buffer = new CGBuffer<glm::vec3>(
        "__geo1_solver1_d_s_pointvop2__DEBUG_add2_sum_debug_buffer",
        numPoints,
        glm::vec3(0.f));
    auto __geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug_buffer = new CGBuffer<glm::vec3>(
        "__geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug_buffer",
        numPoints,
        glm::vec3(0.f));
    auto __geo1_solver1_d_s_pointvop2__DEBUG_add1_sum_debug_buffer = new CGBuffer<glm::vec3>(
        "__geo1_solver1_d_s_pointvop2__DEBUG_add1_sum_debug_buffer",
        numPoints,
        glm::vec3(0.f));
    auto __geo1_solver1_d_s_pointvop2__DEBUG_multiply1_product_debug_buffer = new CGBuffer<glm::vec3>(
        "__geo1_solver1_d_s_pointvop2__DEBUG_multiply1_product_debug_buffer",
        numPoints,
        glm::vec3(0.f));
    auto __geo1_solver1_d_s_pointvop2__DEBUG_curlnoise1_noise_debug_buffer = new CGBuffer<glm::vec3>(
        "__geo1_solver1_d_s_pointvop2__DEBUG_curlnoise1_noise_debug_buffer",
        numPoints,
        glm::vec3(0.f));
    auto __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer = new CGBuffer<glm::vec3>(
        "__geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer",
        numPoints,
        glm::vec3(0.f));

    // ===== create output buffer
    // pbuffer
    // initialize by copying the input pbuffer
    auto geo1_solver1_d_s_pointvop2__DEBUG_geometryvopoutput1_Pbuffer = new CGBuffer<glm::vec3>();
    geo1_solver1_d_s_pointvop2__DEBUG_geometryvopoutput1_Pbuffer->copy(geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer);

    // ===== load from another json???
    int startFrame = 0;
    int endFrame = 10;
    float FPS = 24.f;
    int blockSize = 128;
    float TimeInc = 1.0 / FPS;

    for (int i = startFrame; i < endFrame; ++i) {
        //hard code var block
        float Time = i / FPS;
        float Frame = i;

#if CPU_VERSION
        CodeGenerator::ParticleAdvect(
            geo1_solver1_d_s_pointvop2__DEBUG_turb_turb,
            geo1_solver1_d_s_pointvop2__DEBUG_amp_amp,
            geo1_solver1_d_s_pointvop2__DEBUG_freq_freq,
            geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer,
            geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_vbuffer,
            TimeInc,
            __geo1_solver1_d_s_pointvop2__DEBUG_add2_sum_debug_buffer,
            __geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug_buffer,
            __geo1_solver1_d_s_pointvop2__DEBUG_add1_sum_debug_buffer,
            __geo1_solver1_d_s_pointvop2__DEBUG_multiply1_product_debug_buffer,
            __geo1_solver1_d_s_pointvop2__DEBUG_curlnoise1_noise_debug_buffer,
            __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer,
            geo1_solver1_d_s_pointvop2__DEBUG_geometryvopoutput1_Pbuffer);

#elif GPU_VERSION
        CodeGenerator::CUDA::ParticleAdvect(
            geo1_solver1_d_s_pointvop2__DEBUG_turb_turb,
            geo1_solver1_d_s_pointvop2__DEBUG_amp_amp,
            geo1_solver1_d_s_pointvop2__DEBUG_freq_freq,
            geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer,
            geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_vbuffer,
            TimeInc,
            __geo1_solver1_d_s_pointvop2__DEBUG_add2_sum_debug_buffer,
            __geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug_buffer,
            __geo1_solver1_d_s_pointvop2__DEBUG_add1_sum_debug_buffer,
            __geo1_solver1_d_s_pointvop2__DEBUG_multiply1_product_debug_buffer,
            __geo1_solver1_d_s_pointvop2__DEBUG_curlnoise1_noise_debug_buffer,
            __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer,
            geo1_solver1_d_s_pointvop2__DEBUG_geometryvopoutput1_Pbuffer,
            blockSize);
#endif

        // save pos buffer as obj file
        std::string frame = std::to_string(Frame);
        std::string outputObjFilePath = "../userOutputData/pos_";
        outputObjFilePath.append(frame);
        outputObjFilePath.append(".obj");
        geo1_solver1_d_s_pointvop2__DEBUG_geometryvopoutput1_Pbuffer->outputObj(outputObjFilePath);
    }
}
