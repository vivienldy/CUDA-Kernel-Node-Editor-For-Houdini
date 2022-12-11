// 实现 CodeGenerator::ParticleAdvect() // CPU FORLOOP

#include "SimpleParticle.h"
#include "../../include/xnoise/XNoise.h"

int curlNoiseTest()
{
    std::string filename = "../userInputData/posRawBufferData.txt";
    auto geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer =
        dynamic_cast<CGBuffer<glm::vec3>*>(CGBuffer<float>::loadFromFile(filename));

    CurlNoise4DVector(
        MakeDefaultCurlNoise(),
        geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer,
        geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer,
        XNoiseDataManager::GetInstance()->GetXNoiseData(), 0.0f, 0.0416);

    geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer->printData();
    return -1;
}

int curlNoiseTestGPU()
{
    std::string filename = "../userInputData/posRawBufferData.txt";
    auto geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer =
        dynamic_cast<CGBuffer<glm::vec3>*>(CGBuffer<float>::loadFromFile(filename));

    geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer->malloc();
    geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer->loadHostToDevice();

    auto noiseParam = MakeDefaultCurlNoiseDevice();
    CUDA::CurlNoise4DVector(
        noiseParam,
        geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer,
        geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer,
        XNoiseDataManager::GetInstance()->GetXNoiseDataDevice(), 0.0f, 0.0416);


    geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer->loadDeviceToHost();
    geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer->printData();
    return -1;
}


int main() // ? 还是main函数应该是现在单独的main.cpp 里
{
    //curlNoiseTest();
    //curlNoiseTestGPU();
    //return -1;
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

    // ===== load from another json???
    int startFrame = 0;
    int endFrame = 500;
    float FPS = 24.f;
    int blockSize = 128;
    float TimeInc = 1.0 / FPS;

#if GPU_VERSION 
    geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer->malloc();
    geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer->loadHostToDevice();

    geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_vbuffer->malloc();
    geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_vbuffer->loadHostToDevice();

    __geo1_solver1_d_s_pointvop2__DEBUG_add2_sum_debug_buffer->malloc();
    __geo1_solver1_d_s_pointvop2__DEBUG_add2_sum_debug_buffer->loadHostToDevice();

    __geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug_buffer->malloc();
    __geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug_buffer->loadHostToDevice();

    __geo1_solver1_d_s_pointvop2__DEBUG_add1_sum_debug_buffer->malloc();
    __geo1_solver1_d_s_pointvop2__DEBUG_add1_sum_debug_buffer->loadHostToDevice();

    __geo1_solver1_d_s_pointvop2__DEBUG_multiply1_product_debug_buffer->malloc();
    __geo1_solver1_d_s_pointvop2__DEBUG_multiply1_product_debug_buffer->loadHostToDevice();

    __geo1_solver1_d_s_pointvop2__DEBUG_curlnoise1_noise_debug_buffer->malloc();
    __geo1_solver1_d_s_pointvop2__DEBUG_curlnoise1_noise_debug_buffer->loadHostToDevice();

    __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer->malloc();
    __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer->loadHostToDevice();

    __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer->malloc();
    __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer->loadHostToDevice();
#endif

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
            __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer);

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
            blockSize);
        cudaDeviceSynchronize();
        geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer->loadDeviceToHost();
#endif

        // save pos buffer as obj file
        std::string frame = std::to_string(Frame);
        std::string outputObjFilePath = "../userOutputData/pos_";
// for debug
#if CPU_VERSION
        outputObjFilePath.append("cpu_");
#elif GPU_VERSION
        outputObjFilePath.append("gpu_");
#endif
        outputObjFilePath.append(frame);
        outputObjFilePath.append(".obj");

        //geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer->outputObj(outputObjFilePath);
    }
}


