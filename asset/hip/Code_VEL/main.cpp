#include "SimpleParticle.h"

int main() 
{
    // ===== create input buffer
    // pbuffer, vbuffer, timeInc is defined inside for loop
    // load pbuffer from file
    // std::string filename = "../userInputData/posRawBufferData.txt";
    // auto geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer =
    //     dynamic_cast<CGBuffer<glm::vec3>*>(CGBuffer<float>::loadFromFile(filename));

    auto pbuffer = dynamic_cast<CGBuffer<glm::vec3>*>(CGBuffer<float>::loadFromFile("../userInputData/posRawBufferData.txt")); 


    // int numPoints = geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer->getSize();
    int numPoints = pbuffer->getSize(); 
;

    // dynamically create and initialize vbuffer
    // auto geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_vbuffer = new CGBuffer<glm::vec3>(
    //     "vel",
    //     numPoints,
    //     glm::vec3(0.f));
    auto vbuffer = new CGBuffer<glm::vec3>("v", numPoints, glm::vec3(0, 0, 0)); 



    // ===== create vop user custom parameter
    // load from another json ???
    // int geo1_solver1_d_s_pointvop2__DEBUG_turb_turb = 3;
    // float geo1_solver1_d_s_pointvop2__DEBUG_amp_amp = 3.f;
    // glm::vec3 geo1_solver1_d_s_pointvop2__DEBUG_freq_freq = glm::vec3(1.f, 2.f, 1.f);

    // dynamically create and initalize debug buffer
    // auto __geo1_solver1_d_s_pointvop2__DEBUG_add2_sum_debug_buffer = new CGBuffer<glm::vec3>(
    //     "__geo1_solver1_d_s_pointvop2__DEBUG_add2_sum_debug_buffer",
    //     numPoints,
    //     glm::vec3(0.f));
    // auto __geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug_buffer = new CGBuffer<glm::vec3>(
    //     "__geo1_solver1_d_s_pointvop2__DEBUG_multiply2_product_debug_buffer",
    //     numPoints,
    //     glm::vec3(0.f));
    // auto __geo1_solver1_d_s_pointvop2__DEBUG_add1_sum_debug_buffer = new CGBuffer<glm::vec3>(
    //     "__geo1_solver1_d_s_pointvop2__DEBUG_add1_sum_debug_buffer",
    //     numPoints,
    //     glm::vec3(0.f));
    // auto __geo1_solver1_d_s_pointvop2__DEBUG_multiply1_product_debug_buffer = new CGBuffer<glm::vec3>(
    //     "__geo1_solver1_d_s_pointvop2__DEBUG_multiply1_product_debug_buffer",
    //     numPoints,
    //     glm::vec3(0.f));
    // auto __geo1_solver1_d_s_pointvop2__DEBUG_curlnoise1_noise_debug_buffer = new CGBuffer<glm::vec3>(
    //     "__geo1_solver1_d_s_pointvop2__DEBUG_curlnoise1_noise_debug_buffer",
    //     numPoints,
    //     glm::vec3(0.f));
    // auto __geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer = new CGBuffer<glm::vec3>(
    //     "__geo1_solver1_d_s_pointvop2__DEBUG_multiply3_product_debug_buffer",
    //     numPoints,
    //     glm::vec3(0.f));
    int geo1_solver1_d_s_pointvop2__DEBUG_turb_turb = 3; 
	float geo1_solver1_d_s_pointvop2__DEBUG_amp_amp = 3.0; 
	glm::vec3 geo1_solver1_d_s_pointvop2__DEBUG_freq_freq = glm::vec3(1.0, 2.0, 1.0); 
	auto __geo1_solver1_d_s_pointvop2__DEBUG_add2_sum_debug_buffer_buffer = new CGBuffer<glm::vec3>("__geo1_solver1_d_s_pointvop2__DEBUG_add2_sum_debug_buffer", numPoints, glm::vec3(0)); 
	auto multiply2_product_debug_buffer = new CGBuffer<glm::vec3>("multiply2_product_debug", numPoints, glm::vec3(0)); 
	auto add1_sum_debug_buffer = new CGBuffer<glm::vec3>("add1_sum_debug", numPoints, glm::vec3(0)); 
	

    // ===== load from another json???
    // int startFrame = 0;
    // int endFrame = 500;
    // float FPS = 24.f;
    // int blockSize = 128;
    int startFrame = 0; 
	int endFrame = 100; 
	float FPS = 24; 
	int blockSize = 128; 
	

    float TimeInc = 1.0 / FPS;

    for (int i = startFrame; i < endFrame; ++i) {
        //hard code var block
        float Time = i / FPS;
        float Frame = i;

#if CPU_VERSION
        CodeGenerator::particle_advect(
            geo1_solver1_d_s_pointvop2__DEBUG_turb_turb, geo1_solver1_d_s_pointvop2__DEBUG_amp_amp, geo1_solver1_d_s_pointvop2__DEBUG_freq_freq, pbuffer, vbuffer, __geo1_solver1_d_s_pointvop2__DEBUG_add2_sum_debug_buffer_buffer, multiply2_product_debug_buffer, add1_sum_debug_buffer);

#elif GPU_VERSION
        CodeGenerator::CUDA::particle_advect(
            geo1_solver1_d_s_pointvop2__DEBUG_turb_turb, geo1_solver1_d_s_pointvop2__DEBUG_amp_amp, geo1_solver1_d_s_pointvop2__DEBUG_freq_freq, pbuffer, vbuffer, __geo1_solver1_d_s_pointvop2__DEBUG_add2_sum_debug_buffer_buffer, multiply2_product_debug_buffer, add1_sum_debug_buffer,
            blockSize);
        cudaDeviceSynchronize();
        //geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer->loadDeviceToHost();
        pbuffer->loadDeviceToHost(); 
		vbuffer->loadDeviceToHost(); 
		
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
        pbuffer->outputObj(outputObjFilePath); 

    }
}
