#include "@PROJ_NAME@.h"

int main() 
{
    // ===== create input buffer
    // pbuffer, vbuffer, timeInc is defined inside for loop
    // load pbuffer from file
    // std::string filename = "../userInputData/posRawBufferData.txt";
    // auto geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer =
    //     dynamic_cast<CGBuffer<glm::vec3>*>(CGBuffer<float>::loadFromFile(filename));

    @GLOBAL_INIT_LOAD@

    // int numPoints = geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer->getSize();
    int numPoints = @GLOBAL_GET_SIZE@;

    // dynamically create and initialize vbuffer
    // auto geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_vbuffer = new CGBuffer<glm::vec3>(
    //     "vel",
    //     numPoints,
    //     glm::vec3(0.f));
    @GLOBAL_INIT@


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
    @CUSTOM_INIT@

    // ===== load from another json???
    // int startFrame = 0;
    // int endFrame = 500;
    // float FPS = 24.f;
    // int blockSize = 128;
    @PARAM_INIT@

    float TimeInc = 1.0 / FPS;

    for (int i = startFrame; i < endFrame; ++i) {
        //hard code var block
        float Time = i / FPS;
        float Frame = i;

#if CPU_VERSION
        CodeGenerator::@FUNC_NAME@(
            @FUNC_PARAM_LIST@);

#elif GPU_VERSION
        CodeGenerator::CUDA::@FUNC_NAME@(
            @FUNC_PARAM_LIST@,
            blockSize);
        cudaDeviceSynchronize();
        //geo1_solver1_d_s_pointvop2__DEBUG_geometryvopglobal1_Pbuffer->loadDeviceToHost();
        @GLOABL_LOAD_TO_HOST@
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
        @GLOBAL_LOAD_TO_OBJ@
    }
}
