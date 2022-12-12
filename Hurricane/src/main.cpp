#include "volumevop1.h"
#include "ParticleAdvect.h"

#include "BaseOperation.h"
#include "CGGenerator.h"
#include "CGGeometry.h"
#include "glm/gtc/matrix_transform.hpp"

#define CPU_VERSION 0
#define GPU_VERSION 1

void colorCreateVolumeCreate() {
    glm::vec3 pivot = glm::vec3(-1.f, 0.6f, 3.7f);
    glm::vec3 res = glm::vec3(65, 7, 64);
    glm::vec3 fieldSize = glm::vec3(13.f, 1.4f, 12.8f);
    float voxelSize = 0.2f;
    std::string filename = "../userInputData/color_create_volume.txt";
    CGField3D<float>* noiseColor = new CGField3D<float>(pivot, res, fieldSize, voxelSize, "noise_color", filename);
    CGVectorField3D<float>* colorField = new CGVectorField3D<float>(noiseColor, noiseColor, noiseColor, "noise_color_field");
}

int main() {
    // ===== create input buffer
    // pbuffer for particle advect
    // load pbuffer from file

    int numPoints = 2;
    auto posBuffer = new CGBuffer<glm::vec3>("position", numPoints, glm::vec3(0.f));

    // dynamically create and initialize buffers
    auto velBuffer = new CGBuffer<glm::vec3>("velocity", numPoints, glm::vec3(0.f));
    auto ageBuffer = new CGBuffer<float>("age", numPoints, 0.f);
    auto cdBuffer = new CGBuffer<glm::vec3>("color", numPoints, glm::vec3(0.f));
    auto agePingpongBuffer = new CGBuffer<float>("age", numPoints, 0.f); 

    // ===== create particle emitter
    // initilize data read from task json
    ParticleGenerator::RAWDesc desc;
    desc.direction = glm::vec3(0, 0, 1);
    desc.speed = 1;
    desc.size = glm::vec2(12, 12);
    desc.deltaX = 0.27f;
    desc.center = glm::vec3(-1.f, 0.5f, 1.77f);

    ParticleGenerator* particleGenerator = new ParticleGenerator(desc);
    particleGenerator->delegatePointBuffer(posBuffer);
    particleGenerator->delegatePointBuffer(ageBuffer);
    particleGenerator->delegatePointBuffer(agePingpongBuffer);
    particleGenerator->delegatePointBuffer(velBuffer);
    particleGenerator->delegatePointBuffer(cdBuffer);

    // ====== create velocity field
    glm::vec3 pivot = glm::vec3(-1.17241f, 12.8777f, 1.88102f);
    glm::vec3 res = glm::vec3(22, 33, 22);
    glm::vec3 fieldSize = glm::vec3(17.6f, 26.4f, 17.6f);
    float voxelSize = 0.8f;

    // field data is initialize to zero
    CGField3D<float>* fieldX = new CGField3D<float>(
        pivot,
        res,
        fieldSize,
        voxelSize,
        "vel_x");
    CGField3D<float>* fieldY = new CGField3D<float>(
        pivot,
        res,
        fieldSize,
        voxelSize,
        "vel_y");
    CGField3D<float>* fieldZ = new CGField3D<float>(
        pivot,
        res,
        fieldSize,
        voxelSize,
        "vel_z");
    CGVectorField3D<float>* velocityField = new CGVectorField3D<float>(
        fieldX,
        fieldY,
        fieldZ,
        "velocity_field");

    // ===== create vop user custom parameter
    float volumeVop1_input5 = 0.f; // max radius
    float  volumeVop1_input4 = 7.7f; // center cohesion
    float volumeVop1_input3 = 0.079f; // up vector push
    float volumeVop1_input2 = 25.9f; // input number 2

    float particleAdvectVop_input2 = 0.5f;
    float particleAdvectVop_input3 = 3.f;
    glm::vec3 particleAdvectVop_offset = glm::vec3(0.f, 1.f, 0.f);

    // ===== create vop CGGeometry input/OpInput
     // volumeVop_OpInput1: a velfield, a geometry bounding box
     // volumeVop1_OpInput2: a geometry bounding box
    CGAABB volumeVop_OpInput1_bbox = fieldX->GetAABB();
    CGGeometry* volumeVop1_OpInput1 = new CGGeometry(volumeVop_OpInput1_bbox, nullptr, nullptr, velocityField);

    glm::vec4 aabb_min = glm::vec4(-2.01958f, 2.38065f, 4.53478f, 1.f);
    glm::vec4 aabb_max = glm::vec4(-1.01958f, 3.38065f, 5.53478f, 1.f);
    CGAABB volumeVop_OpInput2_bbox = CGAABB(glm::vec3(aabb_min), glm::vec3(aabb_max));
    CGGeometry* volumeVop1_OpInput2 = new CGGeometry(volumeVop_OpInput2_bbox, nullptr, nullptr, nullptr);

    // particleAdvectVop_OpInput1: a volume for creating color
    // particleAdvectVop_OpInput2: a velocity field
    // create the volume for creating color from file
    glm::vec3 cdVolumePivot = glm::vec3(-1.f, 0.6f, 1.77f);
    glm::vec3  cdVolumeRes = glm::vec3(65, 7, 64);
    glm::vec3  cdVolumeFieldSize = glm::vec3(13.f, 1.4f, 12.8f);
    float  cdVolumeVoxelSize = 0.2f;
    std::string  cdVolumeFilename = "../userInputData/color_create_volume.txt";
    CGField3D<float>* noiseColor = new CGField3D<float>(cdVolumePivot, cdVolumeRes, cdVolumeFieldSize, cdVolumeVoxelSize, "noise_color", cdVolumeFilename);
    CGVectorField3D<float>* colorField = new CGVectorField3D<float>(noiseColor, noiseColor, noiseColor, "noise_color_field");
    CGGeometry* particleAdvectVop_OpInput1 = new CGGeometry(CGAABB(), nullptr, nullptr, colorField);

    CGGeometry* particleAdvectVop_OpInput2 = new CGGeometry(CGAABB(), nullptr, nullptr, velocityField);

     // ===== load from task json
    int startFrame = 0;
    int endFrame = 200;
    float FPS = 24.f;
    int blockSize = 128;
    float TimeInc = 1.0 / FPS;

#if GPU_VERSION
    volumeVop1_OpInput1->DeviceMalloc();
    volumeVop1_OpInput1->LoadToDevice();

    volumeVop1_OpInput2->DeviceMalloc();
    volumeVop1_OpInput2->LoadToDevice();

    posBuffer->malloc();
    posBuffer->loadHostToDevice();
    ageBuffer->malloc();
    ageBuffer->loadHostToDevice();
    cdBuffer->malloc();
    cdBuffer->loadHostToDevice();
    agePingpongBuffer->malloc();
    agePingpongBuffer->loadHostToDevice();

    particleAdvectVop_OpInput1->DeviceMalloc();
    particleAdvectVop_OpInput1->LoadToDevice();

    particleAdvectVop_OpInput2->DeviceMalloc();
    particleAdvectVop_OpInput2->LoadToDevice();
# endif

    for (int i = startFrame; i < endFrame; ++i) {
        //hard code var block
        float Time = i / FPS;
        float Frame = i;

        glm::mat4 transM = glm::translate(glm::mat4(1.0f), glm::vec3(glm::sin(glm::radians(Frame*2.f)), -0.1524f, glm::cos(glm::radians(Frame * 2.f))));
        aabb_min = transM * glm::vec4(-2.01958f, 2.38065f, 4.53478f, 1.f);
        aabb_max = transM * glm::vec4(-1.01958f, 3.38065f, 5.53478f, 1.f);
        volumeVop_OpInput2_bbox = CGAABB(glm::vec3(aabb_min), glm::vec3(aabb_max));
        volumeVop1_OpInput2->m_Bbox = volumeVop_OpInput2_bbox;

#if CPU_VERSION
        // particle emitter
        particleGenerator->generateParticlesCPU();

        // create velocity field
        CodeGenerator::volumevop1(
            volumeVop1_input5,
            volumeVop1_input4,
            volumeVop1_input3,
            volumeVop1_input2,
            volumeVop1_OpInput1,
            volumeVop1_OpInput2);

        CodeGenerator::ParticleAdvect(
            TimeInc * Frame,
            particleAdvectVop_input3,
            particleAdvectVop_input2,
            posBuffer, 
            ageBuffer,
            cdBuffer,
            TimeInc, 
            particleAdvectVop_OpInput1,
            particleAdvectVop_OpInput2,
            agePingpongBuffer);

        ageBuffer->copy(agePingpongBuffer);
#elif GPU_VERSION
        // particle emitter
        particleGenerator->generateParticlesGPU();

        CodeGenerator::CUDA::volumevop1(
            volumeVop1_input5,
            volumeVop1_input4,
            volumeVop1_input3,
            volumeVop1_input2,
            volumeVop1_OpInput1,
            volumeVop1_OpInput2);

        //velocityField->LoadToHost();

        CodeGenerator::CUDA::ParticleAdvect(
            (TimeInc * Frame),
            particleAdvectVop_input3,
            particleAdvectVop_input2,
            posBuffer,
            ageBuffer,
            cdBuffer,
            TimeInc,
            particleAdvectVop_OpInput1,
            particleAdvectVop_OpInput2,
            agePingpongBuffer);
       
        cudaMemcpy(ageBuffer->getDevicePointer(), agePingpongBuffer->getDevicePointer(), sizeof(float) * ageBuffer->getSize(), cudaMemcpyDeviceToDevice);
        posBuffer->loadDeviceToHost();
        cdBuffer->loadDeviceToHost();
#endif

        // save vel_field buffer as obj file
        std::string frame = std::to_string(Frame+1);
        std::string outputObjFilePathBase = "../userOutputData/vel_field_test";
        std::string velXFilePath = "../userOutputData/vel_field_test/velx_move_";

#if CPU_VERSION
        velXFilePath.append("cpu_");

#elif GPU_VERSION
        velXFilePath.append("gpu_");

#endif
        velXFilePath.append(frame);
        velXFilePath.append(".obj");
        //velocityField->WriteFieldAsObj(velXFilePath);

        // save pos buffer as obj file
        std::string posOutputObjFilePathBase = "../userOutputData/pos/pos_noise_move_";
        std::string cdOutputObjFilePathBase = "../userOutputData/pos/cd_noise_move_";

#if CPU_VERSION
        posOutputObjFilePathBase.append("cpu_");
        cdOutputObjFilePathBase.append("cpu_");
#elif GPU_VERSION
        posOutputObjFilePathBase.append("gpu_");
        cdOutputObjFilePathBase.append("gpu_");
#endif
        posOutputObjFilePathBase.append(frame);
        posOutputObjFilePathBase.append(".obj");
        cdOutputObjFilePathBase.append(frame);
        cdOutputObjFilePathBase.append(".obj");

        posBuffer->outputObj(posOutputObjFilePathBase);
        cdBuffer->outputObj(cdOutputObjFilePathBase);
        std::cout << "-------- frame: "  << frame << std::endl;
    }
}