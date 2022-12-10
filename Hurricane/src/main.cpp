#include "volumevop1.h"

#include "BaseOperation.h"
#include "CGGenerator.h"
#include "CGGeometry.h"

#define CPU_VERSION 0
#define GPU_VERSION 1

void testFieldMain() {
    glm::vec3 pivot = glm::vec3(0.f);

    glm::vec3 res = glm::vec3(5, 5, 5);

    glm::vec3 fieldSize = glm::vec3(1.8f, 1.8f, 1.8f);

    float voxelSize = 0.36f;

    std::string filename = "../userInputData/3dfield_data.txt";

    CGField3D<float>* fieldX = new CGField3D<float>(
        pivot,
        res,
        fieldSize,
        voxelSize,
        "test_field_x",
        filename);

    CGField3D<float>* fieldY = new CGField3D<float>(
        pivot,
        res,
        fieldSize,
        voxelSize,
        "test_field_y",
        filename);

    CGField3D<float>* fieldZ = new CGField3D<float>(
        pivot,
        res,
        fieldSize,
        voxelSize,
        "test_field_z",
        filename);

    CGVectorField3D<float>* velocityField = new CGVectorField3D<float>(
        fieldX,
        fieldY,
        fieldZ,
        "velocity_field"
        );

    std::cout << "---------" << std::endl;

    std::string posFileName = "../userInputData/3dfield_pos_data.txt";
    CGBuffer<glm::vec3>* posBuffer = dynamic_cast<CGBuffer<glm::vec3>*>(CGBuffer<float>::loadFromFile(posFileName));
    auto posData = posBuffer->getData();

    //glm::vec3 pos = glm::vec3(0.612341f, -0.661776f, 0.323438f);
    //auto vel = CodeGenerator::Field::GenericCode::SampleValueVectorField<float>(pos, velocityField->GetFieldRAWData());
    //std::cout << "vel: " << vel.x << ", " << vel.y << ", " << vel.z << std::endl;


    for (int i = 0; i < posBuffer->getSize(); ++i) {
        auto pos = posData[i];

        auto vel = CodeGenerator::Field::GenericCode::SampleValueVectorField<float>(pos, velocityField->GetFieldRAWData());
        std::cout << "vel: " << vel.x << ", " << vel.y << ", " << vel.z << std::endl;
    }

}

int main() {
    // ===== create input buffer
    // pbuffer for particle advect
    // load pbuffer from file

    int numPoints = 500;
    auto posBuffer = new CGBuffer<glm::vec3>("pos", numPoints, glm::vec3(0.f));

    // dynamically create and initialize vbuffer
    auto velBuffer = new CGBuffer<glm::vec3>("vel", numPoints, glm::vec3(0.f));

    // ===== create particle emitter
    // initilize data read from task json
    ParticleGenerator::RAWDesc desc;
    desc.direction = glm::vec3(0, 0, 1);
    desc.speed = 2;
    desc.size = glm::vec2(2, 2);
    desc.deltaX = 0.5;
    desc.center = glm::vec3(0, 0.f, 0.f);

    ParticleGenerator* particleGenerator = new ParticleGenerator(desc);
    particleGenerator->delegatePointBuffer(posBuffer);
    particleGenerator->delegatePointBuffer(velBuffer);

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

    glm::vec3 aabb_min = glm::vec3(-2.01958f, 2.38065f, 4.53478f);
    glm::vec3 aabb_max = glm::vec3(-1.01958f, 3.38065f, 5.53478f);
    CGAABB volumeVop_OpInput2_bbox = CGAABB(aabb_min, aabb_max);
    CGGeometry* volumeVop1_OpInput2 = new CGGeometry(volumeVop_OpInput2_bbox, nullptr, nullptr, nullptr);

    //CGGeometry particleAdvectVop_OpInput2; // a velocity field->velocityField

    // ===== load from task json
    int startFrame = 0;
    int endFrame = 1;
    float FPS = 24.f;
    int blockSize = 128;
    float TimeInc = 1.0 / FPS;

    for (int i = startFrame; i < endFrame; ++i) {
        //hard code var block
        float Time = i / FPS;
        float Frame = i;

#if CPU_VERSION
        // particle emitter
       //particleGenerator->generateParticlesCPU();

        // create velocity field
        CodeGenerator::volumevop1(
            volumeVop1_input5,
            volumeVop1_input4,
            volumeVop1_input3,
            volumeVop1_input2,
            volumeVop1_OpInput1,
            volumeVop1_OpInput2);

        //CodeGenerator::ParticleAdvect(
        //    posBuffer, 
        //    TimeInc, 
        //    particleAdvectVop_OpInput2);
#elif GPU_VERSION
        // particle emitter
        // first load pos buffer to device
        //posBuffer->malloc();
        //posBuffer->loadHostToDevice();
        //particleGenerator->generateParticlesGPU();

        CodeGenerator::CUDA::volumevop1(
            volumeVop1_input5,
            volumeVop1_input4,
            volumeVop1_input3,
            volumeVop1_input2,
            volumeVop1_OpInput1,
            volumeVop1_OpInput2);
        velocityField->LoadToHost();
#endif

        // save pos buffer as obj file
        std::string frame = std::to_string(Frame);
        std::string outputObjFilePathBase = "../userOutputData/vel_field_test";
        std::string velXFilePath = "../userOutputData/vel_field_test/velx_";
        std::string velYFilePath = "../userOutputData/vel_field_test/vely_";
        std::string velZFilePath = "../userOutputData/vel_field_test/velz_";

#if CPU_VERSION
        velXFilePath.append("cpu_");
        velYFilePath.append("cpu_");
        velZFilePath.append("cpu_");
#elif GPU_VERSION
        velXFilePath.append("gpu_");
        velYFilePath.append("gpu_");
        velZFilePath.append("gpu_");
#endif
        velXFilePath.append(frame);
        velXFilePath.append(".obj");
        velYFilePath.append(frame);
        velYFilePath.append(".obj");
        velZFilePath.append(frame);
        velZFilePath.append(".obj");

        //posBuffer->outputObj(outputObjFilePath);

        velocityField->m_FieldX->GetVoxelBufferPtr()->outputObj(velXFilePath);
        std::cout << velYFilePath << std::endl;
        velocityField->m_FieldY->GetVoxelBufferPtr()->outputObj(velYFilePath);
        std::cout << velZFilePath << std::endl;
        velocityField->m_FieldZ->GetVoxelBufferPtr()->outputObj(velZFilePath);
    }
}