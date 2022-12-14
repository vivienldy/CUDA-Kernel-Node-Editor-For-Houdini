#include "BaseOperation.h"
#include "CGGenerator.h"
#include "CGGeometry.h"

#include "simpleParticle.h"

#define CPU_VERSION 0
#define GPU_VERSION 1

int main() {
    // ===== create input buffer
    // pbuffer for particle advect
    // load pbuffer from file

    int numPoints = 2;
    auto posBuffer = new CGBuffer<glm::vec3>("position", numPoints, glm::vec3(0.f));

    //std::string posFileName = "../userInputData/simple_particle_demo.txt";
    //CGBuffer<glm::vec3>* posBuffer = dynamic_cast<CGBuffer<glm::vec3>*>(CGBuffer<float>::loadFromFile(posFileName));

    // dynamically create and initialize buffers
    auto velBuffer = new CGBuffer<glm::vec3>("velocity", posBuffer->getSize(), glm::vec3(0.f));

    // ===== create particle emitter
    // initilize data read from task json
    ParticleGenerator::RAWDesc desc;
    desc.direction = glm::vec3(0, 1, 0);
    desc.speed = 1;
    desc.size = glm::vec2(1, 1);
    desc.deltaX = 0.1f;
    desc.center = glm::vec3(-1.f, 0.5f, 3.7f);

    ParticleGenerator* particleGenerator = new ParticleGenerator(desc);
    particleGenerator->delegatePointBuffer(posBuffer);
    particleGenerator->delegatePointBuffer(velBuffer);

    // custome parm
     CGGeometry* simpleParticleVop_OpInput1 = new CGGeometry(CGAABB(), nullptr, nullptr, nullptr);
     glm::vec3 simpleParticleVop_force = glm::vec3(0.f, 0.f, 1.f);

     // ===== load from task json
    int startFrame = 0;
    int endFrame = 240;
    float FPS = 24.f;
    int blockSize = 128;
    float TimeInc = 1.0 / FPS;

#if GPU_VERSION
    posBuffer->malloc();
    posBuffer->loadHostToDevice();
    velBuffer->malloc();
    velBuffer->loadHostToDevice();
# endif

    for (int i = startFrame; i < endFrame; ++i) {
        //hard code var block
        float Time = i / FPS;
        float Frame = i;

#if CPU_VERSION
        // particle emitter
        particleGenerator->generateParticlesCPU();

        // create velocity field
        CodeGenerator::simpleParticle(
            simpleParticleVop_force,
            TimeInc * Frame,
            posBuffer,
            velBuffer, 
            TimeInc,
            simpleParticleVop_OpInput1);

#elif GPU_VERSION
        particleGenerator->generateParticlesGPU();
        CodeGenerator::CUDA::simpleParticle(
            simpleParticleVop_force,
            TimeInc * Frame,
            posBuffer,
            velBuffer,
            TimeInc,
            simpleParticleVop_OpInput1);
        posBuffer->loadDeviceToHost();
#endif

        // save pos buffer as obj file
        std::string frame = std::to_string(Frame + 1);
        std::string posOutputObjFilePathBase = "../userOutputData/simple_particle_pos_";

#if CPU_VERSION
        posOutputObjFilePathBase.append("cpu_");
#elif GPU_VERSION
        posOutputObjFilePathBase.append("gpu_");
#endif
        posOutputObjFilePathBase.append(frame);
        posOutputObjFilePathBase.append(".obj");

        posBuffer->outputObj(posOutputObjFilePathBase);
        std::cout << "-------- frame: "  << frame << std::endl;
    }
}