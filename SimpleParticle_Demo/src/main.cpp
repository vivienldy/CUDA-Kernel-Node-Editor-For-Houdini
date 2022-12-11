#include "BaseOperation.h"
#include "CGGenerator.h"
#include "CGGeometry.h"

#include "simpleParticle.h"

#define CPU_VERSION 1
#define GPU_VERSION 0

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
    desc.direction = glm::vec3(0, 0, 1);
    desc.speed = 1;
    desc.size = glm::vec2(12, 12);
    desc.deltaX = 1.f;
    desc.center = glm::vec3(-1.f, 0.5f, 3.7f);

    ParticleGenerator* particleGenerator = new ParticleGenerator(desc);
    particleGenerator->delegatePointBuffer(posBuffer);
    particleGenerator->delegatePointBuffer(velBuffer);

     // ===== load from task json
    int startFrame = 0;
    int endFrame = 200;
    float FPS = 24.f;
    int blockSize = 128;
    float TimeInc = 1.0 / FPS;

#if GPU_VERSION
    // malloc() here
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
            posBuffer,
            velBuffer, 
            TimeInc);

#elif GPU_VERSION

#endif

        // save pos buffer as obj file
        std::string frame = std::to_string(Frame + 1);
        std::string posOutputObjFilePathBase = "../userOutputData/simple_particle_pos";

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