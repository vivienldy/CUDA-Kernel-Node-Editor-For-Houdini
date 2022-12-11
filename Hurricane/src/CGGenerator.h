#pragma once

#include <cuda.h>

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <unordered_set>

#include <glm/glm.hpp>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include "CGBuffer.h"

class ParticleGenerator
{
public:
	struct RAWDesc {
		glm::vec3 direction;
		glm::vec2 size;
		glm::vec3 center;

		float speed;
		float deltaX;

		RAWDesc()
			: direction(glm::vec3(1.f, 0, 0)), size(glm::vec2(0, 0)), center(glm::vec3(0, 0, 0)),
			speed(1), deltaX(1) {}

		RAWDesc(glm::vec3 direction, glm::vec2 size, glm::vec3 center, float speed, float deltaX)
			: direction(direction), size(size), center(center),
			speed(speed), deltaX(deltaX) {}
	};

	ParticleGenerator()
		: ParticleGenerator("Autonomy", RAWDesc()) {}

	ParticleGenerator(RAWDesc desc)
		: ParticleGenerator("Autonomy", desc) {}

	ParticleGenerator(std::string name)
		: ParticleGenerator(name, RAWDesc()) {}

	ParticleGenerator(std::string name, RAWDesc desc)
		: name(name), m_Desc(desc) {}

	std::string getName() { return name; }

	void delegatePointBuffer(CGBufferBase* pointer) {
		m_pointBuffers.insert(pointer);
		m_pointBuffers.insert(pointer);
	}

	void generateParticlesCPU() {
		this->generateParticlesCPU(this->m_Desc);
	}

	void generateParticlesCPU(RAWDesc desc) {
		int appendSize = int(desc.size.x / desc.deltaX + 1) * int(desc.size.y / desc.deltaX + 1) * int(desc.speed / desc.deltaX);
		for (auto& pointBuffer : this->m_pointBuffers) {
			auto bufferName = pointBuffer->getName();
			auto buffer = pointBuffer;

			if (bufferName == "velocity") {
				buffer->reallocationHost(appendSize);
				std::vector<glm::vec3> velocity(appendSize, desc.direction * desc.speed);

				std::vector<glm::vec3>* ptr = (std::vector<glm::vec3>*)buffer->getRawPtr();
				std::copy(velocity.begin(), velocity.end(), ptr->begin() + (((int)buffer->getSize()) - appendSize));
			}
			else if (bufferName == "position") {
				buffer->reallocationHost(appendSize);

				std::vector<glm::vec3> velocity;
				for (int k = 0; k < desc.speed; k++) {
					for (int i = 0; i <= desc.size.y / desc.deltaX; i++) {
						for (int j = 0; j <= desc.size.x / desc.deltaX; j++) {
							velocity.push_back(desc.center - glm::vec3(desc.size.x, 0, desc.size.y) / 2.f + glm::vec3(j * desc.deltaX, 0.f, i * desc.deltaX));
							velocity.back() += k * desc.deltaX * desc.direction;
						}
					}
				}

				std::vector<glm::vec3>* ptr = (std::vector<glm::vec3>*)buffer->getRawPtr();
				std::copy(velocity.begin(), velocity.end(), ptr->begin() + (((int)buffer->getSize()) - appendSize));
			}
			else if (bufferName == "color") {
				buffer->reallocationHost(appendSize);

				std::vector<glm::vec3> colors(appendSize, glm::vec3(0.f));

				std::vector<glm::vec3>* ptr = (std::vector<glm::vec3>*)buffer->getRawPtr();
				std::copy(colors.begin(), colors.end(), ptr->begin() + (((int)buffer->getSize()) - appendSize));
			}
			else if (bufferName == "age") {
				buffer->reallocationHost(appendSize, false);

				std::vector<float> ages(appendSize, 0.f);

				std::vector<float>* ptr = (std::vector<float>*)buffer->getRawPtr();
				ptr->insert(ptr->begin() + (((int)buffer->getSize()) - appendSize), ages.begin(), ages.end());
			}
		}
	}

	void generateParticlesGPU() {
		this->generateParticlesGPU(this->m_Desc);
	}

	void generateParticlesGPU(RAWDesc desc) {
		int appendSize = int(desc.size.x / desc.deltaX + 1) * int(desc.size.y / desc.deltaX + 1) * int(desc.speed / desc.deltaX);

		for (auto& pointBuffer : this->m_pointBuffers) {
			auto bufferName = pointBuffer->getName();
			auto buffer = pointBuffer;

			if (bufferName == "velocity") {
				buffer->reallocationDevice(appendSize);
				std::vector<glm::vec3> velocity(appendSize, desc.direction * desc.speed);

				cudaMemcpy((glm::vec3*)buffer->getDevicePtr() + ((int)buffer->getSize() - appendSize), velocity.data(), appendSize * buffer->typeSize(), cudaMemcpyHostToDevice);
			}
			else if (bufferName == "position") {
				buffer->reallocationDevice(appendSize);

				std::vector<glm::vec3> velocity;
				for (int k = 0; k < desc.speed / desc.deltaX; k++) {
					for (int i = 0; i <= desc.size.y / desc.deltaX; i++) {
						for (int j = 0; j <= desc.size.x / desc.deltaX; j++) {
							velocity.push_back(desc.center - glm::vec3(desc.size, 0) / 2.f + glm::vec3(j * desc.deltaX, i * desc.deltaX, 0));
							velocity.back() += k * desc.deltaX * desc.direction;
						}
					}
				}
				cudaMemcpy(((glm::vec3*)buffer->getDevicePtr()) + ((int)buffer->getSize() - appendSize), velocity.data(), appendSize * buffer->typeSize(), cudaMemcpyHostToDevice);
			}
			else if (bufferName == "color") {
				buffer->reallocationDevice(appendSize);

				std::vector<glm::vec3> colors(appendSize, glm::vec3(0.f));

				cudaMemcpy(((glm::vec3*)buffer->getDevicePtr()) + ((int)buffer->getSize() - appendSize), colors.data(), appendSize * buffer->typeSize(), cudaMemcpyHostToDevice);
			}
			else if (bufferName == "age") {
				buffer->reallocationDevice(appendSize);

				std::vector<float> ages(appendSize, 0.f);

				cudaMemcpy(((glm::vec3*)buffer->getDevicePtr()) + ((int)buffer->getSize() - appendSize), ages.data(), appendSize * buffer->typeSize(), cudaMemcpyHostToDevice);
			}
		}
	}

private:
	std::string name;

	std::unordered_set<CGBufferBase*> m_pointBuffers;

	RAWDesc m_Desc;
};
//
//#define CPU 0
//#define GPU 1
//
//void main() {
//	CGBuffer<glm::vec3>* vbuffer = new CGBuffer<glm::vec3>("velocity", 2, glm::vec3(1.f));
//	CGBuffer<glm::vec3>* pbuffer = new CGBuffer<glm::vec3>("position", 2, glm::vec3(3.f, 4, 5));
//
//	vbuffer->malloc();
//	vbuffer->loadHostToDevice();
//
//	pbuffer->malloc();
//	pbuffer->loadHostToDevice();
//
//	ParticleGenerator::RAWDesc desc;
//	desc.direction = glm::vec3(0, 0, 1);
//	desc.speed = 2;
//	desc.size = glm::vec2(2, 2);
//
//	ParticleGenerator* pg = new ParticleGenerator(desc);
//	pg->delegatePointBuffer(vbuffer);
//	pg->delegatePointBuffer(pbuffer);
//
//#if CPU
//	pg->generateParticlesCPU();
//#elif GPU
//	pg->generateParticlesGPU();
//#endif
//
//	{
//#if GPU
//		pbuffer->loadDeviceToHost();
//		vbuffer->loadDeviceToHost();
//#endif
//
//		glm::vec3* pos = (pbuffer->getRawData());
//		glm::vec3* vec = (vbuffer->getRawData());
//		int size = pbuffer->getSize();
//		for (int i = 0; i < size; i++) {
//			glm::vec3 p = pos[i];
//			glm::vec3 v = vec[i];
//			std::cout << p.x << " " << p.y << " " << p.z << "  V: " << v.x << " " << v.y << " " << v.z << std::endl;
//		}
//	}
//
//	std::cout << "================= Next Frame =================" << std::endl;
//
//#if GPU
//	CUDA::updatePosition(pbuffer->getSize(), pbuffer->getDevicePointer(), vbuffer->getDevicePointer());
//#elif CPU
//	glm::vec3* pos = (pbuffer->getRawData());
//	glm::vec3* vec = (vbuffer->getRawData());
//	int size = pbuffer->getSize();
//	for (int i = 0; i < size; i++) {
//		pos[i] += vec[i];
//	}
//
//#endif
//
//	{
//#if GPU
//		pbuffer->loadDeviceToHost();
//		vbuffer->loadDeviceToHost();
//#endif
//
//		glm::vec3* pos = (pbuffer->getRawData());
//		glm::vec3* vec = (vbuffer->getRawData());
//		int size = pbuffer->getSize();
//		for (int i = 0; i < size; i++) {
//			glm::vec3 p = pos[i];
//			glm::vec3 v = vec[i];
//			std::cout << p.x << " " << p.y << " " << p.z << "  V: " << v.x << " " << v.y << " " << v.z << std::endl;
//		}
//	}
//}