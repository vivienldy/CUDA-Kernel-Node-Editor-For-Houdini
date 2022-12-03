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
		: name(name), m_Desc(desc)	{}

	std::string getName() { return name; }

	void delegatePointBuffer(CGBufferBase* pointer) {
		if (m_pointBuffers.find(pointer->getName()) != m_pointBuffers.end()) {
			return;
		}

		m_pointBuffers[pointer->getName()] = pointer;
	}

	void generateParticlesCPU() {
		this->generateParticlesCPU(this->m_Desc);
	}

	void generateParticlesCPU(RAWDesc desc) {
		int appendSize = (desc.size.x / desc.deltaX + 1) * (desc.size.y / desc.deltaX + 1) * (desc.speed);

		for (auto& pointBuffer : this->m_pointBuffers) {
			auto bufferName = pointBuffer.first;
			auto buffer = pointBuffer.second;

			if (bufferName == "velocity") {
				buffer->reallocationHost(appendSize);
				std::vector<glm::vec3> velocity(appendSize, desc.direction * desc.speed);
				std::vector<glm::vec3>* ptr = (std::vector<glm::vec3>*)buffer->getRawPtr();
				ptr->insert(ptr->begin() + (((int)buffer->getSize()) - appendSize), velocity.begin(), velocity.end());
			}
			else if (bufferName == "position") {
				buffer->reallocationHost(appendSize);

				std::vector<glm::vec3> velocity;
				for (int k = 0; k <= desc.speed; k++) {
					for (int i = 0; i <= desc.size.y / desc.deltaX; i++) {
						for (int j = 0; j <= desc.size.x / desc.deltaX; j++) {
							velocity.push_back(desc.center - glm::vec3(desc.size, 0) / 2.f + glm::vec3(j * desc.deltaX, i * desc.deltaX, 0));
							velocity.back() += k / desc.deltaX * desc.direction;
						} 
					}
				}

				std::vector<glm::vec3>* ptr = (std::vector<glm::vec3>*)buffer->getRawPtr();
				ptr->insert(ptr->begin() + (((int)buffer->getSize()) - appendSize), velocity.begin(), velocity.end());
			}
		}
	}

	void generateParticlesGPU() {
		this->generateParticlesGPU(this->m_Desc);
	}

	void generateParticlesGPU(RAWDesc desc) {
		int appendSize = (desc.size.x / desc.deltaX + 1) * (desc.size.y / desc.deltaX + 1) * (desc.speed);

		for (auto &pointBuffer: this->m_pointBuffers) {
			auto bufferName = pointBuffer.first;
			auto buffer = pointBuffer.second;

			if (bufferName == "velocity") {
				buffer->reallocationDevice(appendSize);
				std::vector<glm::vec3> velocity(appendSize, desc.direction * desc.speed);
				cudaMemcpy((glm::vec3*)buffer->getDevicePtr() + ((int)buffer->getSize() - appendSize), velocity.data(), appendSize * buffer->typeSize(), cudaMemcpyHostToDevice);
			}
			else if (bufferName == "position") {
				buffer->reallocationDevice(appendSize);

				std::vector<glm::vec3> velocity;
				for (int k = 0; k <= desc.speed; k++) {
					for (int i = 0; i <= desc.size.y / desc.deltaX; i++) {
						for (int j = 0; j <= desc.size.x / desc.deltaX; j++) {
							velocity.push_back(desc.center - glm::vec3(desc.size, 0) / 2.f + glm::vec3(j * desc.deltaX, i * desc.deltaX, 0));
							velocity.back() += k / desc.deltaX * desc.direction;
						}
					}
				}
				cudaMemcpy(((glm::vec3*)buffer->getDevicePtr()) + ((int)buffer->getSize() - appendSize), velocity.data(), appendSize * buffer->typeSize(), cudaMemcpyHostToDevice);
			}
		}
	}

private:
	std::string name;

	std::unordered_map<std::string, CGBufferBase*> m_pointBuffers;

	RAWDesc m_Desc;
};
