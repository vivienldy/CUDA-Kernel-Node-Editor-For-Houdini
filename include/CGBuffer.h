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

#include <glm/glm.hpp>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

class CGBufferBase {
protected:
	uint32_t m_bufferSize;
	uint32_t m_appendSize;

	std::string bufferNmae;
	bool isMalloc;

	void* m_rawPtr;
	void* m_devicePtr;

public:

	CGBufferBase()
		: CGBufferBase(0, "Autonomy: " + std::to_string(0))
	{

	}

	CGBufferBase(int size)
		: CGBufferBase(size, "Autonomy: " + std::to_string(0)) {}

	CGBufferBase(int size, std::string name)
		: m_bufferSize(size), bufferNmae(name), isMalloc(false), m_rawPtr(nullptr), m_devicePtr(nullptr)
	{

	}

	virtual uint32_t typeSize() = 0;
	//virtual void setDevicePtr(void* rawPtr) = 0;

	void* getRawPtr() { return m_rawPtr; }

	void reallocationHost(int appendSize) {

		std::vector<glm::vec3> vec(appendSize, glm::vec3(0.f));
		std::vector<glm::vec3>* p = ((std::vector<glm::vec3>*)m_rawPtr);
		p->insert(p->end(), vec.begin(), vec.end());

		m_appendSize = appendSize;
		m_bufferSize += appendSize;
	}

	void reallocationDevice(int appendSize) {
		void* devicePtr;
		cudaMalloc((void**)&devicePtr, (m_bufferSize + appendSize) * typeSize());
		cudaMemcpy(devicePtr, m_devicePtr, m_bufferSize * typeSize(), cudaMemcpyDeviceToDevice);

		cudaFree(m_devicePtr);

		m_devicePtr = devicePtr;
		m_appendSize = appendSize;
		m_bufferSize += appendSize;
	}

	uint32_t getSize() {
		return m_bufferSize;
	}

	void* getDevicePtr() { return m_devicePtr; }

	uint32_t getAppendSize() {
		return m_appendSize;
	}

	void setSize(uint32_t size) {
		m_bufferSize = size;
	}

	bool hasMemory() {
		return isMalloc;
	}

	std::string getName() {
		return bufferNmae;
	}

	void setName(std::string name) {
		bufferNmae = name;
	}

	virtual ~CGBufferBase() {};
};

template<class T>
class CGBuffer : public CGBufferBase {
private:
	std::vector<T> m_data;

public:
	CGBuffer()
		: CGBufferBase()
	{
	}

	CGBuffer(int size, T value)
		: CGBuffer("", size, value)
	{
	}

	CGBuffer(std::string name, int size, T value)
		: CGBufferBase(size, name),
		m_data(std::vector<T>(size, value))
	{
		m_rawPtr = &m_data;
	}

	void initializeFromFile(std::string filename) {
		char* fname = (char*)filename.c_str();

		std::ifstream fp_in;
		fp_in.open(fname);
		if (!fp_in.is_open()) {
			std::cout << "Error reading from file - aborting!" << std::endl;
			throw;
		}

		std::string valueType = "";
		getline(fp_in, valueType, '|');

		std::vector<float> values;
		while (fp_in.good()) {
			std::string line;
			getline(fp_in, line, ',');
			if (!line.empty()) {
				values.push_back(std::stof(line));
			}
		}

		int offset = 1;
		m_data.clear();

		CGBufferBase* instance;
		if (valueType == "3f") {
			offset = 3;
			std::vector<glm::vec3>* vec = new std::vector<glm::vec3>();
			for (int i = 0; i < values.size(); i += offset) {
				vec->push_back(glm::vec3(values[i], values[i + 1], values[i + 2]));
			}

			m_rawPtr = vec;
		}
		else if (valueType == "1f") {
			offset = 1;
			std::vector<float>* vec = new std::vector<float>();
			m_rawPtr = &vec;
			for (int i = 0; i < values.size(); i += offset) {
				vec->push_back(values[i]);
			}

			m_rawPtr = vec;
		}

		m_data = *(std::vector<T>*)m_rawPtr;
		m_bufferSize = m_data.size();
	}

	void setToZero() {
		m_data = std::vector<T>(m_bufferSize, 0.f);
		m_rawPtr = &m_data;
	}

	uint32_t typeSize() { return sizeof(T); }

	void addVlaue(T value) {
		m_data.push_back(value);
		m_bufferSize = m_data.size();
	}

	void copy(CGBuffer<T>* object) {
		std::vector<T> vec = object->getData();
		m_data = std::vector<T>(vec.begin(), vec.end());

		m_bufferSize = object->getSize();

		if (object->hasMemory()) {
			this->malloc();
			this->loadFromDevice(object->getDevicePointer(), -1, false);
		}
	}

	CGBuffer<T>* clone() {
		CGBuffer<T>* object = new CGBuffer<T>();

		object->copy(this);

		return object;
	}

	T* getRawData() {
		return m_data.data();
	}

	std::vector<T> getData() {
		return m_data;
	}

	void setHostData(std::vector<T>& v) {
		m_data = std::vector<T>(v.begin(), v.end());
		m_rawPtr = &m_data;
	}

	T* getDevicePointer() {
		return (T*)m_devicePtr;
	}

	void setSize(int size) {
		m_data.resize(size);
		m_bufferSize = size;
	}

	bool malloc() {
		isMalloc = true;
		cudaMalloc((void**)&m_devicePtr, m_bufferSize * sizeof(T));

		//m_rawPtr = m_devicePtr;

		return true;
	}

	bool loadHostToDevice() {
		cudaMemcpy(m_devicePtr, m_data.data(), m_bufferSize * sizeof(T), cudaMemcpyHostToDevice);
		return true;
	}

	bool loadDeviceToHost() {
		m_data.resize(m_bufferSize);
		cudaMemcpy(m_data.data(), m_devicePtr, m_bufferSize * sizeof(T), cudaMemcpyDeviceToHost);
		return true;
	}

	bool loadFromDevice(T* devicePtr, int N = -1, bool toHost = true) {
		if (N == -1) {
			N = m_bufferSize;
		}
		N = std::min(N, (int)m_bufferSize);

		checkMalloc();
		cudaMemcpy(m_devicePtr, devicePtr, N * sizeof(T), cudaMemcpyDeviceToDevice);

		if (toHost) {
			this->loadDeviceToHost();
		}

		return true;
	}

	bool loadToDevice(T* devicePtr, int N = -1) {
		if (N == -1) {
			N = m_bufferSize;
		}
		N = min(N, m_bufferSize);

		checkMalloc();
		cudaMemcpy(devicePtr, m_devicePtr, N * sizeof(T), cudaMemcpyDeviceToDevice);
		return true;
	}

	void checkMalloc() {
		if (!isMalloc) {
			this->malloc();
		}
	}

	static CGBufferBase* loadFromFile(std::string filename);
	static CGBuffer<T>* loadFromFileCGBuffer(std::string filename) {
		return dynamic_cast<CGBuffer<T>*>(loadFromFile(filename));
	}

	void clear() {
		m_data.clear();
		m_bufferSize = 0;
		isMalloc = false;
		cudaFree(m_devicePtr);
	}

	void outputObj(std::string filename = "testing.obj") {
		std::string objContent = "g\n";

		std::string type = typeid(T).name(); //struct glm::tvec3<float,0>

		if (type == "struct glm::tvec3<float,0>") {
			std::vector<glm::vec3> ptr = *(std::vector<glm::vec3>*)m_rawPtr;

			for (int i = 0; i < m_bufferSize; i++) {
				glm::vec3 p = ptr[i];
				objContent += "v " + std::to_string(p[0]) + " " + std::to_string(p[1]) + " " + std::to_string(p[2]) + "\n";
			}
		}
		else if (type == "float") {
			std::vector<float> ptr = *(std::vector<float>*)m_rawPtr;

			for (int i = 0; i < m_bufferSize; i++) {
				float p = ptr[i];
				objContent += "v " + std::to_string(p) + "\n";
			}
		}

		std::ofstream myfile;
		myfile.open(filename);
		myfile << objContent;
		myfile.close();
	}

	~CGBuffer() {
		this->clear();
	}
};

template< typename T >
CGBufferBase* CGBuffer<T>::loadFromFile(std::string filename) {
	char* fname = (char*)filename.c_str();

	std::ifstream fp_in;
	fp_in.open(fname);
	if (!fp_in.is_open()) {
		std::cout << "Error reading from file - aborting!" << std::endl;
		throw;
	}

	std::string valueType = "";
	getline(fp_in, valueType, '|');

	std::vector<float> values;
	while (fp_in.good()) {
		std::string line;
		getline(fp_in, line, ',');
		if (!line.empty()) {
			values.push_back(std::stof(line));
		}
	}

	int offset = 1;

	CGBufferBase* instance;
	if (valueType == "3f") {
		CGBuffer<glm::vec3>* o1 = new CGBuffer<glm::vec3>();

		offset = 3;
		for (int i = 0; i < values.size(); i += offset) {
			if (valueType == "3f") {
				o1->addVlaue(glm::vec3(values[i], values[i + 1], values[i + 2]));
			}
		}

		o1->setSize(o1->getData().size());
		instance = dynamic_cast<CGBufferBase*>(o1);
		return instance;
	}

	return instance;
}

//int CGBufferBase::id = 0;

///**
//* C main function.
//*/
//int main(int argc, char* argv[]) {
//	std::string filename = "../geoData_example.txt";
//	CGBuffer<glm::vec3>* posBuffer = dynamic_cast<CGBuffer<glm::vec3>*>(CGBuffer<float>::loadFromFile(filename));
//  CGBuffer<glm::vec3>* velBuffer = new CGBuffer<glm::vec3>("buffer name", 10000, glm::vec3(1.f));
//	posBuffer->malloc();
//  posBuffer->loadHostToDevice();
// 
//  CGBuffer<glm::vec3>* posTemp = new CGBuffer<glm::vec3>();
//	posTemp->copy(posBuffer);
//  posTemp = posBuffer.clone();
//
//  posBuffer.outputObj();
//}