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

enum DataType {
	VEC3,
	FLOAT
};

class CGBufferBase {
protected:
	uint32_t m_bufferSize;
	std::string bufferNmae;
public:
	CGBufferBase()
		: m_bufferSize(0), bufferNmae("")
	{

	}
	uint32_t size() {
		return m_bufferSize;
	}

	void print() {
		std::cout << "This is a parent function." << std::endl;
	}

	void setSize(uint32_t size) {
		m_bufferSize = size;
	}

	//template<typename T>
	//void addValue(T) {};

	//template<typename T>
	//T* getRawData() {};

	//template<typename T>
	//T* getDevicePointer() {};

	//virtual bool malloc() = 0;

	//virtual bool loadHostToDevice() = 0;

	//virtual bool loadDeviceToHost() = 0;

	//virtual void clear() {};

	virtual ~CGBufferBase() {};
};

template<class T>
class CGBuffer: public CGBufferBase {
private:
	std::vector<T> m_data;
	T* m_devicePtr;

public:
	CGBuffer()
		: CGBufferBase(),
		m_devicePtr(nullptr)
	{
		std::cout << "CGBuffer created" << std::endl;
	}

	CGBuffer(int size, T value)
		: CGBufferBase(), 
		m_data(std::vector<T>(size, value)), m_devicePtr(nullptr)
	{
	}

	void addVlaue(T value) {
		m_data.push_back(value);
	}

	void print() {
		std::cout << sizeof(T) << std::endl;
		std::cout << "This is a derived function." << std::endl;
		std::cout << m_data[0][0] << ", "<<m_data[0][1] << ", " << m_data[0][2] << std::endl;
	}

	T* getRawData() {
		return m_data.data;
	}

	std::vector<T> getData() {
		return m_data;
	}

	T* getDevicePointer() {
		return m_devicePtr;
	}

	void resize(int size) {
		m_data.resize(size);
		m_bufferSize = size;
	}

	bool malloc() {
		cudaMalloc((void**)&m_devicePtr, m_bufferSize * sizeof(T));
		return true;
	}

	bool loadHostToDevice() {
		cudaMemcpy(m_devicePtr, m_data.data(), m_bufferSize * sizeof(T), cudaMemcpyHostToDevice);
		return true;
	}

	bool loadDeviceToHost() {
		cudaMemcpy(m_data.data(), m_devicePtr,  m_bufferSize * sizeof(T), cudaMemcpyDeviceToHost);
		return true;
	}

	bool loadFromDevice(T* devicePtr, int N=-1){
		if (N == -1) {
			N = m_bufferSize;
		}
		N = min(N, m_bufferSize);

		cudaMemcpy(m_devicePtr, devicePtr, N * sizeof(T), cudaMemcpyDeviceToDevice);
		return true;
	}

	bool loadToDevice(T* devicePtr, int N = -1) {
		if (N == -1) {
			N = m_bufferSize;
		}
		N = min(N, m_bufferSize);

		cudaMemcpy(devicePtr, m_devicePtr, N * sizeof(T), cudaMemcpyDeviceToDevice);
		return true;
	}

	static CGBufferBase* loadFromFile(std::string filename);

	void clear() {
		m_data.clear();
		m_bufferSize = 0;
		cudaFree(m_devicePtr);
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

	CGBufferBase *instance;
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

/**
* C main function.
*/
int main(int argc, char* argv[]) {
	std::string filename = "../geoData_example.txt";
	posBuffer = dynamic_cast<CGBuffer<glm::vec3>*>(CGBuffer<float>::loadFromFile(filename));
	posBuffer->resize();
	posBuffer->malloc();
	posBuffer->loadFromDevice(dev_pos);
}
