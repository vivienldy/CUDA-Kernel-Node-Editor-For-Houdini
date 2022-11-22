﻿#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <stdio.h>
#include <cmath>

#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "CGBuffer.h"


struct CGField3DInfo
{
    glm::vec3 Pivot; // center of the field
    glm::vec3 FieldSize; // size of the field
    glm::vec3 Resolution; // every side voxel count
    float VoxelSize; // size of one voxel
    float HalfVoxelSize;
    int NumVoxels; // voxel count of the field
};

template<class T>
class CGField3D
{
private:
    CGBuffer<T> m_VoxelBuffer; // save the actual data of the field
    CGField3DInfo m_FieldInfo; // save the field description data

    std::string m_Name;

public:
    // ---------- Pack Field Info for CPU and GPU
    // ShareCode will use this struct 
    // CPU and GPU data are different but use the same struct
    struct RAWData
    {
        bool IsValid; // is this field nullptr, for function pass a nullptr as field
        T* VoxelData; // m_voxelBuffer.getDevice()/getData()
        CGField3DInfo FieldInfo;
    };

    // get the CPU used rawdata
    RAWData GetFieldRawData() {
        RAWData rawData;
        rawData.IsValid = true;
        rawData.VoxelData = this->GetVoxelBufferPtr()->getRawData();
        rawData.FieldInfo = this->GetFieldInfo();
        return rawData;
    }

    // get the GPU used rawdata
    RAWData GetFieldRawDataDevice() {
        RAWData rawData;
        rawData.IsValid = true;
        // [TODO]  什么时候上GPU？如何确保devicepointer的确存在？
        rawData.VoxelData = this->GetVoxelBufferPtr()->getDevicePointer();  
        rawData.FieldInfo = this->GetFieldInfo();
        return rawData;
    }

    // ---------- Constructor
    CGField3D(
        glm::vec3 pivot, 
        glm::vec3 res, 
        glm::vec3 fieldSize, 
        float voxelSize, 
        std::string name,
        std::string voxelBufferFilePath = "") // if null initialize the voxelBuffer as zero
    {
        // initialize CGFieldInfo
        Init(pivot, res, fieldSize, voxelSize, voxelBufferFilePath);
        m_Name = name;
    }

    ~CGField3D() {

    }

    // ---------- Initializing Field
    // initialize Field
    void Init(
        glm::vec3 pivot, 
        glm::vec3 res, 
        glm::vec3 fieldSize,
        float voxelSize, 
        std::string name,
        std::string voxelBufferFilePath = "") 
    {
        // initialize m_FieldInfo
        SetPivot(pivot);
        SetFieldResolution(res);
        SetFieldSize(fieldSize);
        SetVoxelSize(voxelSize);

        // initialize m_VoxelBuffer
        m_VoxelBuffer.setName(name + ".voxels");
        if (voxelBufferFilePath != "") {
            // [TODO] 问题，现在loadFromFile返回的是个pointer???
            // [TODO] 或许需要一个CGBuffer.readFromFile()直接根据file数据初始化m_data???
            // 因为现在已经有了m_VoxelBuffer这个实例，只是没有初始化m_data
            // CGBuffer还需要支持非vector类型 比如1f 纯float类型的数据输入
            m_VoxelBuffer = *(dynamic_cast<CGBuffer<T>*>(CGBuffer<T>::loadFromFile(voxelBufferFilePath)));
        }
        else {
            // [TODO] 非常需要验证是否正确是否可行。。
           // 需要CGBuffer有一个setToZero()的function
           // 同时，需要cgbuffer不仅有一个size，还需要有一个capacity，capacity是一个很大的定值，size是真实大小
           // 当setToZero的时候，是按照capacity set value to zero
            // m_VoxelBuffer.SetToZero();
        }
    }

    void SetPivot(glm::vec3 pivot) {
        m_FieldInfo.Pivot = pivot;
    }

    void SetFieldResolution(glm::vec3 res) {
        m_FieldInfo.Resolution = res;
        m_FieldInfo.NumVoxels = res.x * res.y * res.z;
    }

    void SetFieldSize(glm::vec3 size) {
        m_FieldInfo.FieldSize = size;
    }

    void SetVoxelSize(float size) {
        m_FieldInfo.VoxelSize = size;
        m_FieldInfo.HalfVoxelSize = size * 0.5f;
    }

    // ---------- helper function
    // if m_VoxelBuffer has device pointer
    bool HasDeviceData() {

    }

    void SetToZero()
    {
        std::memset(m_VoxelBuffer.getRawData(), 0, m_VoxelBuffer.getSize() * sizeof(T));
    }

    CGBuffer<T>* GetVoxelBufferPtr()
    {
        return &m_VoxelBuffer;
    }

    CGField3DInfo GetFieldInfo()
    {
        return this->m_FieldInfo;
    }

};

void testFieldMain() {
    glm::vec3 pivot = glm::vec3(0.f);
    glm::vec3 res = glm::vec3(3, 3, 3);
    glm::vec3 fieldSize = glm::vec3(3.f, 3.f, 3.f);
    float voxelSize = 1.f;
    CGField3D<float> field = CGField3D<float>(
        pivot,
        res,
        fieldSize,
        voxelSize,
        "test_field");
}

