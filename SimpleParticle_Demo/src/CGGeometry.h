#pragma once

#include "CGBuffer.h"
#include "XNoise.h"

class CGGeometry
{
public:
    CGAABB m_Bbox;
    CGBuffer<glm::vec3>* m_PosBuffer;
    CGBuffer<glm::vec3>* m_CdBuffer;
    CGVectorField3D<float>* m_VelField;
    void* noiseData;

    struct RAWData {
        CGAABB bbox;
        glm::vec3* posBuffer;
        glm::vec3* cdBuffer;
        CGVectorField3D<float>::RAWData velFieldRAWData;
        void* noiseData;
    };

    RAWData GetGeometryRawData() {
        RAWData rawData;
        rawData.bbox = m_Bbox;

        rawData.posBuffer = (m_PosBuffer != nullptr) ? m_PosBuffer->getRawData() : nullptr;
        rawData.cdBuffer = (m_CdBuffer != nullptr) ? m_CdBuffer->getRawData() : nullptr;
        rawData.velFieldRAWData = (m_VelField != nullptr) ? m_VelField->GetFieldRAWData() : CGVectorField3D<float>::RAWData();
        rawData.noiseData = XNoiseDataManager::GetInstance()->GetXNoiseData();
        return rawData;
    }

    RAWData GetGeometryRawDataDevice() {
        RAWData rawData;
        rawData.bbox = m_Bbox;

        rawData.posBuffer = (m_PosBuffer != nullptr) ? m_PosBuffer->getDevicePointer() : nullptr;
        rawData.cdBuffer = (m_CdBuffer != nullptr) ? m_CdBuffer->getDevicePointer() : nullptr;
        rawData.velFieldRAWData = (m_VelField != nullptr) ? m_VelField->GetFieldRAWDataDevice() : CGVectorField3D<float>::RAWData();
        rawData.noiseData = XNoiseDataManager::GetInstance()->GetXNoiseDataDevice();
        return rawData;
    }

    CGGeometry() 
      : CGGeometry(CGAABB(glm::vec3(0, 0, 0), glm::vec3(0, 0, 0)), nullptr, nullptr, nullptr)
    {

    }

    CGGeometry(
    CGAABB bbox,
    CGBuffer<glm::vec3>* posBuffer,
    CGBuffer<glm::vec3>* cdBuffer,
    CGVectorField3D<float>* velField) 
      : m_Bbox(bbox), m_PosBuffer(posBuffer), m_CdBuffer(cdBuffer), m_VelField(velField), noiseData(nullptr)
    {

    }

    // ---------- CUDA fucntion
    bool DeviceMalloc()
    {
        if (m_PosBuffer) {
            m_PosBuffer->malloc();
        }
        if (m_CdBuffer) {
            m_CdBuffer->malloc();
        }
        if (m_VelField) {
           m_VelField->DeviceMalloc();
        }
        return true;
    }

    void LoadToHost()
    {
        if (m_PosBuffer) {
            m_PosBuffer->loadDeviceToHost();
        }
        if (m_CdBuffer) {
            m_CdBuffer->loadDeviceToHost();
        }
        if (m_VelField) {
            m_VelField->LoadToHost();
        }
    }

    void LoadToDevice()
    {
        if (m_PosBuffer) {
            m_PosBuffer->loadHostToDevice();
        }
        if (m_CdBuffer) {
            m_CdBuffer->loadHostToDevice();
        }
        if (m_VelField) {
           m_VelField->LoadToDevice();
        }
    }
};
