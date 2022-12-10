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

//template<class T>
//class CGVectorField3D
//{
//public:
//  struct RAWData {
//
//  };
//
//  RAWData GetFieldRAWData();
//  RAWData GetFieldRAWDataDevice();
//};

struct AABB {
  glm::vec3 min;
  glm::vec3 max;

  AABB()
    : min(glm::vec3(0.f)), max(glm::vec3(0.f))
  {

  }

  AABB(glm::vec3 min, glm::vec3 max)
    : min(min), max(max)
  {

  }
};

class CGGeometry
{
public:
    AABB m_Bbox;
    CGBuffer<glm::vec3>* m_PosBuffer;
    CGBuffer<glm::vec3>* m_CdBuffer;
    CGVectorField3D<float>* velField;

    struct RAWData {
        AABB bbox;
        glm::vec3* posBuffer;
        glm::vec3* cdBuffer;
        CGVectorField3D<float>::RAWData velFieldRAWData;
    };

    RAWData GetGeometryRawData() {
        RAWData rawData;
        rawData.bbox = m_Bbox;

        rawData.posBuffer = (m_PosBuffer != nullptr) ? m_PosBuffer->getRawData() : nullptr;
        rawData.cdBuffer = (m_CdBuffer != nullptr) ? m_CdBuffer->getRawData() : nullptr;
        rawData.velFieldRAWData = velField->GetFieldRAWData();
    }

    RAWData GetGeometryRawDataDevice() {
        RAWData rawData;
        rawData.bbox = m_Bbox;

        rawData.posBuffer = (m_PosBuffer != nullptr) ? m_PosBuffer->getDevicePointer() : nullptr;
        rawData.cdBuffer = (m_CdBuffer != nullptr) ? m_CdBuffer->getDevicePointer() : nullptr;
        rawData.velFieldRAWData = velField->GetFieldRAWDataDevice();
    }

    CGGeometry() 
      : CGGeometry(AABB(glm::vec3(0, 0, 0), glm::vec3(0, 0, 0)), nullptr, nullptr, nullptr)
    {

    }

    CGGeometry(
    AABB bbox,
    CGBuffer<glm::vec3>* posBuffer,
    CGBuffer<glm::vec3>* cdBuffer,
    CGVectorField3D<float>* velField) 
      : m_Bbox(bbox), m_PosBuffer(posBuffer), m_CdBuffer(cdBuffer), velField(velField)
    {

    }
};
