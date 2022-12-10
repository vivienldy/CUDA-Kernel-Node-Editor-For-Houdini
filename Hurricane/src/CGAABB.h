#pragma once

#include "CGBuffer.h"

struct CGAABB {
  glm::vec3 min;
  glm::vec3 max;

  CGAABB()
    : min(glm::vec3(0.f)), max(glm::vec3(0.f))
  {

  }

  CGAABB(glm::vec3 min, glm::vec3 max)
    : min(min), max(max)
  {

  }
};
