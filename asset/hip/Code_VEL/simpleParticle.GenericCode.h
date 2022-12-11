#include <glm/glm.hpp>
#include "BaseOperation.h"

namespace CodeGenerator 
{ 
	namespace GenericCode 
 	{ 
        __host__ __device__ inline void simpleParticle(glm::vec3* geo1_simpleParticle_geometryvopglobal1_Pbuffer, glm::vec3* geo1_simpleParticle_geometryvopglobal1_vbuffer, float geo1_simpleParticle_geometryvopglobal1_TimeInc, int idx)
        {
            // Data Load 
            // Geometry Global Input
glm::vec3 geo1_simpleParticle_geometryvopglobal1_P = geo1_simpleParticle_geometryvopglobal1_Pbuffer[idx];
glm::vec3 geo1_simpleParticle_geometryvopglobal1_v = geo1_simpleParticle_geometryvopglobal1_vbuffer[idx];


            // Compute graph
            
 // Generate by curlnoise1
glm::vec3 geo1_simpleParticle_curlnoise1_noise = curlnoise(char(pnoise), geo1_simpleParticle_geometryvopglobal1_P, glm::vec3(1.0f,1.0f,1.0f), glm::vec3(0.0f,0.0f,0.0f), float(1.0f), float(0.5f), float(1.0f), int(3), float(0.0001f), float(1.0f), float(1.0f), glm::vec3(0.0f,0.0f,0.0f), char(), int(0));

 // Generate by multiply1
glm::vec3 geo1_simpleParticle_multiply1_product = geo1_simpleParticle_curlnoise1_noise * geo1_simpleParticle_geometryvopglobal1_TimeInc;

 // Generate by add1
glm::vec3 geo1_simpleParticle_add1_sum = geo1_simpleParticle_multiply1_product + geo1_simpleParticle_geometryvopglobal1_v;

 // Generate by multiply2
glm::vec3 geo1_simpleParticle_multiply2_product = geo1_simpleParticle_add1_sum * geo1_simpleParticle_geometryvopglobal1_TimeInc;

 // Generate by add2
glm::vec3 geo1_simpleParticle_add2_sum = geo1_simpleParticle_multiply2_product + geo1_simpleParticle_geometryvopglobal1_P;


            // Write back
            glm::vec3 global_output_geo1_simpleParticle_geometryvopoutput1_P = geo1_simpleParticle_add2_sum;
geo1_simpleParticle_geometryvopglobal1_Pbuffer[idx] = global_output_geo1_simpleParticle_geometryvopoutput1_P;


        }
	} 
} 