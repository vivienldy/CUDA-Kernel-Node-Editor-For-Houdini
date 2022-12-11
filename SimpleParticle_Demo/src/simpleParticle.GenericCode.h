#include <glm/glm.hpp>
#include "BaseOperation.h"

namespace CodeGenerator
{
	namespace GenericCode
	{
		__host__ __device__ inline void simpleParticle(glm::vec3 geo1_simpleParticle_parm2_force, float geo1_simpleParticle_parm1_time, glm::vec3* geo1_simpleParticle_geometryvopglobal1_Pbuffer, glm::vec3* geo1_simpleParticle_geometryvopglobal1_vbuffer, float geo1_simpleParticle_geometryvopglobal1_TimeInc, CGGeometry::RAWData geo1_simpleParticle_geometryvopglobal1_OpInput1, int idx)
		{
			// Data Load 
			// Geometry Global Input
			glm::vec3 geo1_simpleParticle_geometryvopglobal1_P = geo1_simpleParticle_geometryvopglobal1_Pbuffer[idx];
			glm::vec3 geo1_simpleParticle_geometryvopglobal1_v = geo1_simpleParticle_geometryvopglobal1_vbuffer[idx];


			// Compute graph

 // Generate by xnoise1
			glm::vec3 geo1_simpleParticle_xnoise1_noise = xnoise(
				geo1_simpleParticle_geometryvopglobal1_OpInput1, 
				geo1_simpleParticle_geometryvopglobal1_P, 
				glm::vec3(0.1f, 0.1f, 0.1f), 
				glm::vec3(0.0f, 0.0f, 0.0f), 
				glm::vec3(0.0f, 0.0f, 0.0f), 
				int(3),
				int(0), 
				float(0.05f),
				float(1.0f), 
				float(1.0f),
				float(0.001f), 
				geo1_simpleParticle_parm1_time);

			// Generate by multiply3
			glm::vec3 geo1_simpleParticle_multiply3_product = geo1_simpleParticle_xnoise1_noise * geo1_simpleParticle_geometryvopglobal1_TimeInc;

			// Generate by add1
			glm::vec3 geo1_simpleParticle_add1_sum = 
				geo1_simpleParticle_geometryvopglobal1_v +
				geo1_simpleParticle_multiply3_product + 
				geo1_simpleParticle_parm2_force * geo1_simpleParticle_geometryvopglobal1_TimeInc;

			// Generate by multiply2
			glm::vec3 geo1_simpleParticle_multiply2_product = geo1_simpleParticle_add1_sum * geo1_simpleParticle_geometryvopglobal1_TimeInc * 10.f;

			// Generate by add2
			glm::vec3 geo1_simpleParticle_add2_sum = geo1_simpleParticle_multiply2_product + geo1_simpleParticle_geometryvopglobal1_P;


			// Write back
			glm::vec3 global_output_geo1_simpleParticle_geometryvopoutput1_P = geo1_simpleParticle_add2_sum;
			geo1_simpleParticle_geometryvopglobal1_Pbuffer[idx] = global_output_geo1_simpleParticle_geometryvopoutput1_P;

			glm::vec3 global_output_geo1_simpleParticle_geometryvopoutput1_v = geo1_simpleParticle_add1_sum;
			geo1_simpleParticle_geometryvopglobal1_vbuffer[idx] = global_output_geo1_simpleParticle_geometryvopoutput1_v;


		}
	}
}