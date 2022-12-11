#include <glm/glm.hpp>
#include "BaseOperation.h"
#include "XNoise.h"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include "CGBuffer.h"

namespace CodeGenerator
{
	namespace GenericCode
	{
		__host__ __device__ inline glm::vec3 curlnoise(char* (type), glm::vec3 pos, glm::vec3 freq, glm::vec3 offset, float amp, float rough, float atten, int turb, float h, float radius, float distance, glm::vec3 nml, char* (geo), int bounce)
		{
			CGBufferV3 *cg_pos = new CGBufferV3("position", 1, pos);//CGBufferV3(sizeof(pos), pos);

			CurlNoiseParam noiseParam;
			noiseParam.Amplitude = 10.f;
			noiseParam.Attenuation = atten;
			noiseParam.Frequency = freq * 0.5f;
			noiseParam.Offset = glm::vec4(offset, 0.f);
			noiseParam.Roughness = rough;
			noiseParam.Turbulence = turb;
			noiseParam.StepSize = h;

			CurlNoise4DVector(
				noiseParam,
				cg_pos,
				cg_pos,
				XNoiseDataManager::GetInstance()->GetXNoiseData(), 0.0f, 0.0416);

			return glm::vec3(*cg_pos->getRawData());
		}

		__host__ __device__ inline void simpleParticle(glm::vec3* geo1_simpleParticle_geometryvopglobal1_Pbuffer, glm::vec3* geo1_simpleParticle_geometryvopglobal1_vbuffer, float geo1_simpleParticle_geometryvopglobal1_TimeInc, int idx)
		{
			// Data Load 
			// Geometry Global Input
			glm::vec3 geo1_simpleParticle_geometryvopglobal1_P = geo1_simpleParticle_geometryvopglobal1_Pbuffer[idx];
			glm::vec3 geo1_simpleParticle_geometryvopglobal1_v = geo1_simpleParticle_geometryvopglobal1_vbuffer[idx];


			// Compute graph

 // Generate by curlnoise1
			glm::vec3 geo1_simpleParticle_curlnoise1_noise = curlnoise(/*char(pnoise),*/ geo1_simpleParticle_geometryvopglobal1_P/*, glm::vec3(1.0f,1.0f,1.0f), glm::vec3(0.0f,0.0f,0.0f), float(1.0f), float(0.5f), float(1.0f), int(3), float(0.0001f), float(1.0f), float(1.0f), glm::vec3(0.0f,0.0f,0.0f), char(), int(0)*/);

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