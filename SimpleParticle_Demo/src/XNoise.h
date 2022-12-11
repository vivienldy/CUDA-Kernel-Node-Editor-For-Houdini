#ifndef __X_NOISE_H__
#define __X_NOISE_H__


#include "CGBuffer.h"

#define TIME_SCALE_CONSTANT 30
#define DEFAULT_NOISE_SIZE  256



struct CurlNoiseParam
{

	CG_SHARE_FUNC CurlNoiseParam()
	{
		// AlphaCore::ProceduralContent::AxNoiseType::kPerlinNoise;
		StepSize = 0.001f;
		Threshold = 0.001f;
		NoiseData = nullptr;
		Frequency = glm::vec3(1.0f, 1.0f, 1.0f);
		Offset = glm::vec4();
		Turbulence = 3;
		Amplitude = 1.0f;
		Roughness = 0.5f;
		Attenuation = 1.0;
	}
	CG_SHARE_FUNC ~CurlNoiseParam()
	{

	}
	glm::vec3 Frequency;
	glm::vec4 Offset;
	int Turbulence;//fbm
	float Amplitude;
	float Roughness;
	float Attenuation;
	float TimeOffset;
 	float StepSize;
	void* CollisionSDF;
	void* NoiseData;
	float Threshold;
};



static int permutation[DEFAULT_NOISE_SIZE] = { 151,160,137,91,90,15,								// Hash lookup table as defined by Ken Perlin.  This is a randomly
	131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,    // arranged array of all numbers from 0-255 inclusive.
	190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
	88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
	77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
	102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
	135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
	5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
	223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
	129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
	251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
	49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
	138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
};


namespace ShareCode
{
	template<typename T>
	CG_SHARE_FUNC T grad(int hash, T x, T y, T z) {
		int h = hash & 15;									// Take the hashed value and take the first 4 bits of it (15 == 0b1111)
		T u = h < 8 /* 0b1000 */ ? x : y;					// If the most significant bit (MSB) of the hash is 0 then set u = x.  Otherwise y.
		T v;												// In Ken Perlin's original implementation this was another conditional operator (?:).  I
															// expanded it for readability.
		if (h < 4 /* 0b0100 */)								// If the first and second significant bits are 0 set v = y
			v = y;
		else if (h == 12 /* 0b1100 */ || h == 14 /* 0b1110*/)// If the first and second significant bits are 1 set v = x
			v = x;
		else 												// If the first and second significant bits are not equal (0/1, 1/0) set v = z
			v = z;
		return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v); // Use the last 2 bits to decide if u and v are positive or negative.  Then return their addition.
	}
}


void CurlNoise4DVector(
	CurlNoiseParam curlNoiseParam,
	CGBufferV3* posBuf,
	CGBufferV3* outVecBuf,
	void* noiseRawData,
	float time,
	float dt);

namespace CUDA
{
	void CurlNoise4DVector(
		CurlNoiseParam curlNoiseParam,
		CGBufferV3* posBuf,
		CGBufferV3* outVecBuf,
		void* noiseRawData,
		float time,
		float dt,
		int blockSize = 128);
}

class XNoiseDataManager
{
public:
	XNoiseDataManager();
	~XNoiseDataManager();
	static XNoiseDataManager* GetInstance();
	void* GetXNoiseData();
	void* GetXNoiseDataDevice();

private:
	static XNoiseDataManager* m_Instance;

	CGBufferI*	m_RandomIndexBuffer;
	CGBufferF*	m_RandomGradBuffer;
	CGBufferRaw* m_XNoideRawData;

};



static inline CurlNoiseParam MakeDefaultCurlNoise()
{
	CurlNoiseParam noise;
	noise.Frequency = MakeVector3(1.0f, 1.0f, 1.0f);
	noise.Offset = MakeVector4();
	noise.Turbulence = 3;
	noise.Amplitude = 1.0;
	noise.Roughness = 0.5f;
	noise.Attenuation = 1.0f;
	noise.TimeOffset;

	noise.StepSize = 0.001f;
	noise.Threshold = 0.001f;
	noise.NoiseData = XNoiseDataManager::GetInstance()->GetXNoiseData();

	return noise;
}

static inline CurlNoiseParam MakeDefaultCurlNoiseDevice()
{
	CurlNoiseParam noise = MakeDefaultCurlNoise();
	noise.NoiseData = XNoiseDataManager::GetInstance()->GetXNoiseDataDevice();
	return noise;
}


#endif