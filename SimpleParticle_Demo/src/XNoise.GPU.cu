
#include "XNoise.h"

namespace CodeGenerator {
	namespace CUDAKernel {


		__global__ void curlNoise4DVector(CurlNoiseParam curlNoiseParam,
			glm::vec3* posRaw,
			glm::vec3* outVecRaw,
			void* noiseData,
			float time,
			float dt,
			int maxThreads);

	}//@namespace end of : ProceduralContect
}

#include "XNoise.GenericCode.h"

__global__ void CodeGenerator::CUDAKernel::curlNoise4DVector(CurlNoiseParam curlNoiseParam,
	glm::vec3* posRaw,
	glm::vec3* outVecRaw,
	void* noiseRawData,
	float time,
	float dt,
	int maxThreads)
{
	int tid = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (tid >= maxThreads)
		return;
	CodeGenerator::GenericCode::CurlNoise4DVector(tid,
		curlNoiseParam,
		posRaw,
		outVecRaw,
		noiseRawData,
		time,
		dt);
}

void CUDA::CurlNoise4DVector(
	CurlNoiseParam curlNoiseParam,
	CGBufferV3* posBuf,
	CGBufferV3* outVecBuf,
	void* noiseRawData,
	float time,
	float dt,
	int blockSize)
{
	auto posRaw = posBuf->getDevicePointer();
	auto outVecRaw = outVecBuf->getDevicePointer();
	int numThreads = posBuf->getSize();
	glm::vec2 blkInfo = ThreadBlockInfo(blockSize, numThreads);
	CodeGenerator::CUDAKernel::curlNoise4DVector << <blkInfo.x, blkInfo.y >> > (curlNoiseParam,
		posRaw,
		outVecRaw,
		noiseRawData,
		time,
		dt,
		numThreads);
	CG_GET_DEVICE_LAST_ERROR;
}
