#include <glm/glm.hpp>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include "CGField.h"

namespace CodeGenerator 
{ 
    namespace Field
    {
        namespace GenericCode
        {
            // ----- helper function
            __host__ __device__ inline glm::vec3 curlnoise(glm::vec3 pos, glm::vec3 freq, glm::vec3 offset, glm::vec3 nml, string type, string geo, int turb, int bounce, float amp, float rough, float atten, float distance, float radius, float h) {
                return glm::vec3(1.f);
            }

            __host__ __device__ inline float fit(float x, float omin, float omax, float nmin, float nmax) {
                float t;
                t = glm::clamp((x - omin)/(omax - omin), 0, 1);
                return glm::mix(nmin, nmax, t);
            }

            template<class T>
            __host__ __device__ inline float length(T vec) {
                return glm::length(vec);
            }

            template<class T>
            __host__ __device__ inline T normalize(T vec) {
                return glm::normalize(vec);
            }

            template<class T>
            __host__ __device__ inline float cross(T a, T b) {
                return glm::cross(a, b);
            }

            __host__ __device__ inline void	vectofloat(glm::vec3 v, float& x, float& y, float& z) {
	            x = v[0]; y = v[1]; z = v[2]; 
            }

            __host__ __device__ inline bool	compare(float val, float val_to_compare, string op) {
                if (string == "Equal") return val == val_to_compare;
                if (string == "Less Than") return val < val_to_compare;
                if (string == "Greater Than") return val > val_to_compare;
                if (string == "Less Than Or Equal") return val <= val_to_compare;
                if (string == "Greater Than Or Equal") return val >= val_to_compare;
                if (string == "Not Equal") return val != val_to_compare;
            }

            template<class T>
            __host__ __device__ inline T twoway(bool condition, T input1, T input2, string signature, string conditionStr) {
                if (conditionStr == "Use Input 1 If Condition True") {
                    if (condition) return input1;
                    else return input2;
                }
                else {
                    if (!condition) return input1;
                    else return input2;
                }
            }
        
            template<class T>
            __host__ __device__ inline void VectorFieldDataSplit(
                const CGVectorField3D<T>::RAWData& inputVectorField,
                CGField3D<T>::RAWData* seperateScalarFields) {
                CGField3D<T>::RAWData rawDataX;
                CGField3D<T>::RAWData rawDataY;
                CGField3D<T>::RAWData rawDataZ;
                rawDataX.IsValid = true;
                rawDataX.FieldInfo = inputVectorField.FieldInfoX;
                rawDataX.VoxelData = inputVectorField.VoxelDataX;
                rawDataY.IsValid = true;
                rawDataY.FieldInfo = inputVectorField.FieldInfoY;
                rawDataY.VoxelData = inputVectorField.VoxelDataY;
                rawDataZ.IsValid = true;
                rawDataZ.FieldInfo = inputVectorField.FieldInfoZ;
                rawDataZ.VoxelData = inputVectorField.VoxelDataZ;
                seperateScalarFields[0] = rawDataX;
                seperateScalarFields[1] = rawDataX;
                seperateScalarFields[2] = rawDataX;
            }

            __host__ __device__ inline int Index3DToIndex1D(
                int idx, 
                int idy, 
                int idz, 
                CGField3DInfo& info){
                return idz * info.Resolution.x * info.Resolution.y + idy * info.Resolution.x + idx;
            }

            template<class T>
            __host__ __device__ inline glm::vec3 PosToIndex3D(
                glm::vec3 pos,
                CGField3D<T>::RAWData rawData) {
                // according to position, calculate the x, y, z for the pos
                int x = (pos.x - (rawData.FieldInfo.Pivot.x - rawData.FieldInfo.FieldSize.x * 0.5f)) * rawData.FieldInfo.InverseVoxelSize;
                int y = (pos.y - (rawData.FieldInfo.Pivot.y - rawData.FieldInfo.FieldSize.y * 0.5f)) * rawData.FieldInfo.InverseVoxelSize;
                int z = (pos.z - (rawData.FieldInfo.Pivot.z - rawData.FieldInfo.FieldSize.z * 0.5f)) * rawData.FieldInfo.InverseVoxelSize;
                return glm::vec3(x, y, z);
            }

            template<class T>
            __host__ __device__ inline glm::vec3 PosToIndex1D(
                glm::vec3 pos,
                CGField3D<T>::RAWData rawData) {
                glm::vec3 index3D = Field::GenericCode::PosToIndex3D<T>(pos, rawData);
                int index1D = Field::GenericCode::Index3DToIndex1D(index3D.x, index3D.y, index3D.z, rawData.FieldInfo);
                return index1D;
            }

            __host__ __device__ inline bool IsInside(
                int idx,
                int idy,
                int idz, 
                const CGField3DInfo& info)
            {
                if (idx < 0 || idx >= (int)info.Resolution.x ||
                    idy < 0 || idy >= (int)info.Resolution.y ||
                    idz < 0 || idz >= (int)info.Resolution.z)
                    return false;
                return true;
            }

            //Index3ToPos() {}

            // ----- GetValue SampleValue function
            template<class T>
            __host__ __device__ inline T GetValue(
                int idx,
                int idy,
                int idz,
                T* rawData,
                CGField3DInfo& info)
            {
                // boundary condition, if outside return 0
                if (!Field::GenericCode::IsInside(idx, idy, idz, info))
                {
                    return 0; // [TODO] type?  if T int?
                }
                else
                {
                    return rawData[Field::GenericCode::Index3DToIndex1D(idx, idy, idz, info)];
                }
            }

            template<class T>
            __host__ __device__ inline T SampleValueScalarField(
                glm::vec3 pos,
                T* rawData,
                const CGField3DInfo& info)
            {
                glm::vec3 origin = info.Pivot - info.FieldSize * 0.5f;
                glm::vec3 bboxSpace = glm::vec3(
                    ((pos.x - origin.x) / info.FieldSize.x),
                    ((pos.y - origin.y) / info.FieldSize.y),
                    ((pos.z - origin.z) / info.FieldSize.z));

                glm::vec3 voxelCoord = glm::vec3(
                    bboxSpace.x * (float)info.Resolution.x,
                    bboxSpace.y * (float)info.Resolution.y,
                    bboxSpace.z * (float)info.Resolution.z);

                glm::vec3 coordB = voxelCoord - 0.5f;
                glm::vec3 lowerLeft = glm::vec3(
                    floor(coordB.x),
                    floor(coordB.y),
                    floor(coordB.z));

                float cx = coordB.x - (float)lowerLeft.x;
                float cy = coordB.y - (float)lowerLeft.y;
                float cz = coordB.z - (float)lowerLeft.z;

                auto v0 = Field::GenericCode::GetValue<T>(lowerLeft.x, lowerLeft.y, lowerLeft.z, rawData, info) * 0.003921f;
                auto v1 = Field::GenericCode::GetValue<T>(lowerLeft.x + 1, lowerLeft.y, lowerLeft.z, rawData, info) * 0.003921f;
                auto v2 = Field::GenericCode::GetValue<T>(lowerLeft.x, lowerLeft.y + 1, lowerLeft.z, rawData, info) * 0.003921f;
                auto v3 = Field::GenericCode::GetValue<T>(lowerLeft.x + 1, lowerLeft.y + 1, lowerLeft.z, rawData, info) * 0.003921f;
                auto v4 = Field::GenericCode::GetValue<T>(lowerLeft.x, lowerLeft.y, lowerLeft.z + 1, rawData, info) * 0.003921f;
                auto v5 = Field::GenericCode::GetValue<T>(lowerLeft.x + 1, lowerLeft.y, lowerLeft.z + 1, rawData, info) * 0.003921f;
                auto v6 = Field::GenericCode::GetValue<T>(lowerLeft.x, lowerLeft.y + 1, lowerLeft.z + 1, rawData, info) * 0.003921f;
                auto v7 = Field::GenericCode::GetValue<T>(lowerLeft.x + 1, lowerLeft.y + 1, lowerLeft.z + 1, rawData, info) * 0.003921f;

                auto vy0 = glm::mix(glm::mix(v0, v1, cx), glm::mix(v2, v3, cx), cy);
                auto vy1 = glm::mix(glm::mix(v4, v5, cx), glm::mix(v6, v7, cx), cy);
                return glm::mix(vy0, vy1, cz);
            }           

            //SampleValue Vector3
            template<class T>
            __host__ __device__ inline glm::vec3 SampleValueVectorField(
                glm::vec3 pos, 
                CGVectorField3D<T>::RAWData& vectorField)
            {
                CGField3D<T>::RAWData scalarFields[3];
                Field::GenericCode::VectorFieldDataSplit<T>(vectorField, scalarFields);
                auto x = Field::GenericCode::SampleValueScalarField<T>(pos, scalarFields[0].VoxelData, scalarFields[0].FieldInfo);
                auto y = Field::GenericCode::SampleValueScalarField<T>(pos, scalarFields[1].VoxelData, scalarFields[1].FieldInfo);
                auto z = Field::GenericCode::SampleValueScalarField<T>(pos, scalarFields[2].VoxelData, scalarFields[2].FieldInfo);
                return glm::vec3(x, y, z);
            }
           
        }
    }
} 