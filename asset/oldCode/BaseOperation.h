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
            template<typename T>
            __host__ __device__ inline void VectorFieldDataSplit(
                typename CGVectorField3D<T>::RAWData& inputVectorField,
                typename CGField3D<T>::RAWData* seperateScalarFields)
            {
                typename CGField3D<T>::RAWData rawDataX;
                typename CGField3D<T>::RAWData rawDataY;
                typename CGField3D<T>::RAWData rawDataZ;
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
                CGField3DInfo& info)
            {
                return idz * info.Resolution.x * info.Resolution.y + idy * info.Resolution.x + idx;
            }

            template<typename T>
            __host__ __device__ inline glm::vec3 PosToIndex3D(
                glm::vec3 pos,
                typename CGField3D<T>::RAWData rawData)
            {
                // according to position, calculate the x, y, z for the pos
                int x = (pos.x - (rawData.FieldInfo.Pivot.x - rawData.FieldInfo.FieldSize.x * 0.5f)) * rawData.FieldInfo.InverseVoxelSize;
                int y = (pos.y - (rawData.FieldInfo.Pivot.y - rawData.FieldInfo.FieldSize.y * 0.5f)) * rawData.FieldInfo.InverseVoxelSize;
                int z = (pos.z - (rawData.FieldInfo.Pivot.z - rawData.FieldInfo.FieldSize.z * 0.5f)) * rawData.FieldInfo.InverseVoxelSize;
                return glm::vec3(x, y, z);
            }

            template<typename T>
            __host__ __device__ inline glm::vec3 PosToIndex1D(
                glm::vec3 pos,
                typename CGField3D<T>::RAWData rawData)
            {
                glm::vec3 index3D = Field::GenericCode::PosToIndex3D<T>(pos, rawData);
                int index1D = Field::GenericCode::Index3DToIndex1D(index3D.x, index3D.y, index3D.z, rawData.FieldInfo);
                return index1D;
            }

            __host__ __device__ inline glm::vec3 IndexToIndex3(
                int idx, 
                const CGField3DInfo& info)
            {
                glm::vec3 res = info.Resolution;
                int nvSlice = res.x * res.y;
                int x = idx % (int)res.x;
                int y = (idx % nvSlice) / res.x;
                int z = idx / nvSlice;
                return glm::vec3(x, y, z);
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

            // ----- GetValue SampleValue function
            template<typename T>
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
                    return 0.f; // [TODO] type?  if T int?
                }
                else
                {
                    return rawData[Field::GenericCode::Index3DToIndex1D(idx, idy, idz, info)];
                }
            }

            template<typename T>
            __host__ __device__ inline T SampleValueScalarField(
                glm::vec3 pos,
                T* rawData,
                CGField3DInfo& info)
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

                auto v0 = Field::GenericCode::GetValue<T>(lowerLeft.x, lowerLeft.y, lowerLeft.z, rawData, info);
                auto v1 = Field::GenericCode::GetValue<T>(lowerLeft.x + 1, lowerLeft.y, lowerLeft.z, rawData, info);
                auto v2 = Field::GenericCode::GetValue<T>(lowerLeft.x, lowerLeft.y + 1, lowerLeft.z, rawData, info);
                auto v3 = Field::GenericCode::GetValue<T>(lowerLeft.x + 1, lowerLeft.y + 1, lowerLeft.z, rawData, info);
                auto v4 = Field::GenericCode::GetValue<T>(lowerLeft.x, lowerLeft.y, lowerLeft.z + 1, rawData, info);
                auto v5 = Field::GenericCode::GetValue<T>(lowerLeft.x + 1, lowerLeft.y, lowerLeft.z + 1, rawData, info);
                auto v6 = Field::GenericCode::GetValue<T>(lowerLeft.x, lowerLeft.y + 1, lowerLeft.z + 1, rawData, info);
                auto v7 = Field::GenericCode::GetValue<T>(lowerLeft.x + 1, lowerLeft.y + 1, lowerLeft.z + 1, rawData, info);

                auto vy0 = glm::mix(glm::mix(v0, v1, cx), glm::mix(v2, v3, cx), cy);
                auto vy1 = glm::mix(glm::mix(v4, v5, cx), glm::mix(v6, v7, cx), cy);
                return glm::mix(vy0, vy1, cz);
            }

            //SampleValue Vector3
            template<typename T>
            __host__ __device__ inline glm::vec3 SampleValueVectorField(
                glm::vec3 pos,
                typename CGVectorField3D<T>::RAWData& vectorField)
            {
                CGField3D<T>::RAWData scalarFields[3];
                Field::GenericCode::VectorFieldDataSplit<T>(vectorField, scalarFields);
                auto x = Field::GenericCode::SampleValueScalarField<T>(pos, scalarFields[0].VoxelData, scalarFields[0].FieldInfo);
                auto y = Field::GenericCode::SampleValueScalarField<T>(pos, scalarFields[1].VoxelData, scalarFields[1].FieldInfo);
                auto z = Field::GenericCode::SampleValueScalarField<T>(pos, scalarFields[2].VoxelData, scalarFields[2].FieldInfo);
                return glm::vec3(x, y, z);
            }

            // scalar field set value
            template<typename T>
            __host__ __device__ inline void SetValueScalar(
                int idx,
                T value,
                typename CGField3D<T>::RAWData& rawData)
            {
                rawData.VoxelData[idx] = value;
            }

            // vector field set value
            template<typename T>
            __host__ __device__ inline void SetValueVector3(
                int idx,
                glm::vec3 vec3,
                typename CGVectorField3D<T>::RAWData& vectorField)
            {
                vectorField.VoxelDataX[idx] = vec3.x;
                vectorField.VoxelDataY[idx] = vec3.y;
                vectorField.VoxelDataZ[idx] = vec3.z;
            }

            template<typename T>
            __host__ __device__ inline int GetNumOfVoxel(
                typename CGVectorField3D<T>::RAWData& vectorField)
            {
                if (vectorField.FieldInfoX.NumVoxels == vectorField.FieldInfoY.NumVoxels &&
                    vectorField.FieldInfoX.NumVoxels == vectorField.FieldInfoZ.NumVoxels &&
                    vectorField.FieldInfoY.NumVoxels == vectorField.FieldInfoZ.NumVoxels) {
                    return vectorField.FieldInfoX.NumVoxels;
                }
                return -1;
            }

            template<typename T>
            __host__ __device__ inline glm::vec3 Index3ToPos(
                int indexX, int indexY, int indexZ,
                CGField3DInfo& info)
            {

                glm::vec3 origin = info.Pivot - info.FieldSize * 0.5f;
                glm::vec3 localPos = glm::vec3(
                    ((float)indexX + 0.5f) * info.VoxelSize,
                    ((float)indexY + 0.5f) * info.VoxelSize,
                    ((float)indexZ + 0.5f) * info.VoxelSize);
                return localPos + origin;
            }

            template<typename T>
            __host__ __device__ inline glm::vec3 IndexToPos(
                int idx, 
                typename CGVectorField3D<T>::RAWData& vectorField)
            {
                // because we are sure now three scalar field infos are same, so just simply get fieldX's info
               
                // index to index3
                glm::vec3 idx3 = Field::GenericCode::IndexToIndex3(idx, vectorField.FieldInfoX);

                // index3 to pos
                return Field::GenericCode::Index3ToPos<T>(idx3.x, idx3.y, idx3.z, vectorField.FieldInfoX);
            }

        }
    }
}