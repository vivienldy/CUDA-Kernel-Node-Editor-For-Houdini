#include <glm/glm.hpp>
#include "BaseOperation.h"

namespace CodeGenerator
{
    namespace GenericCode
    {
        __host__ __device__ inline void volumevop1(
            float geo1_volumevop1_input8_input5, 
            float geo1_volumevop1_input7_input4, f
        loat geo1_volumevop1_input5_input3, float geo1_volumevop1_input3_input2, CGGeometry::RawData geo1_volumevop1_volumevopglobal1_OpInput1, CGGeometry::RawData geo1_volumevop1_volumevopglobal1_OpInput2, int idx)
        {
            // Data Load
            // Geometry Global Input

            // Compute graph

            // Generate by threadIdx
            int geo1_volumevop1_threadIdx_idx = idx;

            // Generate by ToVoxelCenter
            glm::vec3 geo1_volumevop1_ToVoxelCenter_p = ToVoxelCenter(geo1_volumevop1_threadIdx_idx);

            // Generate by input6
            glm::vec3 geo1_volumevop1_input6_input2 = glm::vec3(1, 0, 1);

            // Generate by input4
            glm::vec3 geo1_volumevop1_input4_input2 = glm::vec3(0, 1, 0);

            // Generate by input2
            glm::vec3 geo1_volumevop1_input2_input2 = glm::vec3(0.5f, 0.5f, 0.5f);

            // Generate by relbbox1
            glm::vec3 geo1_volumevop1_relbbox1_bbdelta = relbbox(geo1_volumevop1_volumevopglobal1_OpInput2, geo1_volumevop1_ToVoxelCenter_p);

            // Generate by subtract1
            glm::vec3 geo1_volumevop1_subtract1_diff = geo1_volumevop1_relbbox1_bbdelta - geo1_volumevop1_input2_input2;

            // Generate by multiply3
            glm::vec3 geo1_volumevop1_multiply3_product = geo1_volumevop1_subtract1_diff * geo1_volumevop1_input6_input2;

            // Generate by cross1
            glm::vec3 geo1_volumevop1_cross1_crossprod = cross(geo1_volumevop1_multiply3_product, glm::vec3(0.0f, 1.0f, 0.0f));

            // Generate by normalize1
            glm::vec3 geo1_volumevop1_normalize1_nvec = normalize(geo1_volumevop1_cross1_crossprod);

            // Generate by length1
            float geo1_volumevop1_length1_len = length(geo1_volumevop1_multiply3_product);

            // Generate by fit2
            float geo1_volumevop1_fit2_shift = fit(geo1_volumevop1_length1_len, float(0.0f), float(2.5f), float(0.0f), float(1.0f));

            // Generate by compare1
            int geo1_volumevop1_compare1_bool = compare(geo1_volumevop1_length1_len, float(7.5f), int(1));

            // Generate by negate1
            glm::vec3 geo1_volumevop1_negate1_negated = -geo1_volumevop1_multiply3_product;

            // Generate by multiply4
            glm::vec3 geo1_volumevop1_multiply4_product = geo1_volumevop1_negate1_negated * geo1_volumevop1_fit2_shift * geo1_volumevop1_input7_input4;

            // Generate by twoway1
            glm::vec3 geo1_volumevop1_twoway1_result = twoway(geo1_volumevop1_compare1_bool, geo1_volumevop1_multiply4_product, glm::vec3(0.0f, 0.0f, 0.0f), int(0));

            // Generate by relbbox2
            glm::vec3 geo1_volumevop1_relbbox2_bbdelta = relbbox(geo1_volumevop1_volumevopglobal1_OpInput1, geo1_volumevop1_ToVoxelCenter_p);

            // Generate by vectofloat1
            float geo1_volumevop1_vectofloat1_fval1;
            float geo1_volumevop1_vectofloat1_fval2;
            float geo1_volumevop1_vectofloat1_fval3;
            vectofloat(geo1_volumevop1_relbbox2_bbdelta, &geo1_volumevop1_vectofloat1_fval1, &geo1_volumevop1_vectofloat1_fval2, &geo1_volumevop1_vectofloat1_fval3);

            // Generate by ramp1
            float geo1_volumevop1_ramp1_ramp;
            char geo1_volumevop1_ramp1_ramp_the_basis_strings;
            float geo1_volumevop1_ramp1_ramp_the_key_positions;
            float geo1_volumevop1_ramp1_ramp_the_key_values;
            char geo1_volumevop1_ramp1_ramp_the_color_space;
            int geo1_volumevop1_ramp1_ramp_struct;
            rampparm(geo1_volumevop1_vectofloat1_fval2, &geo1_volumevop1_ramp1_ramp, &geo1_volumevop1_ramp1_ramp_the_basis_strings, &geo1_volumevop1_ramp1_ramp_the_key_positions, &geo1_volumevop1_ramp1_ramp_the_key_values, &geo1_volumevop1_ramp1_ramp_the_color_space, &geo1_volumevop1_ramp1_ramp_struct);

            // Generate by complement1
            float geo1_volumevop1_complement1_complem = 1 - geo1_volumevop1_ramp1_ramp;

            // Generate by multiply1
            glm::vec3 geo1_volumevop1_multiply1_product = geo1_volumevop1_input4_input2 * geo1_volumevop1_input5_input3 * geo1_volumevop1_complement1_complem;

            // Generate by add1
            glm::vec3 geo1_volumevop1_add1_sum = geo1_volumevop1_normalize1_nvec + geo1_volumevop1_multiply1_product;

            // Generate by ramp2
            float geo1_volumevop1_ramp2_rampx;
            char geo1_volumevop1_ramp2_rampx_the_basis_strings;
            float geo1_volumevop1_ramp2_rampx_the_key_positions;
            float geo1_volumevop1_ramp2_rampx_the_key_values;
            char geo1_volumevop1_ramp2_rampx_the_color_space;
            int geo1_volumevop1_ramp2_rampx_struct;
            rampparm(geo1_volumevop1_vectofloat1_fval2, &geo1_volumevop1_ramp2_rampx, &geo1_volumevop1_ramp2_rampx_the_basis_strings, &geo1_volumevop1_ramp2_rampx_the_key_positions, &geo1_volumevop1_ramp2_rampx_the_key_values, &geo1_volumevop1_ramp2_rampx_the_color_space, &geo1_volumevop1_ramp2_rampx_struct);

            // Generate by multiply5
            float geo1_volumevop1_multiply5_product = geo1_volumevop1_ramp2_rampx * geo1_volumevop1_input8_input5;

            // Generate by fit1
            float geo1_volumevop1_fit1_shift = fit(geo1_volumevop1_length1_len, geo1_volumevop1_multiply5_product, float(0.0f), float(0.0f), float(1.0f));

            // Generate by multiply2
            glm::vec3 geo1_volumevop1_multiply2_product = geo1_volumevop1_add1_sum * geo1_volumevop1_fit1_shift * geo1_volumevop1_input3_input2;

            // Generate by add2
            glm::vec3 geo1_volumevop1_add2_sum = geo1_volumevop1_twoway1_result + geo1_volumevop1_multiply2_product;

            // Generate by setVoxelData
            setVoxelData(geo1_volumevop1_volumevopglobal1_OpInput1, geo1_volumevop1_add2_sum, geo1_volumevop1_threadIdx_idx)

            // Write back
        }
    }
}