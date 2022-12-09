#include <glm/glm.hpp>
#include "BaseOperation.h"

namespace CodeGenerator
{
        namespace GenericCode
        {
                __host__ __device__ inline void pointvop1(glm::vec3 geo1_pointvop1_offset_offset, float geo1_pointvop1_input3_input3, float geo1_pointvop1_input2_input2, glm::vec3 *geo1_pointvop1_geometryvopglobal1_Pbuffer, glm::vec3 *geo1_pointvop1_geometryvopglobal1_Cdbuffer, struct geo1_pointvop1_geometryvopglobal1_OpInput2, int idx)
                {
                        // Data Load
                        // Geometry Global Input
                        glm::vec3 geo1_pointvop1_geometryvopglobal1_P = geo1_pointvop1_geometryvopglobal1_Pbuffer[idx];
                        glm::vec3 geo1_pointvop1_geometryvopglobal1_Cd = geo1_pointvop1_geometryvopglobal1_Cdbuffer[idx];

                        // Compute graph

                        // Generate by vectofloat1
                        float geo1_pointvop1_vectofloat1_fval1;
                        float geo1_pointvop1_vectofloat1_fval2;
                        float geo1_pointvop1_vectofloat1_fval3;
                        vectofloat(geo1_pointvop1_geometryvopglobal1_Cd, &geo1_pointvop1_vectofloat1_fval1, &geo1_pointvop1_vectofloat1_fval2, &geo1_pointvop1_vectofloat1_fval3);

                        // Generate by multiply2
                        float geo1_pointvop1_multiply2_product = geo1_pointvop1_vectofloat1_fval1 * geo1_pointvop1_input3_input3;

                        // Generate by clamp1
                        float geo1_pointvop1_clamp1_clamp = clamp(geo1_pointvop1_multiply2_product, float(0.0f), float(1.0f));

                        // Generate by floattovec2
                        glm::vec3 geo1_pointvop1_floattovec2_vec = floattovec(float(0.0f), geo1_pointvop1_clamp1_clamp, float(0.0f));

                        // Generate by add3
                        glm::vec3 geo1_pointvop1_add3_sum = geo1_pointvop1_floattovec2_vec;

                        // Generate by volumesamplefile1
                        float geo1_pointvop1_volumesamplefile1_volumevalue = volumesamplefile(geo1_pointvop1_geometryvopglobal1_OpInput2, geo1_pointvop1_geometryvopglobal1_P);

                        // Generate by volumesamplefile2
                        float geo1_pointvop1_volumesamplefile2_volumevalue = volumesamplefile(geo1_pointvop1_geometryvopglobal1_OpInput2, geo1_pointvop1_geometryvopglobal1_P);

                        // Generate by volumesamplefile3
                        float geo1_pointvop1_volumesamplefile3_volumevalue = volumesamplefile(geo1_pointvop1_geometryvopglobal1_OpInput2, geo1_pointvop1_geometryvopglobal1_P);

                        // Generate by floattovec1
                        glm::vec3 geo1_pointvop1_floattovec1_vec = floattovec(geo1_pointvop1_volumesamplefile1_volumevalue, geo1_pointvop1_volumesamplefile2_volumevalue, geo1_pointvop1_volumesamplefile3_volumevalue);

                        // Generate by multiply1
                        glm::vec3 geo1_pointvop1_multiply1_product = geo1_pointvop1_floattovec1_vec * geo1_pointvop1_input2_input2 * geo1_pointvop1_vectofloat1_fval1;

                        // Generate by curlnoise2
                        glm::vec3 geo1_pointvop1_curlnoise2_noise = curlnoise(char(xnoise), geo1_pointvop1_geometryvopglobal1_P, glm::vec3(0.24f, 0.24f, 0.24f), geo1_pointvop1_offset_offset, float(6.0f), float(0.5f), float(1.0f), int(3), float(0.0001f), float(1.0f), float(1.0f), glm::vec3(0.0f, 0.0f, 0.0f), char(), int(0));

                        // Generate by add2
                        glm::vec3 geo1_pointvop1_add2_sum = geo1_pointvop1_multiply1_product + geo1_pointvop1_curlnoise2_noise + geo1_pointvop1_add3_sum;

                        // Generate by add4
                        glm::vec3 geo1_pointvop1_add4_sum = geo1_pointvop1_add2_sum + geo1_pointvop1_geometryvopglobal1_P;

                        // Write back
                        glm::vec3 global_output_geo1_pointvop1_geometryvopoutput1_P = geo1_pointvop1_add4_sum;
                        geo1_pointvop1_geometryvopglobal1_Pbuffer[idx] = global_output_geo1_pointvop1_geometryvopoutput1_P;
                }
        }
}