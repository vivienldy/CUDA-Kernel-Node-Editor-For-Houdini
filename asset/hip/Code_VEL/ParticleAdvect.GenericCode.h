#include <glm/glm.hpp>
#include "BaseOperation.h"

namespace CodeGenerator 
{ 
	namespace GenericCode 
 	{ 
        __host__ __device__ inline void ParticleAdvect(glm::vec3 geo1_ParticleAdvect_offset_offset, float geo1_ParticleAdvect_input3_input3, float geo1_ParticleAdvect_input2_input2, glm::vec3* geo1_ParticleAdvect_geometryvopglobal1_Pbuffer, float* geo1_ParticleAdvect_geometryvopglobal1_agebuffer, glm::vec3* geo1_ParticleAdvect_geometryvopglobal1_Cdbuffer, float geo1_ParticleAdvect_geometryvopglobal1_TimeInc, CGGeometry::RAWData geo1_ParticleAdvect_geometryvopglobal1_OpInput1, CGGeometry::RAWData geo1_ParticleAdvect_geometryvopglobal1_OpInput2, float* __geo1_ParticleAdvect_add1_sum_debug_buffer, int idx)
        {
            // Data Load 
            // Geometry Global Input
glm::vec3 geo1_ParticleAdvect_geometryvopglobal1_P = geo1_ParticleAdvect_geometryvopglobal1_Pbuffer[idx];
float geo1_ParticleAdvect_geometryvopglobal1_age = geo1_ParticleAdvect_geometryvopglobal1_agebuffer[idx];
glm::vec3 geo1_ParticleAdvect_geometryvopglobal1_Cd = geo1_ParticleAdvect_geometryvopglobal1_Cdbuffer[idx];


            // Compute graph
            
 // Generate by add1
float geo1_ParticleAdvect_add1_sum = geo1_ParticleAdvect_geometryvopglobal1_age + geo1_ParticleAdvect_geometryvopglobal1_TimeInc;

 // Generate by volumesamplefile1
float geo1_ParticleAdvect_volumesamplefile1_volumevalue = volumesamplefile(geo1_ParticleAdvect_geometryvopglobal1_OpInput2, int(0), geo1_ParticleAdvect_geometryvopglobal1_P);

 // Generate by volumesamplefile2
float geo1_ParticleAdvect_volumesamplefile2_volumevalue = volumesamplefile(geo1_ParticleAdvect_geometryvopglobal1_OpInput2, int(1), geo1_ParticleAdvect_geometryvopglobal1_P);

 // Generate by volumesamplefile3
float geo1_ParticleAdvect_volumesamplefile3_volumevalue = volumesamplefile(geo1_ParticleAdvect_geometryvopglobal1_OpInput2, int(2), geo1_ParticleAdvect_geometryvopglobal1_P);

 // Generate by floattovec1
glm::vec3 geo1_ParticleAdvect_floattovec1_vec = floattovec(geo1_ParticleAdvect_volumesamplefile1_volumevalue, geo1_ParticleAdvect_volumesamplefile2_volumevalue, geo1_ParticleAdvect_volumesamplefile3_volumevalue);

 // Generate by curlnoise1
glm::vec3 geo1_ParticleAdvect_curlnoise1_noise = curlnoise(char(pnoise), geo1_ParticleAdvect_geometryvopglobal1_P, glm::vec3(1.0f,1.0f,1.0f), geo1_ParticleAdvect_offset_offset, float(1.0f), float(0.5f), float(1.0f), int(3), float(0.0001f), float(1.0f), float(1.0f), glm::vec3(0.0f,0.0f,0.0f), char(), int(0));

 // Generate by createColor
glm::vec3 geo1_ParticleAdvect_createColor__Cd = createColor(geo1_ParticleAdvect_geometryvopglobal1_P, geo1_ParticleAdvect_geometryvopglobal1_age, geo1_ParticleAdvect_geometryvopglobal1_Cd, geo1_ParticleAdvect_geometryvopglobal1_OpInput1, geo1_ParticleAdvect_geometryvopglobal1_TimeInc);

 // Generate by vectofloat1
float geo1_ParticleAdvect_vectofloat1_fval1;
float geo1_ParticleAdvect_vectofloat1_fval2;
float geo1_ParticleAdvect_vectofloat1_fval3;
vectofloat(geo1_ParticleAdvect_createColor__Cd, &geo1_ParticleAdvect_vectofloat1_fval1, &geo1_ParticleAdvect_vectofloat1_fval2, &geo1_ParticleAdvect_vectofloat1_fval3);

 // Generate by multiply1
glm::vec3 geo1_ParticleAdvect_multiply1_product = geo1_ParticleAdvect_floattovec1_vec * geo1_ParticleAdvect_input2_input2 * geo1_ParticleAdvect_vectofloat1_fval1;

 // Generate by multiply2
float geo1_ParticleAdvect_multiply2_product = geo1_ParticleAdvect_vectofloat1_fval1 * geo1_ParticleAdvect_input3_input3;

 // Generate by clamp1
float geo1_ParticleAdvect_clamp1_clamp = clamp(geo1_ParticleAdvect_multiply2_product, float(0.0f), float(1.0f));

 // Generate by floattovec2
glm::vec3 geo1_ParticleAdvect_floattovec2_vec = floattovec(float(0.0f), geo1_ParticleAdvect_clamp1_clamp, float(0.0f));

 // Generate by add3
glm::vec3 geo1_ParticleAdvect_add3_sum = geo1_ParticleAdvect_floattovec2_vec;

 // Generate by add2
glm::vec3 geo1_ParticleAdvect_add2_sum = geo1_ParticleAdvect_multiply1_product + geo1_ParticleAdvect_add3_sum + geo1_ParticleAdvect_curlnoise1_noise;

 // Generate by multiply3
glm::vec3 geo1_ParticleAdvect_multiply3_product = geo1_ParticleAdvect_add2_sum * geo1_ParticleAdvect_geometryvopglobal1_TimeInc;

 // Generate by add4
glm::vec3 geo1_ParticleAdvect_add4_sum = geo1_ParticleAdvect_multiply3_product + geo1_ParticleAdvect_geometryvopglobal1_P;


            // Write back
            float __geo1_ParticleAdvect_add1_sum_debug = geo1_ParticleAdvect_add1_sum;
__geo1_ParticleAdvect_add1_sum_debug_buffer[idx] = __geo1_ParticleAdvect_add1_sum_debug;

glm::vec3 global_output_geo1_ParticleAdvect_geometryvopoutput1_P = geo1_ParticleAdvect_add4_sum;
geo1_ParticleAdvect_geometryvopglobal1_Pbuffer[idx] = global_output_geo1_ParticleAdvect_geometryvopoutput1_P;

glm::vec3 global_output_geo1_ParticleAdvect_geometryvopoutput1_Cd = geo1_ParticleAdvect_createColor__Cd;
geo1_ParticleAdvect_geometryvopglobal1_Cdbuffer[idx] = global_output_geo1_ParticleAdvect_geometryvopoutput1_Cd;


        }
	} 
} 