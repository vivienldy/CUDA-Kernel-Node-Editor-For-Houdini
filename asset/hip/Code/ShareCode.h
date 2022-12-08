#include <glm/glm.hpp>

namespace CodeGenerator 
{ 
	namespace GenericCode 
 	{ 
        __host__ __device__ inline void volumevop1(float geo1_volumevop1_input8_input5, float geo1_volumevop1_input7_input4, float geo1_volumevop1_input5_input3, float geo1_volumevop1_input3_input2, glm::vec3* geo1_volumevop1_volumevopglobal1_Pbuffer, struct* geo1_volumevop1_volumevopglobal1_OpInput1buffer, struct* geo1_volumevop1_volumevopglobal1_OpInput2buffer, glm::vec3* geo1_volumevop1_velbuffer, int idx)
        {
            // Data Load 
            // Geometry Global Input
glm::vec3 geo1_volumevop1_volumevopglobal1_P = geo1_volumevop1_volumevopglobal1_Pbuffer[idx];
struct geo1_volumevop1_volumevopglobal1_OpInput1 = geo1_volumevop1_volumevopglobal1_OpInput1buffer[idx];
struct geo1_volumevop1_volumevopglobal1_OpInput2 = geo1_volumevop1_volumevopglobal1_OpInput2buffer[idx];


            // Compute graph
            glm::vec3 geo1_volumevop1_input6_input2 = 1, 0, 1;
glm::vec3 geo1_volumevop1_input4_input2 = 0, 1, 0;
glm::vec3 geo1_volumevop1_input2_input2 = 0.5f, 0.5f, 0.5f;
glm::vec3 geo1_volumevop1_relbbox1_bbdelta = relbbox(geo1_volumevop1_volumevopglobal1_OpInput2, geo1_volumevop1_volumevopglobal1_P);
glm::vec3 geo1_volumevop1_subtract1_diff = geo1_volumevop1_relbbox1_bbdelta - geo1_volumevop1_input2_input2;
glm::vec3 geo1_volumevop1_multiply3_product = geo1_volumevop1_subtract1_diff * geo1_volumevop1_input6_input2;
glm::vec3 geo1_volumevop1_cross1_crossprod = cross(geo1_volumevop1_multiply3_product);
glm::vec3 geo1_volumevop1_normalize1_nvec = normalize(geo1_volumevop1_cross1_crossprod);
float geo1_volumevop1_length1_len = length(geo1_volumevop1_multiply3_product);
float geo1_volumevop1_fit2_shift = fit(geo1_volumevop1_length1_len, float(0.0f), float(2.5f), float(0.0f), float(1.0f));
int geo1_volumevop1_compare1_bool = compare(geo1_volumevop1_length1_len);
glm::vec3 geo1_volumevop1_negate1_negated =  - geo1_volumevop1_multiply3_product;
glm::vec3 geo1_volumevop1_multiply4_product = geo1_volumevop1_negate1_negated * geo1_volumevop1_fit2_shift * geo1_volumevop1_input7_input4;
glm::vec3 geo1_volumevop1_twoway1_result = twoway(geo1_volumevop1_compare1_bool, geo1_volumevop1_multiply4_product);
glm::vec3 geo1_volumevop1_relbbox2_bbdelta = relbbox(geo1_volumevop1_volumevopglobal1_OpInput1, geo1_volumevop1_volumevopglobal1_P);
float geo1_volumevop1_vectofloat1_fval2 = vectofloat(geo1_volumevop1_relbbox2_bbdelta);
float geo1_volumevop1_ramp1_ramp = rampparm(geo1_volumevop1_vectofloat1_fval2);
float geo1_volumevop1_complement1_complem =  1 - geo1_volumevop1_ramp1_ramp;
glm::vec3 geo1_volumevop1_multiply1_product = geo1_volumevop1_input4_input2 * geo1_volumevop1_input5_input3 * geo1_volumevop1_complement1_complem;
glm::vec3 geo1_volumevop1_add1_sum = geo1_volumevop1_normalize1_nvec + geo1_volumevop1_multiply1_product;
float geo1_volumevop1_ramp2_rampx = rampparm(geo1_volumevop1_vectofloat1_fval2);
float geo1_volumevop1_multiply5_product = geo1_volumevop1_ramp2_rampx * geo1_volumevop1_input8_input5;
float geo1_volumevop1_fit1_shift = fit(geo1_volumevop1_length1_len, geo1_volumevop1_multiply5_product, float(0.0f), float(0.0f), float(1.0f));
glm::vec3 geo1_volumevop1_multiply2_product = geo1_volumevop1_add1_sum * geo1_volumevop1_fit1_shift * geo1_volumevop1_input3_input2;
glm::vec3 geo1_volumevop1_add2_sum = geo1_volumevop1_twoway1_result + geo1_volumevop1_multiply2_product;


            // Write bacl
            glm::vec3 global_output_geo1_volumevop1_vel = geo1_volumevop1_add2_sum;
geo1_volumevop1_velbuffer[idx] = global_output_geo1_volumevop1_vel;


        }
	} 
} 