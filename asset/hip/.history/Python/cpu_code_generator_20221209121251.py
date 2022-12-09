import re
import json

# Codeline class
class Codeline:
    def __init__(self):
        self.left = ""
        self.right = ""
    def toCode(self):
        if not self.left == "" and not self.right == "":
            return self.left + " = " + self.right + ";\n"
        else:
            return ""

# ----- Common ------
def getInputBufferDict(jsonObj):
    buffer_dict = {}
   
    global_input = jsonObj["global_input"] 
    for input_node_key in global_input:
        input_node = global_input[input_node_key]
        
        if ("volumevopglobal" in input_node_key) or ("geometryvopglobal" in input_node_key):
            for input_port in input_node:
                if input_port["is_buffer"] == "True":
                    buffer_dict[input_port["port_name"]] = input_port["variable_name"]
    return buffer_dict

# Fucntion name
def funcNameGenerator(jsonObj):
    return jsonObj["func_name"]

# Parm list
def parmListGenerator(jsonObj):
    buffer_dict = getInputBufferDict(jsonObj)
    result = ""
    global_input = jsonObj["global_input"] 
    for input_node_key in global_input:
        input_node = global_input[input_node_key]
        
        if ("volumevopglobal" in input_node_key) or ("geometryvopglobal" in input_node_key):
            for input_port in input_node:
                # check buffer type
                if input_port["is_buffer"] == "True":
                    result += "CGBuffer<" + input_port["data_type"] + ">* " + input_port["variable_name"] + "buffer" + ", "  #i.e. glm::vec3* posBuffer,
                else:
                    result += input_port["data_type"] + " " + input_port["variable_name"] + ", "  #i.e. float dt,
        # for custom_param
        else:
            for input_port in input_node:
                result += input_port["data_type"] + " " + input_port["variable_name"] + ", "
    
    global_output = jsonObj["global_output"]
    for output_node_key in global_output:
        output_node = global_output[output_node_key]
        
        if ("volumevopoutput1" in output_node_key) or ("geometryvopoutput" in output_node_key):
            for output_port in output_node:
                if not output_port["port_name"] in buffer_dict:
                    result += "CGBuffer<" + output_port["data_type"] + ">* " + output_port["variable_name"] + "buffer" + ", "
        else:
            for output_port in output_node:
                result += "CGBuffer<" + input_port["data_type"] + ">* "  + output_port["variable_name"] + "_buffer"  + ", "

    return result[:-2]
  
def projNameGenerator(jsonObj):
    return jsonObj["proj_name"]

def rawDataGenerator(jsonObj):
    result = ""
    return result

def shareCodeParamGenerator(jsonObj):
    buffer_dict = getInputBufferDict(jsonObj)
    result = ""
    global_input = jsonObj["global_input"] 
    for input_node_key in global_input:
        input_node = global_input[input_node_key]
        
        if ("volumevopglobal" in input_node_key) or ("geometryvopglobal" in input_node_key):
            for input_port in input_node:
                # check buffer type
                if input_port["is_buffer"] == "True":
                    result += input_port["variable_name"] + "buffer" + "->getRawData()" + ", "  #i.e. pos->getRawData(),
                else:
                    result += input_port["variable_name"] + ", "  #i.e. dt,
        # for custom_param
        else:
            for input_port in input_node:
                if input_port["is_buffer"] == "True":
                    result += input_port["variable_name"] + "buffer" + "->getRawData()" + ", "  #i.e. pos->getRawData(),
                else:
                    result += input_port["variable_name"] + ", "
    
    global_output = jsonObj["global_output"]
    for output_node_key in global_output:
        output_node = global_output[output_node_key]
        
        if ("volumevopoutput1" in output_node_key) or ("geometryvopoutput" in output_node_key):
            for output_port in output_node:
                if not output_port["port_name"] in buffer_dict:
                    result += output_port["variable_name"] + "buffer" + "->getRawData()" + ", "
        else:
            for output_port in output_node:
                result += output_port["variable_name"] + "_buffer" + "->getRawData()"  + ", "

    return result[:-2]

def getNumThreadGenerator(jsonObj):
    result = jsonObj["parallel_reference_name"]

    global_input = jsonObj["global_input"] 
    for input_node_key in global_input:
        input_node = global_input[input_node_key]
        
        if ("volumevopglobal" in input_node_key) or ("geometryvopglobal" in input_node_key):
            for input_port in input_node:
                # check buffer type
                if input_port[] == result input_port["is_buffer"] == "True":
                    result += input_port["variable_name"] + "buffer" + "->getRawData()" + ", "  #i.e. pos->getRawData(),
                else:
                    result += input_port["variable_name"] + ", "  #i.e. dt,
    return 

# ----- Code Segment Map -----
replacementMap = {
    "PROJ_NAME":projNameGenerator,
    "FUNC_NAME":funcNameGenerator,
    "FUNC_DECLARE_LIST":parmListGenerator,
    "GET_RAWDATA":rawDataGenerator,
    "SHARE_CODE_PARAM":shareCodeParamGenerator,
    "GET_NUM_THREAD":getNumThreadGenerator
}

# ----- CPU code generate -----

this_node = hou.pwd()  
# Load cpu_json
# geo = this_node.geometry()
# json_str = geo.findGlobalAttrib("cpu_json").strings()
# json_str = json_str[0]
# jsonObj = json.loads(json_str)
json_input = this_node.input(2)
json_str = json_input.parm("/obj/geo1/CPU_JSON/content").eval()
jsonObj = json.loads(json_str)


# Read in the template
with open('./Template/CPUTemplate_h.h', 'r') as file :
    filedata_h = file.read()

with open('./Template/CPUTemplate_cpp.h', 'r') as file :
    filedata_cpp = file.read()

# Parse keywords from template
code_segments_h = re.findall('@(.+?)@', filedata_h)
code_segments_cpp = re.findall('@(.+?)@', filedata_cpp)

# Replace the target string 
for code_segment in code_segments_h:
    if code_segment in replacementMap:
        tmp_str = "@" +  code_segment + "@"
        target_code = replacementMap[code_segment](jsonObj)
        filedata_h = filedata_h.replace(tmp_str, target_code)

for code_segment in code_segments_cpp:
    if code_segment in replacementMap:
        tmp_str = "@" +  code_segment + "@"
        target_code = replacementMap[code_segment](jsonObj)
        filedata_cpp = filedata_cpp.replace(tmp_str, target_code)
        
# Output cpu to file
file_name = "./Code_VEL/" + projNameGenerator(jsonObj=jsonObj) + ".h"
with open(file_name, 'w') as file:
    file.write(filedata_h)   
    
file_name = "./Code_VEL/" + projNameGenerator(jsonObj=jsonObj) + ".cpp"
with open(file_name, 'w') as file:
    file.write(filedata_cpp)   

