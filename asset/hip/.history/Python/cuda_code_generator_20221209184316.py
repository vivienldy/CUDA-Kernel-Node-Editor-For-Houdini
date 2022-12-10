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
    
# ----- Code Segment Generators -----

# File name
def fileNameGenerator(jsonObj):
    return jsonObj["out_filename"]

# Kernel name
def kernelNameGenerator(jsonObj):
    return jsonObj["kernel_name"]

# Fucntion name
def funcNameGenerator(jsonObj):
    return jsonObj["func_name"]

# Kernel Launch name
def kernelLaunchNameGenerator(jsonObj):
    return jsonObj["kernel_launch_name"]

# Kernel parm declare
def kernelParmDeclareGenerator(jsonObj):
    buffer_dict = getInputBufferDict(jsonObj)
    result = ""
    global_input = jsonObj["global_input"] 
    for input_node_key in global_input:
        input_node = global_input[input_node_key]
        
        if ("volumevopglobal" in input_node_key) or ("geometryvopglobal" in input_node_key):
            for input_port in input_node:
                # check buffer type
                if input_port["is_buffer"] == "True":
                    result += input_port["data_type"] + "* " + input_port["variable_name"] + "buffer" + ", "  #i.e. glm::vec3* posBuffer,
                elif input_port["is_OpInput"] == "True":
                    result += "CGGeometry::RAWData " + input_port["variable_name"] + ", " 
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
                    result += output_port["data_type"] + "* " + output_port["variable_name"] + "buffer" + ", "
        else:
            for output_port in output_node:
                result += output_port["data_type"] + "* "  + output_port["variable_name"] + "_buffer"  + ", "

    return result

# Kernel Launch parm declare
def kernelLaunchParmDeclareGenerator(jsonObj):
    buffer_dict = getInputBufferDict(jsonObj)
    result = ""
    global_input = jsonObj["global_input"] 
    for input_node_key in global_input:
        input_node = global_input[input_node_key]
        
        if ("volumevopglobal" in input_node_key) or ("geometryvopglobal" in input_node_key):
            for input_port in input_node:
                # check buffer type
                if input_port["is_buffer"] == "True":
                    result += "CGBuffer<" + input_port["data_type"] + ">" + "* " + input_port["variable_name"] + "buffer" + ", "  
                elif input_port["is_OpInput"] == "True":
                    result += "CGGeometry* " + input_port["variable_name"] + ", "  
                else:
                    result += input_port["data_type"] + " " + input_port["variable_name"] + ", " 
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
                    result += "CGBuffer<" + output_port["data_type"] + ">" + "* " + output_port["variable_name"] + "buffer" + ", "
        else:
            for output_port in output_node:
                result += "CGBuffer<" + output_port["data_type"] + ">" + "* "  + output_port["variable_name"] + "_buffer"  + ", "

    return result

# Share code parm input list
def shareCodeParmInputListGenerator(jsonObj):
    buffer_dict = getInputBufferDict(jsonObj)
    result = ""
    global_input = jsonObj["global_input"] 
    for input_node_key in global_input:
        input_node = global_input[input_node_key]
        
        if ("volumevopglobal" in input_node_key) or ("geometryvopglobal" in input_node_key):
            for input_port in input_node:
                # check buffer type
                if input_port["is_buffer"] == "True":
                    result += input_port["variable_name"] + "buffer" + ", "  
                else:
                    result += input_port["variable_name"] + ", "
        # for custom_param
        else:
            for input_port in input_node:
                result += input_port["variable_name"] + ", "
    
    global_output = jsonObj["global_output"]
    for output_node_key in global_output:
        output_node = global_output[output_node_key]
        
        if ("volumevopoutput1" in output_node_key) or ("geometryvopoutput" in output_node_key):
            for output_port in output_node:
                if not output_port["port_name"] in buffer_dict:
                    result += output_port["variable_name"] + "buffer" + ", "
        else:
            for output_port in output_node:
                result += output_port["variable_name"] + "_buffer"  + ", "

    return result

# Buffer malloc
def bufferMallocGenerator(jsonObj):
    buffer_dict = getInputBufferDict(jsonObj)
    buffer_list = []
    opInput_list = []
    result = ""
    global_input = jsonObj["global_input"] 
    for input_node_key in global_input:
        input_node = global_input[input_node_key]
        
        if ("volumevopglobal" in input_node_key) or ("geometryvopglobal" in input_node_key):
            for input_port in input_node:
                # add buffer to list
                if input_port["is_buffer"] == "True":
                    tmpName = input_port["variable_name"] + "buffer"
                    buffer_list.append(tmpName)
                elif input_port["is_OpInput"] == "True":
                    OpInput_list.append(input_port["variable_name"])
    
    global_output = jsonObj["global_output"]
    for output_node_key in global_output:
        output_node = global_output[output_node_key]
        
        if ("volumevopoutput1" in output_node_key) or ("geometryvopoutput" in output_node_key):
            for output_port in output_node:
                tmpName = output_port["variable_name"] + "buffer"
                # add buffer to list
                if not output_port["port_name"] in buffer_dict:
                    buffer_list.append(tmpName)
        else:
            for output_port in output_node:
                tmpName = output_port["variable_name"] + "_buffer"
                buffer_list.append(tmpName)

    for buffer in buffer_list:
        result += buffer + "->malloc();\n"
        result += buffer + "->loadHostToDevice();\n\n"

    return result

# Compute num of threads based on this buffer
def refBufferNameGenerator(jsonObj):
    # result = jsonObj["parallel_reference_name"]

    # global_input = jsonObj["global_input"] 
    # for input_node_key in global_input:
    #     input_node = global_input[input_node_key]
        
    #     if ("volumevopglobal" in input_node_key) or ("geometryvopglobal" in input_node_key):
    #         for input_port in input_node:
    #             # check buffer type
    #             if (input_port["variable_name"] == result) and (input_port["is_buffer"] == "True"):
    #                 result += "buffer"
    
    result = jsonObj["parallel_reference_name"]

    if jsonObj["parallel_reference"] == "volume":
        result += "->velField->GetNumVoxels();"
    else:
        result += "buffer->getSize();"

    return result

# Kernel parm input list
def kernelParmInputListGenerator(jsonObj):  
    buffer_dict = getInputBufferDict(jsonObj)
    result = ""
    global_input = jsonObj["global_input"] 
    for input_node_key in global_input:
        input_node = global_input[input_node_key]
        
        if ("volumevopglobal" in input_node_key) or ("geometryvopglobal" in input_node_key):
            for input_port in input_node:
                # check buffer type
                if input_port["is_buffer"] == "True":
                    result += input_port["variable_name"] + "buffer" + "->getDevicePointer(), "  
                elif input_port["is_OpInput"] == "True":
                    result += input_port["variable_name"] + "->GetGeometryRawDataDevice(), "  
                else:
                    result += input_port["variable_name"] + ", "
        # for custom_param
        else:
            for input_port in input_node:
                result += input_port["variable_name"] + ", "
    
    global_output = jsonObj["global_output"]
    for output_node_key in global_output:
        output_node = global_output[output_node_key]
        
        if ("volumevopoutput1" in output_node_key) or ("geometryvopoutput" in output_node_key):
            for output_port in output_node:
                if not output_port["port_name"] in buffer_dict:
                    result += output_port["variable_name"] + "buffer" + "->getDevicePointer(), " 
        else:
            for output_port in output_node:
                result += output_port["variable_name"] + "_buffer"  + "->getDevicePointer(), " 

    return result

# Kernel Launch error msg
def kernelLaunchErrorMsgGenerator(jsonObj):  
    return jsonObj["error_msg"]

# ----- Code Segment Map -----
replacementMap = {
    "OUT_FILE_NAME":fileNameGenerator,
    "KERNEL_NAME":kernelNameGenerator,
    "FUNC_NAME":funcNameGenerator,
    "KERNEL_LAUNCH_NAME":kernelLaunchNameGenerator,

    "KERNEL_PARM_DECLARE_LIST": kernelParmDeclareGenerator,
    "KERNEL_LAUNCH_PARM_DECLARE_LIST": kernelLaunchParmDeclareGenerator,
    "SHARE_CODE_PARM_INPUT_LIST":shareCodeParmInputListGenerator,

    "BUFFER_MALLOC":bufferMallocGenerator,
    "REF_BUFFER_NAME": refBufferNameGenerator,
    "KERNEL_PARM_INPUT_LIST": kernelParmInputListGenerator,

    "KERNEL_LAUNCH_ERROR_MSG":kernelLaunchErrorMsgGenerator
}

# ----- Kernel/Kernel Launch Code generate -----

this_node = hou.pwd()

# Load dag_son
#geo = this_node.geometry()
#json_str = geo.findGlobalAttrib("kernel_launch_json").strings()
#json_str = json_str[0]
#jsonObj = json.loads(json_str)
json_input = this_node.input(2)
json_str = json_input.parm("/obj/geo1/CUDA_JSON/content").eval()
jsonObj = json.loads(json_str)

# Read in the template
with open('./Template/CUDAKernelTemplate_cuh.h', 'r') as file :
  filedata_cuh = file.read()

with open('./Template/CUDAKernelTemplate_cu.h', 'r') as file :
  filedata_cu = file.read()


# Parse keywords from template
code_segments_cuh = re.findall('@(.+?)@', filedata_cuh) 
code_segments_cu = re.findall('@(.+?)@', filedata_cu) 

# Replace the target string 
for code_segment in code_segments_cuh:
    if code_segment in replacementMap:
        tmp_str = "@" +  code_segment + "@"
        target_code = replacementMap[code_segment](jsonObj)
        filedata_cuh = filedata_cuh.replace(tmp_str, target_code)
        
for code_segment in code_segments_cu:
    if code_segment in replacementMap:
        tmp_str = "@" +  code_segment + "@"
        target_code = replacementMap[code_segment](jsonObj)
        filedata_cu = filedata_cu.replace(tmp_str, target_code)

# Output CUDA to file
file_name_cuh = "./Code_VEL/" + fileNameGenerator(jsonObj=jsonObj) + ".cuh"
with open(file_name_cuh, 'w') as file:
  file.write(filedata_cuh) 

file_name_cu = "./Code_VEL/" + fileNameGenerator(jsonObj=jsonObj) + ".cu"
with open(file_name_cu, 'w') as file:
  file.write(filedata_cu) 