import re
import json

this_node = hou.pwd()

# Load dag_son
geo = this_node.geometry()
json_str = geo.findGlobalAttrib("kernel_launch_json").strings()
json_str = json_str[0]
jsonObj = json.loads(json_str)

# Parse keywords from template
template_input = this_node.input(2)
template = template_input.parm("/obj/geo1/solver1/d/s/share_code_file_template/content").eval()
code_segments = re.findall('#(.+?)#', template)

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
    buffer_list = []
    result = ""
    global_input = jsonObj["global_input"] 
    for input_node_key in global_input:
        input_node = global_input[input_node_key]
        
        if input_node_key == "geometryvopglobal1":
            for input_port in input_node:
                # check buffer type
                if input_port["is_buffer"] == "True":
                    result += input_port["data_type"] + "* " + input_port["variable_name"] + "buffer" + ", "  #i.e. glm::vec3* posBuffer,
                    buffer_list.append(input_port["variable_name"])
                else:
                    result += input_port["data_type"] + " " + input_port["variable_name"] + ", "  #i.e. float dt,
        # for custom_param
        else:
            for input_port in input_node:
                result += input_port["data_type"] + " " + input_port["variable_name"] + ", "
    
    global_output = jsonObj["global_output"]
    for output_node_key in global_output:
        output_node = global_output[output_node_key]
        
        if output_node_key == "geometryvopoutput1":
            for output_port in output_node:
                # param in "geometryvopoutput1" are all buffer data
                # we don't won't duplicate param
                # only add the param if not in global input
                if buffer_list.count(output_port["variable_name"]) == 0:
                    result += output_port["data_type"] + "* " + output_port["variable_name"] + "buffer" + ", "
        else:
            for output_port in output_node:
                result += output_port["data_type"] + "* "  + output_port["variable_name"] + "_buffer"  + ", "

    return result

# Kernel Launch parm declare
def kernelLaunchParmDeclareGenerator(jsonObj):
    buffer_list = []
    result = ""
    global_input = jsonObj["global_input"] 
    for input_node_key in global_input:
        input_node = global_input[input_node_key]
        
        if input_node_key == "geometryvopglobal1":
            for input_port in input_node:
                # check buffer type
                if input_port["is_buffer"] == "True":
                    result += "CGBuffer<" + input_port["data_type"] + ">" + "* " + input_port["variable_name"] + "buffer" + ", "  
                    buffer_list.append(input_port["variable_name"])
                else:
                    result += input_port["data_type"] + " " + input_port["variable_name"] + ", " 
        # for custom_param
        else:
            for input_port in input_node:
                result += input_port["data_type"] + " " + input_port["variable_name"] + ", "
    
    global_output = jsonObj["global_output"]
    for output_node_key in global_output:
        output_node = global_output[output_node_key]
        
        if output_node_key == "geometryvopoutput1":
            for output_port in output_node:
                # param in "geometryvopoutput1" are all buffer data
                # we don't won't duplicate param
                # only add the param if not in global input
                if buffer_list.count(output_port["variable_name"]) == 0:
                    result += "CGBuffer<" + output_port["data_type"] + ">" + "* " + output_port["variable_name"] + "buffer" + ", "
        else:
            for output_port in output_node:
                result += "CGBuffer<" + output_port["data_type"] + ">" + "* "  + output_port["variable_name"] + "_buffer"  + ", "

    return result

# Share code parm input list
def shareCodeParmInputListGenerator(jsonObj):
    buffer_list = []
    result = ""
    global_input = jsonObj["global_input"] 
    for input_node_key in global_input:
        input_node = global_input[input_node_key]
        
        if input_node_key == "geometryvopglobal1":
            for input_port in input_node:
                # check buffer type
                if input_port["is_buffer"] == "True":
                    result += input_port["variable_name"] + "buffer" + ", "  
                    buffer_list.append(input_port["variable_name"])
                else:
                    result += input_port["variable_name"] + ", "
        # for custom_param
        else:
            for input_port in input_node:
                result += input_port["variable_name"] + ", "
    
    global_output = jsonObj["global_output"]
    for output_node_key in global_output:
        output_node = global_output[output_node_key]
        
        if output_node_key == "geometryvopoutput1":
            for output_port in output_node:
                # param in "geometryvopoutput1" are all buffer data
                # we don't won't duplicate param
                # only add the param if not in global input
                if buffer_list.count(output_port["variable_name"]) == 0:
                    result += output_port["variable_name"] + "buffer" + ", "
        else:
            for output_port in output_node:
                result += output_port["variable_name"] + "_buffer"  + ", "

    return result

# Buffer malloc
def bufferMallocGenerator(jsonObj):
    buffer_list = []
    result = ""
    global_input = jsonObj["global_input"] 
    for input_node_key in global_input:
        input_node = global_input[input_node_key]
        
        if input_node_key == "geometryvopglobal1":
            for input_port in input_node:
                # add buffer to list
                if input_port["is_buffer"] == "True":
                    tmpName = input_port["variable_name"] + "buffer"
                    buffer_list.append(tmpName)
    
    global_output = jsonObj["global_output"]
    for output_node_key in global_output:
        output_node = global_output[output_node_key]
        
        if output_node_key == "geometryvopoutput1":
            for output_port in output_node:
                tmpName = output_port["variable_name"] + "buffer"
                # add buffer to list
                if buffer_list.count(tmpName) == 0:
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
    return jsonObj["ref_buffer_name"] + "buffer"

# Kernel parm input list
def kernelParmInputListGenerator(jsonObj):  
    buffer_list = []
    result = ""
    global_input = jsonObj["global_input"] 
    for input_node_key in global_input:
        input_node = global_input[input_node_key]
        
        if input_node_key == "geometryvopglobal1":
            for input_port in input_node:
                # check buffer type
                if input_port["is_buffer"] == "True":
                    result += input_port["variable_name"] + "buffer" + "->getDevicePointer(), "  
                    buffer_list.append(input_port["variable_name"])
                else:
                    result += input_port["variable_name"] + ", "
        # for custom_param
        else:
            for input_port in input_node:
                result += input_port["variable_name"] + ", "
    
    global_output = jsonObj["global_output"]
    for output_node_key in global_output:
        output_node = global_output[output_node_key]
        
        if output_node_key == "geometryvopoutput1":
            for output_port in output_node:
                # param in "geometryvopoutput1" are all buffer data
                # we don't won't duplicate param
                # only add the param if not in global input
                if buffer_list.count(output_port["variable_name"]) == 0:
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
    "KERNEL_PARM_INPUT_LIST", kernelParmInputListGenerator,

    "KERNEL_LAUNCH_ERROR_MSG":kernelLaunchErrorMsgGenerator
}


# ----- Kernel/Kernel Launch Code generate -----

# Read in the template
with open('./Template/CUDAKernelTemplate.h', 'r') as file :
  filedata = file.read()

# Replace the target string 
for code_segment in code_segments:
    if code_segment in replacementMap:
        tmp_str = "#" +  code_segment + "#"
        target_code = replacementMap[code_segment](jsonObj)
        filedata = filedata.replace(tmp_str, target_code)

# Output sharecode to file
with open('./Code/SimpleParticle.cu', 'w') as file:
  file.write(filedata) 