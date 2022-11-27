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
        
        if input_node_key == "geometryvopglobal1":
            for input_port in input_node:
                if input_port["is_buffer"] == "True":
                    buffer_dict[input_port["port_name"]] = input_port["variable_name"]
    return buffer_dict

# ----- Code Segment Generators -----

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
        
        if input_node_key == "geometryvopglobal1":
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
        
        if output_node_key == "geometryvopoutput1":
            for output_port in output_node:
                # param in "geometryvopoutput1" are all buffer data
                # we don't won't duplicate param
                # only add the param if not in global input
                if not output_port["port_name"] in buffer_dict:
                    result += "CGBuffer<" + output_port["data_type"] + ">* " + output_port["variable_name"] + "buffer" + ", "
        else:
            for output_port in output_node:
                result += "CGBuffer<" + output_port["data_type"] + ">" + "* " + output_port["variable_name"] + "_buffer"  + ", "

    return result[:-2]
  
def projNameGenerator(jsonObj):
    return jsonObj["proj_name"]

# ----- Code Segment Map -----
replacementMap = {
    "PROJ_NAME":projNameGenerator,
    "FUNC_NAME":funcNameGenerator,
    "FUNC_DECLARE_LIST":parmListGenerator
}


this_node = hou.pwd()

# Load cpu_json
geo = this_node.geometry()
json_str = geo.findGlobalAttrib("cpu_json").strings()
json_str = json_str[0]
jsonObj = json.loads(json_str)

# Parse keywords from template
template_input = this_node.input(1)
template = template_input.parm("/obj/geo1/solver1/d/s/cpu_template/content").eval()
code_segments = re.findall('@(.+?)@', template)

# Read in the template
with open('./Template/CPUTemplate_h.h', 'r') as file :
    filedata = file.read()

# Replace the target string 
for code_segment in code_segments:
    if code_segment in replacementMap:
        tmp_str = "@" +  code_segment + "@"
        target_code = replacementMap[code_segment](jsonObj)
        
        filedata = filedata.replace(tmp_str, target_code)
        
# Output cpu header to file
file_name = "./Code/" + projNameGenerator(jsonObj=jsonObj) + ".h"
with open(file_name, 'w') as file:
    file.write(filedata)   
  
  
