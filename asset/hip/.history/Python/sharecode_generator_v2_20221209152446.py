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

# ----- Method Generators -----

# Add
def addMethodGenerator(node):
    cd = Codeline()
    input_len = len(node["input"])
    for port in node["input"]:
        if node["input"].index(port) == input_len -1:
            cd.right += port["local_input_name"]
        else:
            cd.right += port["local_input_name"] + " + "

    for port in node["output"]:
        cd.left  = port["data_type"] + " " + port["local_output_name"]
    return cd.toCode()

# Subtract
def subtractMethodGenerator(node):
    cd = Codeline()
    input_len = len(node["input"])
    for port in node["input"]:
        if node["input"].index(port) == input_len - 1:
            cd.right += port["local_input_name"]
        else:
            cd.right += port["local_input_name"] + " - "

    for port in node["output"]:
        cd.left  = port["data_type"] + " " + port["local_output_name"]
    return cd.toCode()

# Negate
def negateMethodGenerator(node):
    cd = Codeline()
    for port in node["input"]:
        cd.right = " - " + port["local_input_name"] 

    for port in node["output"]:
        cd.left  = port["data_type"] + " " + port["local_output_name"]
    return cd.toCode()

# Complement
def complementMethodGenerator(node):
    cd = Codeline()
    for port in node["input"]:
        cd.right = " 1 - " + port["local_input_name"] 

    for port in node["output"]:
        cd.left  = port["data_type"] + " " + port["local_output_name"]
    return cd.toCode()

# Multiply
def multiplyMethodGenerator(node):
    cd = Codeline()
    input_len = len(node["input"])
    for port in node["input"]:
        if node["input"].index(port) == input_len -1:
            cd.right += port["local_input_name"]
        else:
            cd.right += port["local_input_name"] + " * "
            
    for port in node["output"]:
        cd.left  = port["data_type"] + " " + port["local_output_name"]
    return cd.toCode()

# Constant Node
def constantMethodGenerator(node):
    cd = Codeline()
    cd.left  = node["output"][0]["data_type"] + " " + node["output"][0]["local_output_name"]
    cd.right = node["output"][0]["data_type"] + "(" + node["input"][0]["local_input_name"] + ")"
    return cd.toCode()

# inlineCode Node
def inlineCodeMethodGenerator(node):
    return node["content"] + "\n"

# General Case 

def generalGenerator(node):
    result = ""
    
    # no output, i.e. settovoxel
    if len(node["output"]) == 0:
        for port in node["input"]:
            if node["input"].index(port) == input_len -1:
                cd.right += port["local_input_name"]
            else:
                cd.right += port["local_input_name"] + ", "

            
    if "multi_outputs" in node:
        for port in node["multi_outputs"]:
            result += port["data_type"] + " " + port["local_output_name"] + ";\n"

    cd = Codeline()
    input_len = len(node["input"])
    
    cd.right += node["method_name"] + "("

    for port in node["input"]:
        if not port["local_input_name"] == "CG_NONE":
            if node["input"].index(port) == input_len -1:
                cd.right += port["local_input_name"]
            else:
                cd.right += port["local_input_name"] + ", "
        else:
            if node["input"].index(port) == input_len -1:
                cd.right += port["data_type"] + "(" + port["default_value"] + ")"
            else:
                cd.right += port["data_type"] + "(" + port["default_value"] + ")" + ", "

    if "multi_outputs" in node:
        cd.right += ", "
        for port in node["multi_outputs"]:
            if node["multi_outputs"].index(port) == len(node["multi_outputs"]) - 1:
                cd.right += "&" + port["local_output_name"]
            else:
                cd.right += "&" + port["local_output_name"] + ", "
        cd.right += ")"
        result += cd.right + ";\n"
    else:
        cd.right += ")"
        for port in node["output"]:
            cd.left  = port["data_type"] + " " + port["local_output_name"]
        result += cd.toCode()
    
    return result

# ----- Method Map -----
customCodeGeneratorMap = {
    "multiply":multiplyMethodGenerator,
    "add":addMethodGenerator,
    "subtract":subtractMethodGenerator,
    "constant":constantMethodGenerator,
    "negate":negateMethodGenerator,
    "complement":complementMethodGenerator,
    "inlineCode":inlineCodeMethodGenerator
    }
    
# ----- Code Segment Generators -----

# Fucntion name
def funcNameGenerator(jsonObj):
    return jsonObj["func_name"]

# Parm list
def ParmListGenerator(jsonObj):
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
                    result += "CGGeometry::RawData " + input_port["variable_name"] + ", " 
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
                # param in "volumevopoutput1" are all buffer data
                # we don't won't duplicate param
                # only add the param if not in global input
                if not output_port["port_name"] in buffer_dict:
                    result += output_port["data_type"] + "* " + output_port["variable_name"] + "buffer" + ", "
        else:
            for output_port in output_node:
                if not len(output_port) == 0:
                    result += output_port["data_type"] + "* "  + output_port["variable_name"] + "_buffer"  + ", "
    
    result += "int idx"

    return result

# Data Load
def dataLoadGenerator(jsonObj):
    result = ""  
    
    global_input = jsonObj["global_input"] 
    for input_node_key in global_input:
        input_node = global_input[input_node_key]
        
        if ("volumevopglobal" in input_node_key) or ("geometryvopglobal" in input_node_key):
            result += "// Geometry Global Input\n"   
            for input_port in input_node:
                cd = Codeline()
                cd.left = input_port["data_type"] + " " + input_port["variable_name"]; # vec3 p
                if input_port["is_buffer"] == "True":
                    cd.right = input_port["variable_name"] + "buffer" + "[idx]"; # i.e., pbuffer[idx];
                    result += cd.toCode() # i.e.: vec3 p = pbuffer[idx]
    
    return result
    
# Compute graph
def computeGraphGenerator(jsonObj):
    result = ""
    
    compute_graph = jsonObj["compute_graph"]
    for node_key in compute_graph:
        result += "\n // Generate by " + node_key + "\n"
        node = compute_graph[node_key]
        if node["method_name"] in customCodeGeneratorMap:
            # function pointer
            result += customCodeGeneratorMap[node["method_name"]](node)
        else:
            result += generalGenerator(node)
    return result

# Write back
def writeBackGenerator(jsonObj):
    result = ""
    buffer_dict = getInputBufferDict(jsonObj)

    global_output = jsonObj["global_output"]
    for output_node_key in global_output:
        output_node = global_output[output_node_key]
        if ("volumevopoutput1" in output_node_key) or ("geometryvopoutput" in output_node_key):
            for output_port in output_node:
                cd = Codeline()
                cd.left = output_port["data_type"] + " " + "global_output_" + output_port["variable_name"] # i.e., vec3 global_output_p
                cd.right = output_port["connection"]# i.e., add
                result += cd.toCode()
                if output_port["port_name"] in buffer_dict:
                    cd.left = buffer_dict[output_port["port_name"]] + "buffer[idx]"
                else:
                    cd.left = output_port["variable_name"] + "buffer[idx]"
                cd.right = "global_output_" + output_port["variable_name"] 
                result += cd.toCode()
                result += "\n"
        else:
            for output_port in output_node:
                cd = Codeline()
                cd.left = output_port["data_type"] + " " + output_port["variable_name"] # i.e., vec3 global_output_p
                cd.right = output_port["connection"]# i.e., add
                result += cd.toCode()
                
                cd.left = output_port["variable_name"] + "_buffer[idx]" # i.e., pbuffer[idx]
                cd.right = output_port["variable_name"] # i.e., global_output_p
                result += cd.toCode()
                result += "\n"
    
    return result
    
# ----- Code Segment Map -----
replacementMap = {
    "FUNC_NAME":funcNameGenerator,
    "PARM_LIST":ParmListGenerator,
    "DATA_LOAD":dataLoadGenerator,
    "COMPUTE_GRAPH":computeGraphGenerator,
    "WRITE_BACK":writeBackGenerator
}


# ----- Share Code generate -----

this_node = hou.pwd()

# Load dag_son
#geo = this_node.geometry()
#json_str = geo.findGlobalAttrib("dag_json").strings()
json_input = this_node.input(2)
json_str = json_input.parm("/obj/geo1/json_file/content").eval()
jsonObj = json.loads(json_str)

# Read in the template
with open('./Template/ShareCodeTemplate.h', 'r') as file :
  filedata = file.read()

# Parse keywords from template
code_segments = re.findall('@(.+?)@', filedata) 
  
# Replace the target string 
for code_segment in code_segments:
    if code_segment in replacementMap:
        tmp_str = "@" +  code_segment + "@"
        target_code = replacementMap[code_segment](jsonObj)
        filedata = filedata.replace(tmp_str, target_code)

# Output sharecode to file
file_name = "./Code_VEL/" + funcNameGenerator(jsonObj=jsonObj) + ".GenericCode.h"
with open(file_name, 'w') as file:
  file.write(filedata) 

  
