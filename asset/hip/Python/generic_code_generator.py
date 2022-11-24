import re
import json

this_node = hou.pwd()

# Load dag_son
geo = this_node.geometry()
json_str = geo.findGlobalAttrib("dag_json").strings()
json_str = json_str[0]
jsonObj = json.loads(json_str)

# Parse keywords from template
template_input = this_node.input(1)
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
    for port in node["input"]:
        cd.right += port["local_input_name"]
    for port in node["output"]:
        cd.left  = port["data_type"] + " " + port["local_output_name"]
    return cd.toCode()
    
# General Case 
def generalGenerator(node):
    cd = Codeline()
    input_len = len(node["input"])
    
    cd.right += node["method_name"] + "("
    for port in node["input"]:
        if node["input"].index(port) == input_len -1:
            cd.right += port["local_input_name"]
        else:
            cd.right += port["local_input_name"] + ", "
    cd.right += ")"
    for port in node["output"]:
        cd.left  = port["data_type"] + " " + port["local_output_name"]
    return cd.toCode() 

# ----- Method Map -----
customCodeGeneratorMap = {
    "multiply":multiplyMethodGenerator,
    "add":addMethodGenerator,
    "constant":constantMethodGenerator
    }
    

# ----- Code Segment Generators -----

# Fucntion name
def funcNameGenerator(jsonObj):
    return jsonObj["func_name"]

# Parm list
def ParmListGenerator(jsonObj):
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
    
    result += "int idx"

    return result

# Data Load
def dataLoadGenerator(jsonObj):
    result = ""  
    
    global_input = jsonObj["global_input"] 
    for input_node_key in global_input:
        input_node = global_input[input_node_key]
        
        if input_node_key == "geometryvopglobal1":
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
    
    global_output = jsonObj["global_output"]
    for output_node_key in global_output:
        output_node = global_output[output_node_key]
        if output_node_key == "geometryvopoutput1":
            for output_port in output_node:
                cd = Codeline()
                cd.left = output_port["data_type"] + " " + "global_output_" + output_port["variable_name"] # i.e., vec3 global_output_p
                cd.right = output_port["connection"]# i.e., add
                result += cd.toCode()
                
                cd.left = output_port["variable_name"] + "buffer[idx]" # i.e., pbuffer[idx]
                cd.right = "global_output_" + output_port["variable_name"] # i.e., global_output_p
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

# Read in the template
with open('./Template/ShareCodeTemplate.h', 'r') as file :
  filedata = file.read()

# Replace the target string 
for code_segment in code_segments:
    if code_segment in replacementMap:
        tmp_str = "#" +  code_segment + "#"
        target_code = replacementMap[code_segment](jsonObj)
        filedata = filedata.replace(tmp_str, target_code)

# Output sharecode to file
with open('./Code/ShareCode.h', 'w') as file:
  file.write(filedata)   
  
  
