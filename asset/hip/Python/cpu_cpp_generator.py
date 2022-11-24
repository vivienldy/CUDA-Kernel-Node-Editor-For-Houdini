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
    
# Fucntion name
def funcNameGenerator(jsonObj):
    return jsonObj["func_name"]

# Parm list
def parmListGenerator(jsonObj):
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
  
def projNameGenerator(jsonObj):
    return jsonObj["proj_name"]

def rawDataGenerator(jsonObj):
    result = ""
    return result

def shareCodeParamGenerator(jsonObj):
    buffer_list = []
    result = ""
    global_input = jsonObj["global_input"] 
    for input_node_key in global_input:
        input_node = global_input[input_node_key]
        
        if input_node_key == "geometryvopglobal1":
            for input_port in input_node:
                # check buffer type
                if input_port["is_buffer"] == "True":
                    result += input_port["variable_name"] + "->getRawData()" + ", "  #i.e. pos->getRawData(),
                    buffer_list.append(input_port["variable_name"])
                else:
                    result += input_port["variable_name"] + ", "  #i.e. dt,
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
                    result += output_port["variable_name"] + "->getRawData()" + ", "
        else:
            for output_port in output_node:
                result += output_port["variable_name"] + "->getRawData()"  + ", "

    return result

def getNumThreadGenerator(jsonObj):
    result = ""

    global_input = jsonObj["global_input"]
    for input_node_key in global_input:
        input_node = global_input[input_node_key]
        
        if input_node_key == "geometryvopglobal1":
            for input_port in input_node:
                # check buffer type
                if input_port["is_buffer"] == "True":
                    resutl += input_port["variable_name"] + "->getSize();"
                    return result

    return result

# ----- Code Segment Map -----
replacementMap = {
    "PROJ_NAME":projNameGenerator,
    "FUNC_NAME":funcNameGenerator,
    "FUNC_DECLARE_LIST":parmListGenerator,
    "GET_RAWDATA":rawDataGenerator,
    "SHARE_CODE_PARAM":shareCodeParamGenerator,
    "GET_NUM_THREAD":getNumThreadGenerator
}


# ----- Share Code generate -----

if __name__ == '__main__':
    # this_node = hou.pwd()

    # Load dag_son
    # geo = this_node.geometry()
    # json_str = geo.findGlobalAttrib("dag_json").strings()
    # json_str = json_str[0]
    # jsonObj = json.loads(json_str)

    # Parse keywords from template
    # template_input = this_node.input(1)
    # template = template_input.parm("/obj/geo1/solver1/d/s/share_code_file_template/content").eval()
    # code_segments = re.findall('#(.+?)#', template)

    # Read in the template
    with open('./Template/CPUTemplate.h', 'r') as file :
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
  
  