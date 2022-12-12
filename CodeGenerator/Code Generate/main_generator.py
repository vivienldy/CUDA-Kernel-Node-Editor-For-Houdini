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
    return jsonObj["TaskInfo"]["func_name"]

# Parm list
def parmListGenerator(jsonObj):
    result = ""

    gBuffers = jsonObj["TaskInfo"]["BufferList"]
    params = jsonObj["OperatorList"]["Operator"]["Parameters"]

    for param in params:
        if param["is_buffer"] == "False":
            result += param["variable_name"] + ", "
    
    for buffer in gBuffers:
        result += buffer["variable_name"] + "buffer, "
    
    for param in params:
        if param["is_buffer"] == "True":
            result += param["variable_name"] + "_buffer, "

    return result[:-2]
  
def projNameGenerator(jsonObj):
    return jsonObj["TaskInfo"]["proj_name"]

def getSizeGenerator(jsonObj):
    result = ""

    buffers = jsonObj["TaskInfo"]["BufferList"]
    for buffer in buffers:
        if buffer["filename"] != "None":
            result += buffer["variable_name"] + "buffer->getSize(); \n"

            return result
    
    return result

def globalInitLoadGenerator(jsonObj):
    result = ""

    buffers = jsonObj["TaskInfo"]["BufferList"]
    for buffer in buffers:
        if buffer["filename"] != "None":
            result += "auto " + buffer["variable_name"] + "buffer = "
            result += "dynamic_cast<CGBuffer<" + buffer["type"] + ">*>(CGBuffer<float>::"
            result += "loadFromFile(\"" + buffer["filename"] + "\")); \n"
    
    return result

def globalInitGenerator(jsonObj):
    result = ""

    buffers = jsonObj["TaskInfo"]["BufferList"]
    for buffer in buffers:
        if len(buffer["value"]) != 0:
            result += "auto " + buffer["variable_name"] + "buffer = "
            result += classInit(buffer["type"], buffer["value"], buffer["variable_name"]) + "; \n"

    return result

def customInitGenerator(jsonObj):
    result = ""

    params = jsonObj["OperatorList"]["Operator"]["Parameters"]

    for param in params:
        if param["is_buffer"] == "True":
            result += "auto " + param["variable_name"] + "_buffer = "
            
            result += classInit(param["type"], param["value"], param["variable_name"]) + "; \n\t"

        else:
            result += param["type"] + " " + param["variable_name"] + " = "

            if len(param["value"]) != 0:
                result += parseValue(param["type"], param["value"])
            
            result += "; \n\t"

    return result

def paramInitGenerator(jsonObj):
    result = ""

    info = jsonObj["TaskInfo"]

    result += "int startFrame = " + str(info["StartFrame"]) + "; \n\t"
    result += "int endFrame = " + str(info["EndFrame"]) + "; \n\t"
    result += "float FPS = " + str(info["FPS"]) + "; \n\t"
    result += "int blockSize = " + str(info["BlockSize"]) + "; \n\t"

    return result

def globalLoadToHostGenerator(jsonObj):
    result = ""

    buffers = jsonObj["TaskInfo"]["BufferList"]
    for buffer in buffers:
        result += buffer["variable_name"] + "buffer->loadDeviceToHost(); \n\t\t"
    
    return result

def globalLoadToObjGenerator(jsonObj):
    result = ""

    buffers = jsonObj["TaskInfo"]["BufferList"]
    for buffer in buffers:
        if buffer["filename"] != "None":
            result += buffer["variable_name"] + "buffer->outputObj(outputObjFilePath); \n"
    
    return result

def classInit(type, value, name):
    result = ""
    
    result += "new CGBuffer<" + type + ">(\"" + name + "\", numPoints"

    if len(value) != 0:
        result += ", " + parseValue(type, value)
    
    result += ")"

    return result

def parseValue(type, value):
    result = ""

    if type == 'int':
        result += str(value[0])
    elif type == 'float':
        result += str(value[0])
    elif type == 'glm::vec3':
        if len(value) == 1:
            result += "glm::vec3(" + str(value[0]) + ")"
        else:
            result += "glm::vec3(" + str(value[0]) + ", " + str(value[1]) + ", " + str(value[2]) + ")"

    return result

# ----- Code Segment Map -----
replacementMap = {
    "PROJ_NAME":projNameGenerator, #
    "FUNC_NAME":funcNameGenerator, #
    "GLOBAL_INIT_LOAD":globalInitLoadGenerator, #
    "GLOBAL_GET_SIZE":getSizeGenerator, # 
    "GLOBAL_INIT":globalInitGenerator, #
    "CUSTOM_INIT":customInitGenerator, #
    "PARAM_INIT":paramInitGenerator, #
    "FUNC_PARAM_LIST":parmListGenerator, #  
    "GLOABL_LOAD_TO_HOST":globalLoadToHostGenerator, # 
    "GLOBAL_LOAD_TO_OBJ":globalLoadToObjGenerator
}


# ----- Share Code generate -----

if __name__ == '__main__':
    # this_node = hou.pwd()

    # # Load dag_son
    # geo = this_node.geometry()
    # json_str = geo.findGlobalAttrib("dag_json").strings()
    # json_str = json_str[0]
    json_str = "asset/hip/tmpJSON/mainJSON.json"
    f = open(json_str, 'r', encoding='utf-8')
    jsonObj = json.load(f)

    # Parse keywords from template
    # template_input = this_node.input(1)
    # template = template_input.parm("/obj/geo1/solver1/d/s/share_code_file_template/content").eval()
    # code_segments = re.findall('#(.+?)#', template)

    code_segments = ["PROJ_NAME", 
    "FUNC_NAME", #
    "GLOBAL_INIT_LOAD", #
    "GLOBAL_GET_SIZE", # 
    "GLOBAL_INIT", #
    "CUSTOM_INIT",
    "PARAM_INIT",
    "FUNC_PARAM_LIST",
    "GLOABL_LOAD_TO_HOST",
    "GLOBAL_LOAD_TO_OBJ"]

    # Read in the template
    with open('asset/hip/Template/mainTemplate.h', 'r') as file :
        filedata = file.read()

    # Replace the target string 
    for code_segment in code_segments:
        if code_segment in replacementMap:
            tmp_str = "@" +  code_segment + "@"
            target_code = replacementMap[code_segment](jsonObj)
            filedata = filedata.replace(tmp_str, target_code)

    print(filedata)

    # Output sharecode to file
    file_name = "asset/hip/Code/main.cpp"
    with open(file_name, 'w') as file:
        file.write(filedata)   
  
  
