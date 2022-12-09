import json

# get the current python node
this_node = hou.pwd()
geo = this_node.geometry()
# get the vop node
vop = this_node.input(1)

kernel_launch_json_dict = dict()

# "//comment" key value
comment_dict = { \
        "#OUT_FILE_NAME#": "out_filename", \
        "#KERNEL_NAME#" : "particleAdvect", \
        "#KERNEL_PARM_DECLARE_LIST#" : ["global_input", "global_output"], \
        "#FUNC_NAME#" : "func_name", \
        "#SHARE_CODE_PARM_INPUT_LIST#" : ["global_input", "global_output"], \
        "#KERNEL_LAUNCH_NAME#" : "kernel_launch_name", \
        "#KERNEL_LAUNCH_PARM_DECLARE_LIST#" : ["global_input", "global_output"], \
        "#BUFFER_MALLOC#": ["global_input", "global_output"], \
        "#REF_BUFFER_NAME#": "ref_buffer_name", \
        "#KERNEL_PARM_INPUT_LIST#" : ["global_input", "global_output"], \
        "#KERNEL_LAUNCH_ERROR_MSG#": "error_msg" \
        } 
kernel_launch_json_dict["//comment"] = comment_dict

# "out_filename" key value
#kernel_launch_json_dict["proj_name"] = "SimpleParticle"
kernel_launch_json_dict["proj_name"] = vop.name().replace("_", "")

# "kernel_launch_name" "kernel_name" "func_name"
function_name = vop.name()
kernel_launch_json_dict["kernel_launch_name"] = function_name
kernel_launch_json_dict["kernel_name"] = function_name
kernel_launch_json_dict["func_name"] = function_name

# "error_msg" key value
kernel_launch_json_dict["error_msg"] = function_name + " error"


dag_json_strs = geo.findGlobalAttrib("dag_json").strings()
dag_json_str = dag_json_strs[0]
dag_json_obj = json.loads(dag_json_str)

# "ref_buffer_name" key value
global_input_value = dag_json_obj["global_input"]
for key in global_input_value.keys():
    if key.find("volumevopglobal") != -1:
        parallel_reference = "volume"
        parallel_reference_name = global_input_value[key][0]["variable_name"]
    if key.find("geometryvopglobal") != -1:
        parallel_reference = "point"
        parallel_reference_name = global_input_value[key][0]["variable_name"]
kernel_launch_json_dict["parallel_reference"] = parallel_reference
kernel_launch_json_dict["parallel_reference_name"] = parallel_reference_name

# "global_input" key value
global_input_dict = dag_json_obj["global_input"]
kernel_launch_json_dict["global_input"] = global_input_dict

# "global_output" key value
global_output_dict = dag_json_obj["global_output"]
kernel_launch_json_dict["global_output"] = global_output_dict

json_str = json.dumps(kernel_launch_json_dict)
geo.addAttrib(hou.attribType.Global, "cpu_json", "")
geo.setGlobalAttribValue("cpu_json", json_str)