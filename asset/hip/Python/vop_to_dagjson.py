class Port:
    def __init__(self):
        self.connected_node = '' # connect to which node                       
        self.connected_index = -1 # index of the port on that node
        self.port_name = ''
        self.port_index = -1
        self.data_type = ''
        self.extra_data = '' # for extradata like constant value
        self.default_value = "CG_NONE"

class Node:
    def __init__(self):
        self.id = -1 # for topology sorted
        self.method_name = ''
        self.node_name = ''
        self.node_name_digit = -1
        self.node_path = ""
        self.inputs = [] # list of Port
        self.outputs = [] # list of Port
        self.raw_output_names = [] # all the output names on the node, no matter if connected
        self.raw_output_types = [] # all the output types on the node, no matter if connected
        
def dfs(node_id, is_node_visited_list, visit_nodes_list, dag_nodes_list):
    global node_name_id_dict
    is_node_visited_list[node_id] = True
    dag_node = dag_nodes_list[node_id]
    # edges = graph.getEdgesOutFromNode(nodeId)
    # get all the connected nodes id
    connected_nodes_list = []
    for output_node in dag_node.outputs:
        id = node_name_id_dict[output_node.connected_node]
        connected_nodes_list.append(id)
    for connected_node_id in connected_nodes_list:
        if not is_node_visited_list[connected_node_id]:
            dfs(connected_node_id, is_node_visited_list, visit_nodes_list, dag_nodes_list)
    visit_nodes_list.append(dag_node)

def is_buffer(port_name):
    global buffer_param_name_list
    for name in buffer_param_name_list:

        if port_name.find("_"+name) != -1:
            return "True"
    return "False"

# GPU is not friendly to double
# for constant or default value
# if is not int, then add "f" at last
def number_formatter(num_str):
    # if is decimal
    if num_str.find(".") !=-1:
        # if vector
        if(num_str.find(", ") != -1):
            new_num_str = ""
            nums = num_str.split(", ")
            for i in range(len(nums)):
                num = nums[i]
                new_num_str += num +"f"
                if (i != (len(nums) - 1)):
                    new_num_str += ", "
            return new_num_str

        else:
            return num_str + "f"
    return num_str

def output_port_dataType_analyzer(raw_output_names, raw_output_datatypes, index):
    name = raw_output_names[index]
    if(name.find("OpInput") != -1):
        return "struct"
    return raw_output_datatypes[index]

def compare_node_input_analyzer(vop_node):
    compare_operation_dict = {"eq" : "equal", "lt" : "less than", "gt" : "greater than", "lte": "less than or equal", "gte": "greater than or equal", "neq": "not equal"}
    ip = Port()
    ip.port_name = "compare_operation"
    ip.data_type = "string"
    ip.default_value = compare_operation_dict[vop_node.parm("cmp").eval()]
    return ip
"""
{
                    "param_name":"compare_operation",
                    "data_type":"string",
                    "default_value":"less than",
                    "local_input_name":"CG_NONE"
}
"""

def twoway_node_input_analyzer(vop_node):
    compare_operation_dict = {0 : "use input 1 if condition true", 1 : "use input 1 if condition false"}
    ip = Port()
    ip.port_name = "condition_type"
    ip.data_type = "string"
    ip.default_value = compare_operation_dict[vop_node.parm("condtype").eval()]
    return ip
"""
{
                    "param_name":"condition_type",
                    "data_type":"string",
                    "default_value":"use input 1 if condition true",
                    "local_input_name":"CG_NONE"
}
"""

            

# ========= global value
general_operation_list = ["curlnoise", "fit", "cross", "compare", "twoway"]
signature_parm_suffix_dict = {"default": ["1", "2", "3"], "v":["_v1", "_v2", "_v3"], "p":["_p1", "_p2", "_p3"], "c":["_cr", "_cg", "_cb"], "n":["_n1", "_n2", "_n3"]} # uv, un, up, uc...

# get the current python node
this_node = hou.pwd()
# get the vop node
vop = this_node.input(0)
# get all nodes inside vop
vop_children = vop.children()

dag_nodes_list = []
# for topo sort use
sorted_dag_nodes_list = []
node_name_id_dict = dict()
is_node_visited_list = [] # tracked the node is visited or not

# ======================== create DAG Node list
for vop_node in vop_children:
    node_path = vop_node.path().replace('/obj/','')
    node_path = node_path.replace("/", "_")
    print("======================================== now creating dag node for: " + str(vop_node) + " ========================================")
    dag_node = Node()
    dag_node.method_name = vop_node.type().name() 
    dag_node.node_name = vop_node.name()
    dag_node.node_name_digit = vop_node.digitsInName()
    dag_node.id = len(dag_nodes_list)
    dag_node.node_path = node_path
    dag_node.raw_output_names = list(vop_node.outputNames())
    dag_node.raw_output_types = list(vop_node.outputDataTypes())
    # ==== create dag node input port list
    if dag_node.method_name not in general_operation_list: # if is base operator: add, multiply only save inputConnections to DAG Node
        if(len(vop_node.inputConnections()) != 0):
            print("=== creating base operator's dag node input port list")
            input_names = vop_node.inputNames()
            input_data_types = vop_node.inputDataTypes()
            for input_connection in vop_node.inputConnections():  
                ip = Port()
                ip.connected_node = input_connection.inputNode().name()
                ip.connected_index = input_connection.outputIndex()
                #port_name = node_path + "_" + input_names[input_connection.inputIndex()]
                ip.port_name = input_names[input_connection.inputIndex()]
                ip.port_index = input_connection.inputIndex()
                ip.data_type = input_data_types[input_connection.inputIndex()]
                print("    " + str(input_connection))
                print("    connected node: " + ip.connected_node)
                print("    connected index: " + str(ip.connected_index))
                print("    port name: " + ip.port_name)
                print("    data type: " + ip.data_type)
                dag_node.inputs.append(ip)

    else: # if is general operation: curlnoise, fit, cross save all the input param to DAG Node
        print("=== creating general operation's dag node input port list")
        input_connector_list = vop_node.inputConnectors()
        input_data_type_list = vop_node.inputDataTypes()
        input_name_list = vop_node.inputNames()
        for i in range (len(input_name_list)):
            param_name = input_name_list[i]
            data_type = input_data_type_list[i]
            input_connector = input_connector_list[i]
            ip = Port()
            ip.port_name = param_name
            ip.data_type = data_type       
            if(len(input_connector)!=0): # if inputConnector is not empty, has connections
                input_connection = input_connector[0]
                ip.connected_node = input_connection.inputNode().name()
                ip.connected_index = input_connection.outputIndex()
                ip.port_index = input_connection.inputIndex()
                print("    *** parm has connection")
                print("    port name: " + ip.port_name)
                print("    data type: " + ip.data_type)
                print("    " + str(input_connection))
                print("    connected node: " + ip.connected_node)
                print("    connected index: " + str(ip.connected_index))
            else: # inputConnector is empty, has no conncections
                # HOUDINI!!!!!! if parm is VECTOR, NEEDS TO GET value SEPERATELY!!! C.PARM("pos1")
                # HOUDINI!!!!!! parm name to get parm!!!!!!!!
                # normally, node has signature
                # normally, if signature is default, no matter point is vector or point, add 1,2,3 to parm name: pos1, pos2, pos3
                # normally, if signature is not default, which has many data type.. need to add v1, v2, v3 / p1, p2, p3 / cr, cg, cg to parm name
                if data_type == "vector" or data_type == "point":
                    parm_suffix = signature_parm_suffix_dict[vop_node.parm("signature").eval()]
                    ip.default_value = number_formatter(str(vop_node.parm(str(param_name) + parm_suffix[0]).eval())) + "," + \
                                    number_formatter(str(vop_node.parm(str(param_name) + parm_suffix[1]).eval())) + "," +    \
                                    number_formatter(str(vop_node.parm(str(param_name) + parm_suffix[2]).eval()))
                else:  
                    ip.default_value = number_formatter(str(vop_node.parm(param_name).eval()))
                print("    *** parm is default")
                print("    port name: " + ip.port_name)
                print("    data type: " + ip.data_type)
                print("    default value: " + ip.default_value)
            dag_node.inputs.append(ip)            
        if dag_node.method_name == "compare": # compare need one more input which is not on input port
            dag_node.inputs.append(compare_node_input_analyzer(vop_node))
        if dag_node.method_name == "twoway":
            dag_node.inputs.append(twoway_node_input_analyzer(vop_node))
        
    # create dag node output port list
    if(len(vop_node.outputConnections()) != 0):
        print("=== creating dag node output port list")
        output_names = vop_node.outputNames()
        output_data_types = vop_node.outputDataTypes()
        output_labels = vop_node.outputLabels()
        for output_connection in vop_node.outputConnections():  
            op = Port()
            op.connected_node = output_connection.outputNode().name()
            op.connected_index = output_connection.inputIndex()
            #port_name = node_path + "_" + output_names[output_connection.outputIndex()]
            op.port_name = output_names[output_connection.outputIndex()]
            op.port_index = output_connection.outputIndex()
            op.data_type = output_port_dataType_analyzer(output_names, output_data_types, output_connection.outputIndex())
            print("    " + str(output_connection))
            print("    connected node: " + op.connected_node)
            print("    connected index: " + str(op.connected_index))
            print("    port name: " + op.port_name)
            print("    data type: " + op.data_type)
            dag_node.outputs.append(op)
            if dag_node.method_name == "constant":
                print("    creating special constant node input")
                ip = Port()
                value_str = output_labels[output_connection.outputIndex()]
                const_name = vop_node.parm("constname").eval()
                ip.extra_data = number_formatter(value_str.replace(const_name + ": ", ""))
                print("    special constant node input port extra data: " + ip.extra_data)
                dag_node.inputs.append(ip)
            
    dag_nodes_list.append(dag_node)
    sorted_dag_nodes_list.append(dag_node)
    is_node_visited_list.append(False)
    node_name_id_dict[dag_node.node_name] = dag_node.id

# ======================== topolopy sort
N = len(dag_nodes_list)
idx = N - 1 # track the insertion position
for node in dag_nodes_list:
    if not is_node_visited_list[node.id]:
        visit_nodes_list = []
        dfs(node.id, is_node_visited_list, visit_nodes_list, dag_nodes_list)
        for visit_node in visit_nodes_list:
            sorted_dag_nodes_list[idx] = visit_node
            idx -= 1

print("=== sorted dag result")            
for dag_node in sorted_dag_nodes_list:
    print(dag_node.node_name)
    
# ======================== dag to json
import json

# the final output dict
dag_json_dict = dict()

global_input_json_dict = dict()
compute_graph_json_dict = dict()
global_output_json_dict = dict()

# for global_input_json_dict
global_input_list = []
custom_param_list = []
global_input_variable_set_list = []
# for global_output_json_dict
global_output_list = []
bind_output_list = []

global_input_node_type_list = ["geometryvopglobal::2.0", "volumevopglobal"]
global_output_node_type_list = ["geometryvopoutput", "volumevopoutput"] 
buffer_param_name_list = ["P", "v", "force", "Cd", "N"]
variable_type_dict = {"vector":"glm::vec3", "float":"float", "int":"int", "point":"glm::vec3", "string":"char", "struct": "struct", "normal": "glm::vec3"}


connection_dict = dict()
for dag_node in sorted_dag_nodes_list:
    # === global_input_json
    if dag_node.method_name in global_input_node_type_list:
        for output in dag_node.outputs:
            port_unique_name = dag_node.node_path + "_" + output.port_name
            if not (port_unique_name in global_input_variable_set_list):
                # make conncection dict
                connection_dict[dag_node.node_name + "_output_" + str(output.port_index)] = port_unique_name
                global_input_variable_set_list.append(port_unique_name)
                port_dict = dict()
                port_dict["is_buffer"] = is_buffer(port_unique_name)
                if output.data_type in variable_type_dict:
                    port_dict["data_type"] = variable_type_dict[output.data_type]
                else:
                    print("*******ERROR when find data_type for !!! " + port_dict["data_type"] +  " !!!in variable_type_dict!!!")
                port_dict["variable_name"] = port_unique_name
                port_dict["port_name"] = output.port_name
                global_input_list.append(port_dict)
        global_input_json_dict[dag_node.node_name] = global_input_list
        
    elif dag_node.method_name == "parameter":   
        for output in dag_node.outputs:
            port_unique_name = dag_node.node_path + "_" + output.port_name
            # make conncection dict
            connection_dict[dag_node.node_name + "_output_" + str(output.port_index)] = port_unique_name
            port_dict = dict()
            port_dict["is_buffer"] = "False" # ??? some param might be buffer
            port_dict["data_type"] = variable_type_dict[output.data_type]
            port_dict["variable_name"] = port_unique_name
            custom_param_list.append(port_dict)
        
    
    # === global_output_json
    elif dag_node.method_name in global_output_node_type_list:
        for input in dag_node.inputs:
            port_dict = dict()
            # from connection dict get local input name
            find_output_key = input.connected_node + "_output_" + str(input.connected_index)
            if find_output_key in connection_dict:
                port_dict["connection"] = connection_dict[find_output_key]
            else:
                print("*******ERROR when find connection in connection_dict!!!")
            if input.data_type in variable_type_dict:
                port_dict["data_type"] = variable_type_dict[input.data_type]
            else:
                print("*******ERROR when find data_type for !!! " + input.data_type +  " !!!in variable_type_dict!!!")
            port_dict["variable_name"] = dag_node.node_path + "_" + input.port_name
            port_dict["port_name"] = input.port_name
        global_output_list.append(port_dict)
        global_output_json_dict[dag_node.node_name] = global_output_list
    
    elif dag_node.method_name == "bind":
        for input in dag_node.inputs:
            port_dict = dict()
            # from connection dict get local input name
            find_output_key = input.connected_node + "_output_" + str(input.connected_index)
            if find_output_key in connection_dict:
                port_dict["connection"] = connection_dict[find_output_key]
            else:
                print("*******ERROR when find connection in connection_dict!!!")
            if input.data_type in variable_type_dict:
                port_dict["data_type"] = variable_type_dict[input.data_type]
            else:
                print("*******ERROR when find data_type for !!! " + input.data_type +  " !!!in variable_type_dict!!!")
            port_dict["variable_name"] = "__" + port_dict["connection"] + "_" + "debug"
        bind_output_list.append(port_dict)
          
    # === compute_graph_json
    else:
        dag_node_dict = dict()
        dag_node_dict["method_name"] = dag_node.method_name
        dag_node_dict["raw_output_names"] = dag_node.raw_output_names
        dag_node_dict["raw_output_types"] = dag_node.raw_output_types

        # "input" key value
        input_list = []
        if dag_node.method_name in general_operation_list: # if general operation
            for input in dag_node.inputs:
                port_dict = dict()
                port_dict["param_name"] = input.port_name
                port_dict["data_type"] = variable_type_dict[input.data_type]
                if input.default_value == "CG_NONE":
                    port_dict["default_value"] = "CG_NONE"
                    find_output_key = input.connected_node + "_output_" + str(input.connected_index)
                    if find_output_key in connection_dict:
                        port_dict["local_input_name"] = connection_dict[find_output_key]
                    else:
                        print("*******ERROR when find connection in connection_dict!!!")
                        print("       " + dag_node.node_name + "  " + input.port_name)
                else:
                    port_dict["default_value"] = input.default_value
                    port_dict["local_input_name"] = "CG_NONE"
                input_list.append(port_dict)
            
        elif dag_node.method_name == "constant": # special Constant Node
            input = dag_node.inputs[0]
            port_dict = dict()
            port_dict["local_input_name"] = input.extra_data
            input_list.append(port_dict)
        
        else: # base operators
            for input in dag_node.inputs:
                port_dict = dict()
                # from connection dict get local input name
                find_output_key = input.connected_node + "_output_" + str(input.connected_index)
                if find_output_key in connection_dict:
                    port_dict["local_input_name"] = connection_dict[find_output_key]
                else:
                    print("*******ERROR when find connection in connection_dict!!!")
                    print("       " + dag_node.node_name + "  " + input.port_name)
                input_list.append(port_dict)
        dag_node_dict["input"] = input_list   
        
        # "output" key value
        output_list = []
        for output in dag_node.outputs:
            port_unique_name = dag_node.node_path + "_" + output.port_name
            # add to conncection dict
            connection_key = dag_node.node_name + "_output_" + str(output.port_index)
            if connection_key not in connection_dict:
                connection_dict[connection_key] = port_unique_name
                port_dict = dict()
                if output.data_type in variable_type_dict:
                    port_dict["data_type"] = variable_type_dict[output.data_type]
                else:
                    print("*******ERROR when find data_type for !!! " + output.data_type +  " !!!in variable_type_dict!!!")
                port_dict["local_output_name"] = port_unique_name
                output_list.append(port_dict)
        dag_node_dict["output"] = output_list   
        compute_graph_json_dict[dag_node.node_name] = dag_node_dict
    
    
    global_input_json_dict["custom_param"] = custom_param_list
    global_output_json_dict["custom_param"] = bind_output_list
    
    dag_json_dict["func_name"] = vop.name()
    dag_json_dict["global_input"] = global_input_json_dict    
    dag_json_dict["compute_graph"] = compute_graph_json_dict
    dag_json_dict["global_output"] = global_output_json_dict  

json_str = json.dumps(dag_json_dict)
geo = this_node.geometry()
geo.addAttrib(hou.attribType.Global, "dag_json", "")
geo.setGlobalAttribValue("dag_json", json_str)