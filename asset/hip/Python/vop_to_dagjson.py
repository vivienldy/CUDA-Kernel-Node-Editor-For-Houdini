class Port:
    def __init__(self):
        self.connected_node = '' # connect to which node                       
        self.connected_index = -1 # index of the port on that node
        self.port_name = ''
        self.port_index = -1
        self.data_type = ''
        self.extra_data = '' # for extradata like constant value

class Node:
    def __init__(self):
        self.id = -1 # for topology sorted
        self.method_name = ''
        self.node_name = ''
        self.node_name_digit = -1
        self.inputs = [] # list of Port
        self.outputs = [] # list of Port
        
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
    print("==== now creating dag node for: " + str(vop_node))
    dag_node = Node()
    dag_node.method_name = vop_node.type().name() 
    dag_node.node_name = vop_node.name()
    dag_node.node_name_digit = vop_node.digitsInName()
    dag_node.id = len(dag_nodes_list)
    # create dag node input port list
    if(len(vop_node.inputConnections()) != 0):
        print("=== creating dag node input port list")
        input_names = vop_node.inputNames()
        input_data_types = vop_node.inputDataTypes()
        for input_connection in vop_node.inputConnections():  
            ip = Port()
            ip.connected_node = input_connection.inputNode().name()
            ip.connected_index = input_connection.outputIndex()
            port_name = node_path + "_" + input_names[input_connection.inputIndex()]
            ip.port_name = port_name
            ip.port_index = input_connection.inputIndex()
            ip.data_type = input_data_types[input_connection.inputIndex()]
            print("    " + str(input_connection))
            print("    connected node: " + ip.connected_node)
            print("    connected index: " + str(ip.connected_index))
            print("    port name: " + ip.port_name)
            print("    data type: " + ip.data_type)
            dag_node.inputs.append(ip)
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
            port_name = node_path + "_" + output_names[output_connection.outputIndex()]
            op.port_name = port_name
            op.port_index = output_connection.outputIndex()
            op.data_type = output_data_types[output_connection.outputIndex()]
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
                ip.extra_data = value_str.replace("Value: ", "")
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

global_input_node_type_list = ["geometryvopglobal::2.0", "parameter"]
global_output_node_type_list = ["geometryvopoutput", "bind"]
buffer_param_name_list = ["P", "v", "force", "Cd", "N"]
variable_type_dict = {"vector":"glm::vec3", "float":"float", "int":"int"}


connection_dict = dict()
for dag_node in sorted_dag_nodes_list:
    # === global_input_json
    if dag_node.method_name == "geometryvopglobal::2.0":
        for output in dag_node.outputs:
            if not (output.port_name in global_input_variable_set_list):
                # make conncection dict
                connection_dict[dag_node.node_name + "_output_" + str(output.port_index)] = output.port_name
                global_input_variable_set_list.append(output.port_name)
                port_dict = dict()
                port_dict["is_buffer"] = is_buffer(output.port_name)
                if output.data_type in variable_type_dict:
                    port_dict["data_type"] = variable_type_dict[output.data_type]
                else:
                    print("*******ERROR when find data_type in variable_type_dict!!!")
                port_dict["variable_name"] = output.port_name
                global_input_list.append(port_dict)
        global_input_json_dict[dag_node.node_name] = global_input_list
        
    elif dag_node.method_name == "parameter":   
        for output in dag_node.outputs:
            # make conncection dict
            connection_dict[dag_node.node_name + "_output_" + str(output.port_index)] = output.port_name
            port_dict = dict()
            port_dict["is_buffer"] = "False" # ??? some param might be buffer
            port_dict["data_type"] = variable_type_dict[output.data_type]
            port_dict["variable_name"] = output.port_name
            custom_param_list.append(port_dict)
        
    
    # === global_output_json
    elif dag_node.method_name == "geometryvopoutput":
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
                print("*******ERROR when find data_type in variable_type_dict!!!")
            port_dict["variable_name"] = input.port_name
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
                print("*******ERROR when find data_type in variable_type_dict!!!")
            port_dict["variable_name"] = "__" + port_dict["connection"] + "_" + "debug"
        bind_output_list.append(port_dict)
          
    # === compute_graph_json
    else:
        dag_node_dict = dict()
        dag_node_dict["method_name"] = dag_node.method_name
        
        input_list = []
        for input in dag_node.inputs:
            port_dict = dict()
            # from connection dict get local input name
            find_output_key = input.connected_node + "_output_" + str(input.connected_index)
            if find_output_key in connection_dict:
                port_dict["local_input_name"] = connection_dict[find_output_key]
            else:
                print("*******ERROR when find connection in connection_dict!!!")
            if dag_node.method_name == "constant":
                port_dict["local_input_name"] = input.extra_data
            input_list.append(port_dict)
        dag_node_dict["input"] = input_list   
        
        output_list = []
        for output in dag_node.outputs:
            local_output_name = output.port_name
            # add to conncection dict
            connection_dict[dag_node.node_name + "_output_" + str(output.port_index)] = local_output_name
            port_dict = dict()
            if output.data_type in variable_type_dict:
                port_dict["data_type"] = variable_type_dict[output.data_type]
            else:
                print("*******ERROR when find data_type in variable_type_dict!!!")
            port_dict["local_output_name"] = local_output_name
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
print(json_str)
geo = this_node.geometry()
geo.addAttrib(hou.attribType.Global, "dag_json", "")
geo.setGlobalAttribValue("dag_json", json_str)