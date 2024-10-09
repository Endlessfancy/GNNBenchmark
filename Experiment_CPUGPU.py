# device_set = [["cpu", "GPU"], "CPU"]
# stage_set = ["stage1", "stage2"]

# for devices in device_set:
#     for device in devices:


import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Flickr
from torch_geometric.loader import NeighborLoader
import torch.onnx
import pandas as pd
import subprocess
from openvino.runtime import Core
import argparse
import time
from dataSplit import split_graph_by_edge_proportion
import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Flickr
from torch_geometric.datasets import Reddit2
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch_geometric.loader import NeighborLoader
import torch.onnx
import pandas as pd
import subprocess
from openvino.runtime import Core
import argparse
import time
from dataSplit import split_graph_by_edge_proportion



# Inference function
def run_inference_CPUonly(model_name, model_component_name, ir_dir, feature_size, current_batch, device):

    core = Core()
    x_all = None
    xs = []

    ir_path_cpu =  ir_dir + f"/{model_name}_{model_component_name}_features_{feature_size}_dynamic.xml"
    # Load the corresponding compiled model for cpu
    compiled_model_cpu = core.compile_model(ir_path_cpu, device)  # cpu model
    if not os.path.exists(ir_path_cpu):
        raise FileNotFoundError(f"IR model not found for {ir_path_cpu}.")
    
    start_time = time.time()
    
    num_nodes_cpu = current_batch.num_nodes
        # Create dummy inputs based on model input type
    if 'propagate' in model_component_name.lower() or 'sagefull' in model_component_name.lower():
        # For GNN models with 'propagate' or 'sage'
        if feature_size == 500:
            x_cpu = current_batch.x
        else:
            x_cpu = torch.randn((num_nodes_cpu, feature_size))
        edge_index_cpu = current_batch.edge_index
        inputs = (x_cpu, edge_index_cpu)
        
    elif 'linear' in model_component_name.lower():
        # For linear models
        message_cpu = torch.randn((num_nodes_cpu, feature_size))
        if feature_size == 500:
            x_cpu = current_batch.x
        else:
            x_cpu = torch.randn((num_nodes_cpu, feature_size))
        inputs = (message_cpu, x_cpu)
    else:
        raise ValueError(f"Unsupported model type: {model_component_name}")
    
    # Perform linear transformation on cpu using OpenVINO
    inputs_cpu = {
        compiled_model_cpu.inputs[0].any_name: inputs[0].numpy(),
        compiled_model_cpu.inputs[1].any_name: inputs[1].numpy()
    }
    infer_request_cpu = compiled_model_cpu.create_infer_request()
    infer_request_cpu.start_async(inputs=inputs_cpu)
    infer_request_cpu.wait()
    output_tensor_cpu = infer_request_cpu.get_output_tensor(0).data
    output_cpu = torch.tensor(output_tensor_cpu)
    # print(f"output_tensor_cpu {output_cpu}")
    cpu_end_time = (time.time() - start_time)/10 * 1000

    return x_all, cpu_end_time
# Parse command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process layer and batch information.")
    parser.add_argument('--model', type=str, help='The index of the current layer to process.')
    parser.add_argument('--dataset', type=str, help='The index of the current batch to process.')
    parser.add_argument('--device', type=str, help='The index of the current device to process.')
    parser.add_argument('--model_componet', type=str, help='The index of the current stage to process.')

    # Parse arguments
    args = parser.parse_args()
    model_name = args.model
    dataset_name = args.dataset
    device = args.device
    model_component_name = args.model_componet
     ######################################################
    # Input Parameter 
    #####################################################
    # model_name = "sage"
    # model_component_name = "sagefull"
    # dataset_name = "Reddit2"
    # device = "GPU"
    # current_layer_idx = 0

    # Load the dataset
    if dataset_name == "Flickr":
        dataset = Flickr(root="data/Flickr")
        data = dataset[0]
        multiple = 5000
    elif dataset_name == "Reddit2":
        dataset = Reddit2(root="data/Reddit2")
        data = dataset[0]
        multiple = 20000
    elif dataset_name == "Product":
        dataset = PygNodePropPredDataset('ogbn-products', 'data/Product')
        data = dataset[0]
        multiple = 20000
    feature_size_list = [dataset.num_features, 256, dataset.num_classes]
    # in_channel_size = feature_size_list[current_layer_idx]
    # out_channel_size = feature_size_list[current_layer_idx + 1]

    # Define the NeighborLoader
    subgraph_loader = NeighborLoader(
        data,
        num_neighbors=[-1],
        batch_size=4096,  # Adjust as needed
        input_nodes=None
    )

    # Define directories
    onnx_dir = f"Onnx/{model_name}_{dataset_name}"
    ir_dir = f"Ir/{model_name}_{dataset_name}"
    if not os.path.exists(onnx_dir):
        os.makedirs(onnx_dir)
    if not os.path.exists(onnx_dir):
        os.makedirs(onnx_dir)
    
    timings = []
    for current_layer_cnt in range(2):
        for current_batch_cnt, current_batch in enumerate(subgraph_loader):
            # Run inferencce
            
            feature_size = feature_size_list[current_layer_cnt]
            output, timing = run_inference_CPUonly(model_name, model_component_name, ir_dir, feature_size, current_batch, device)
            timings.append(timing)
            print(f"{device} time layer {current_layer_cnt} batch {current_batch_cnt} : timing: {timing:.4f}ms  ")

    print(f"{device} total time: {sum(timings):.4f}ms")

    # timings = []
    # for current_layer_cnt in range(2):
    #     for current_batch_cnt, current_batch in enumerate(subgraph_loader):
    #         # Run inferencce
    #         model_name = "propagate"
    #         feature_size = feature_size_list[current_layer_cnt]
    #         output, timing_prop = run_inference_CPUonly(model_name, ir_dir, feature_size, current_batch, device)
    #         model_name = "linear"
    #         feature_size = feature_size_list[current_layer_cnt]
    #         output, timing_lin = run_inference_CPUonly(model_name, ir_dir, feature_size, current_batch, device)
    #         timings.append(timing_lin+timing_prop)
    #         print(f"{device} time layer {current_layer_cnt} batch {current_batch_cnt} : timing_prop: {timing_prop:.4f}ms timings_lin: {timing_lin:.4f}ms, total time: {timing_prop+timing_lin:.4f}ms ")

    # print(f"{device} total time: {sum(timings):.4f}ms")
