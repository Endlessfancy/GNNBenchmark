# device_set = [["NPU", "GPU"], "CPU"]
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
from modelExport_npu_static import PropagateModel_10, LinearModel_10

# Define the SAGE model
class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        feature_sizes = [in_channels] + [hidden_channels] * (num_layers)
        self.feature_sizes = feature_sizes
        for idx in range(num_layers):
            self.convs.append(SAGEConv(feature_sizes[idx], feature_sizes[idx + 1]))

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = x.relu()
        return x
    
# Function to pad nodes and edges to the nearest multiple of 5000
def pad_to_multiple(batch, multiple):
    batch_num_nodes = batch.num_nodes
    batch_num_edges = batch.edge_index.size(1)

    # Calculate the number of nodes and edges to pad to the next multiple of 5000
    padding_nodes = (multiple - batch_num_nodes % multiple) % multiple
    padding_edges = (multiple - batch_num_edges % multiple) % multiple

    # Pad nodes by adding zero features
    if padding_nodes > 0:
        padding_node_features = torch.zeros((padding_nodes, batch.x.size(1)))
        batch.x = torch.cat([batch.x, padding_node_features], dim=0)

    # Pad edges by adding (0, 0) self-loop edges
    if padding_edges > 0:
        padding_edge_index = torch.tensor([[0] * padding_edges, [0] * padding_edges])
        batch.edge_index = torch.cat([batch.edge_index, padding_edge_index], dim=1)

    # Update the number of nodes
    batch.num_nodes = batch.x.size(0)

    return batch
# Function to find the smallest matching node and edge count from CSV
def find_smallest_matching_size_from_csv(csv_file_path, num_nodes_npu, num_edges_npu):
    # Read the CSV file containing the unique sizes (Number of Nodes, Number of Edges)
    df_unique_sizes = pd.read_csv(csv_file_path)

    # Filter rows where num_nodes >= num_nodes_npu and num_edges >= num_edges_npu
    suitable_models = df_unique_sizes[
        (df_unique_sizes['Number of Nodes'] >= num_nodes_npu) &
        (df_unique_sizes['Number of Edges'] >= num_edges_npu)
    ]

    # If no suitable models are found, raise an error
    if suitable_models.empty:
        raise FileNotFoundError(f"No suitable size found with at least {num_nodes_npu} nodes and {num_edges_npu} edges.")

    # Find the row with the smallest num_nodes and num_edges that satisfy the condition
    smallest_model = suitable_models.nsmallest(1, ['Number of Nodes', 'Number of Edges']).iloc[0]

    # Return the node and edge count
    return int(smallest_model['Number of Nodes']), int(smallest_model['Number of Edges'])

# Function to pad nodes and edges to match the smallest suitable model
def pad_to_smallest_matching_model(batch, csv_file_path):
    # Get current batch size
    batch_num_nodes = batch.num_nodes
    batch_num_edges = batch.edge_index.size(1)

    # Find the smallest matching node and edge sizes from the CSV
    try:
        target_num_nodes, target_num_edges = find_smallest_matching_size_from_csv(csv_file_path, batch_num_nodes, batch_num_edges)
        # print(f"Padding to {target_num_nodes} nodes and {target_num_edges} edges.")
    except FileNotFoundError as e:
        print(str(e))
        return batch  # If no suitable size is found, return the original batch

    # Calculate padding for nodes and edges
    padding_nodes = target_num_nodes - batch_num_nodes
    padding_edges = target_num_edges - batch_num_edges

    # Pad nodes by adding zero features
    if padding_nodes > 0:
        padding_node_features = torch.zeros((padding_nodes, batch.x.size(1)))
        batch.x = torch.cat([batch.x, padding_node_features], dim=0)

    # Pad edges by adding (0, 0) self-loop edges
    if padding_edges > 0:
        padding_edge_index = torch.tensor([[0] * padding_edges, [0] * padding_edges])
        batch.edge_index = torch.cat([batch.edge_index, padding_edge_index], dim=1)

    # Update the number of nodes
    batch.num_nodes = batch.x.size(0)

    return batch

# Function to collect unique padded sizes
def collect_unique_padded_sizes(subgraph_loader, multiple):
    unique_sizes = set()
    for batch in subgraph_loader:
        batch = pad_to_multiple(batch, multiple)
        num_nodes = batch.num_nodes
        num_edges = batch.edge_index.size(1)
        unique_sizes.add((num_nodes, num_edges))
    # return unique_sizes
    # Find the minimum size in the unique sizes
    min_num_nodes, min_num_edges = min(unique_sizes, key=lambda x: (x[0], x[1]))

    # Generate smaller combinations where num_edges >= num_nodes and both are multiples of 5000
    additional_sizes = set()
    for nodes in range(multiple, min_num_nodes + 1, multiple):
        for edges in range(multiple, min_num_edges + 1, multiple):
            additional_sizes.add((nodes, edges))

    # Combine the additional sizes with the original unique sizes
    combined_sizes = unique_sizes.union(additional_sizes)
    return combined_sizes
                
def handle_current_batches(subgraph_loader, current_batch):

    total_batch = sum(1 for _ in subgraph_loader)  # Calculate the total number of batches
    current_batch_data = None
    
    # Iterate over all batches in the subgraph loader
    for batch_idx, batch in enumerate(subgraph_loader):
        if batch_idx == current_batch:
            current_batch_data = batch  # The actual current batch
            break

    return current_batch_data


# Inference function
def run_inference_NPUonly(model_name, ir_dir, subgraph_loader, df_unique_sizes, feature_size, current_layer_cnt, current_batch_cnt):
    # Load the compiled model
    timings = []
    core = Core()
    x_all = None
    xs = []

    current_batch = handle_current_batches(subgraph_loader, current_batch_cnt)
    npu_input = current_batch
    npu_input.x = torch.randn((npu_input.num_nodes, feature_size))

    npu_input_padding = pad_to_smallest_matching_model(npu_input, 'unique_padded_sizes.csv')
    # Padding the input & load the model for NPU
    num_nodes_npu = npu_input_padding.num_nodes
    num_edges_npu = npu_input_padding.edge_index.size(1)
    
        # Create dummy inputs based on model input type
    if 'propagate' in model_name.lower() or 'sage' in model_name.lower():
        # For GNN models with 'propagate' or 'sage'
        x_npu = npu_input_padding.x
        edge_index_npu = npu_input_padding.edge_index
        inputs = (x_npu, edge_index_npu)
        ir_path_npu = os.path.join(ir_dir, f"{model_name}_nodes_{num_nodes_npu}_features_{feature_size}_edges_{num_edges_npu}.xml")
    elif 'linear' in model_name.lower():
        # For linear models
        message_npu = torch.randn((num_nodes_npu, feature_size))
        x_npu = npu_input_padding.x
        inputs = (message_npu, x_npu)
        ir_path_npu = os.path.join(ir_dir, f"{model_name}_nodes_{num_nodes_npu}_features_{feature_size}.xml")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
   
    if not os.path.exists(ir_path_npu):
        raise FileNotFoundError(f"IR model not found for {ir_path_npu}.")
    
    # Load the corresponding compiled model for NPU
    compiled_model_npu = core.compile_model(ir_path_npu, "NPU")  # NPU model

    start_time = time.time()
    # Perform linear transformation on NPU using OpenVINO
    inputs_npu = {
        compiled_model_npu.inputs[0].any_name: inputs[0].numpy(),
        compiled_model_npu.inputs[1].any_name: inputs[1].numpy()
    }
    infer_request_npu = compiled_model_npu.create_infer_request()
    infer_request_npu.start_async(inputs=inputs_npu)
    infer_request_npu.wait()
    output_tensor_npu = infer_request_npu.get_output_tensor(0).data
    output_npu = torch.tensor(output_tensor_npu)
    # print(f"output_tensor_npu {output_npu}")
    npu_end_time = time.time() - start_time
    
    print(f"NPU load model {ir_path_npu}")
    print(f"NPU time layer {current_layer_cnt} batch {current_batch_cnt} : {npu_end_time:.4f} seconds")

    return x_all, npu_end_time

# Parse command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process layer and batch information.")
    parser.add_argument('--current_layer', type=int, help='The index of the current layer to process.')
    parser.add_argument('--current_batch', type=int, help='The index of the current batch to process.')
    # parser.add_argument('--plan', type=int, help='The index of the current plan to process.')
    
    # Parse arguments
    args = parser.parse_args()

    current_layer = args.current_layer
    current_batch = args.current_batch
    # plan_cnt = args.plan

    ######################################################
    # Input Parameter 
    #####################################################
    model_name = "propagate"
    dataset_name = "Flickr"
    onnx_dir = "Onnx/test"
    ir_dir = "Ir/test"

    # Load the dataset
    if dataset_name == "Flickr":
        dataset = Flickr(root="data/Flickr")
        data = dataset[0]
        feature_size_list = [500, 256]
    else:
        dataset = Flickr(root="data/Flickr")
        data = dataset[0]
        feature_size_list = [500, 256]

    # Define the NeighborLoader
    subgraph_loader = NeighborLoader(
        data,
        num_neighbors=[-1],
        batch_size=4096,  # Adjust as needed
        input_nodes=None
    )
    # load the model
    model_list = []
    if model_name == "propagate":
        model_1 = PropagateModel_10(in_channels=dataset.num_features, out_channels=256)
        model_2 = PropagateModel_10(in_channels=256, out_channels=dataset.num_classes)
        model_list.append(model_1)
        model_list.append(model_2)
    elif model_name == "linear":
        model_1 = LinearModel_10(in_channels=dataset.num_features, out_channels=256)
        model_2 = LinearModel_10(in_channels=256, out_channels=dataset.num_classes)
        model_list.append(model_1)
        model_list.append(model_2)
  
    # Collect unique padded sizes
    multiple = 5000
    unique_sizes = collect_unique_padded_sizes(subgraph_loader, multiple)

    # Define the NeighborLoader
    subgraph_loader = NeighborLoader(
        data,
        num_neighbors=[-1],
        batch_size=4096,  # Adjust as needed
        input_nodes=None
    )

    # Save unique sizes to CSV
    df_unique_sizes = pd.DataFrame(list(unique_sizes), columns=['Number of Nodes', 'Number of Edges'])
    df_unique_sizes.to_csv('unique_padded_sizes.csv', index=False)

    feature_size = feature_size_list[current_layer]

    # Define directories
    ir_dir = "Ir/test"


    output, timings = run_inference_NPUonly(model_name, ir_dir, subgraph_loader, df_unique_sizes, feature_size, current_layer, current_batch)

    # write to file
    with open('npu_execution_times.txt', 'a') as f:
        f.write(f"Layer: {current_layer}, Batch: {current_batch}, Time: {timings:.4f} seconds\n")
