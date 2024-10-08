import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Flickr
from torch_geometric.datasets import Reddit2
from torch_geometric.loader import NeighborLoader
import torch.onnx
import pandas as pd
import subprocess
from openvino.runtime import Core
import argparse
import time
from dataSplit import split_graph_by_edge_proportion

from sage_split import PropagateModel_10, LinearModel_10, SageFullModel_10

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

# Function to export ONNX models for unique sizes
def export_models_for_unique_sizes(model, df_unique_sizes, onnx_dir):
    feature_sizes = model.feature_sizes
    for index, row in df_unique_sizes.iterrows():
        num_nodes = int(row['Number of Nodes'])
        num_edges = int(row['Number of Edges'])

        for layer_idx in range(model.num_layers):
            onnx_filename = os.path.join(
                onnx_dir,
                f"sage_layer_{layer_idx}_nodes_{num_nodes}_edges_{num_edges}.onnx"
            )

            # Skip exporting if ONNX file already exists
            if os.path.exists(onnx_filename):
                print(f"ONNX model for layer {layer_idx} with {num_nodes} nodes and {num_edges} edges already exists. Skipping export.")
                continue

            in_channels = feature_sizes[layer_idx]
            out_channels = feature_sizes[layer_idx + 1]

            # Create dummy inputs
            dummy_x = torch.randn((num_nodes, in_channels))
            dummy_edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)

            # Get the specific layer
            conv = model.convs[layer_idx]

            # Export the model
            print(f"Exporting ONNX model for layer {layer_idx} with {num_nodes} nodes and {num_edges} edges to {onnx_filename}.")
            torch.onnx.export(
                conv,
                (dummy_x, dummy_edge_index),
                onnx_filename,
                input_names=['x', 'edge_index'],
                output_names=['output'],
                dynamic_axes=None,  # Static shapes
                opset_version=11
            )

# Function to compile IR models for unique sizes
def compile_ir_models_for_unique_sizes(df_unique_sizes, model_num_layers, onnx_dir, ir_dir):
    core = Core()  # Initialize OpenVINO runtime core
    for index, row in df_unique_sizes.iterrows():
        num_nodes = int(row['Number of Nodes'])
        num_edges = int(row['Number of Edges'])

        for layer_idx in range(model_num_layers):
            onnx_filename = os.path.join(
                onnx_dir,
                f"sage_layer_{layer_idx}_nodes_{num_nodes}_edges_{num_edges}.onnx"
            )
            ir_output_dir = os.path.join(ir_dir, f"layer_{layer_idx}_nodes_{num_nodes}_edges_{num_edges}")

            # Check if IR model already exists (skip if XML file is already present)
            ir_model_xml = os.path.join(ir_output_dir, f"sage_layer_{layer_idx}_nodes_{num_nodes}_edges_{num_edges}.xml")
            if os.path.exists(ir_model_xml):
                print(f"IR model for layer {layer_idx} with {num_nodes} nodes and {num_edges} edges already exists. Skipping IR compilation.")
                continue

            if not os.path.exists(ir_output_dir):
                os.makedirs(ir_output_dir)

            print(f"Compiling IR model for layer {layer_idx} with {num_nodes} nodes and {num_edges} edges.")
            try:
                subprocess.run(
                    ['mo', '--input_model', onnx_filename, '--output_dir', ir_output_dir],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Error during OpenVINO model optimization: {e}")
                
def handle_batches(subgraph_loader, current_layer, current_batch):
    """
    Function to iterate through subgraph batches, keeping track of the current and last batch.
    
    Args:
    - subgraph_loader: The data loader that loads the subgraph batches.
    - total_batch: Total number of batches in the subgraph loader.
    - current_layer: The current layer index being processed.
    - current_batch: The current batch index being processed.
    
    Returns:
    - last_batch: The last batch processed (or None if it's the first batch).
    - current_batch: The current batch being processed (or None if it's beyond total_batch).
    """
    total_batch = sum(1 for _ in subgraph_loader)  # Calculate the total number of batches
    last_batch_data = None
    current_batch_data = None
    
    # Iterate over all batches in the subgraph loader
    for batch_idx, batch in enumerate(subgraph_loader):
        # If current_batch is 0, we are at the first batch
        if current_batch == 0:
            last_batch_data = batch
            current_batch_data = batch
            break
        
        # If current_batch is greater than total_batch, handle last case
        if current_batch == total_batch + 1:
            last_batch_data = batch  # Set last_batch_data to the last batch
            current_batch_data = None  # No current batch, as we are beyond total_batch
            continue
        
        # Otherwise, process normally by finding the current batch and saving last batch
        if batch_idx == current_batch - 1:
            last_batch_data = batch  # The batch before current_batch
        if batch_idx == current_batch:
            current_batch_data = batch  # The actual current batch
            break

    return last_batch_data, current_batch_data

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

# input: model_sets, set_num, device_sets
# call other function: device, model, input_batch, current_layer,  
# 
# return time
def run_inference(model_name, model_component_name, ir_dir, csv_path, device, input_batch, feature_size, current_layer_idx):
    # load model
    core = Core()
    x_all = None
    xs = []
    if device == "NPU":
        input_batch = pad_to_smallest_matching_model(input_batch,  csv_path)
        
    num_nodes = input_batch.x.size(0)
    num_edges = input_batch.edge_index.size(1)

    # load the model
    if device == "CPU" or device == "GPU":
        # ir_path = os.path.join(ir_dir, f"sage_{model_name}_features_{feature_size}_dynamic.xml")
        ir_path =  ir_dir + f"/{model_name}_{model_component_name}_features_{feature_size}_dynamic.xml"
    elif device == "NPU":
        if 'propagate' in model_component_name.lower() or 'sagefull' in model_component_name.lower() :
            # ir_path = os.path.join(ir_dir, f"{model_name}_nodes_{num_nodes}_features_{feature_size}_edges_{num_edges}.xml")
            ir_path =  ir_dir + f"/{model_name}_{model_component_name}_features_{feature_size}_nodes_{num_nodes}_edges_{num_edges}.xml"
        elif 'linear' in model_component_name.lower():
            # ir_path = os.path.join(ir_dir, f"{model_name}_nodes_{num_nodes}_features_{feature_size}.xml")
            ir_path =  ir_dir + f"/{model_name}_{model_component_name}_features_{feature_size}_nodes_{num_nodes}.xml"
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
    else:
        raise ValueError(f"Unsupported device type: {device}")

    # compile the model
    compiled_model = core.compile_model(ir_path, device)  # cpu model
    if not os.path.exists(ir_path):
        raise FileNotFoundError(f"IR model not found for {ir_path}.")
    
    # Load the data
    if device == "CPU" or device == "GPU":
        if 'propagate' in model_component_name.lower() or 'sagefull' in model_component_name.lower():
            if current_layer_idx == 0:
                x = input_batch.x
            elif current_layer_idx == 1:
                x = torch.randn((input_batch.x.size(0), feature_size))
            edge_index = input_batch.edge_index
            inputs = (x, edge_index)
        elif 'linear' in model_component_name.lower():
            message = torch.randn((num_nodes, feature_size))
            if current_layer_idx == 0:
                x = input_batch.x
            elif current_layer_idx == 1:
                x = torch.randn((input_batch.x.size(0), feature_size))
            inputs = (message, x)
        else:
            raise ValueError(f"Unsupported model type: {model_component_name}")
    elif device == "NPU":
        if 'propagate' in model_component_name.lower() or 'sagefull' in model_component_name.lower():
            if current_layer_idx == 0:
                x = input_batch.x
            elif current_layer_idx == 1:
                x = torch.randn((input_batch.x.size(0), feature_size))
            edge_index = input_batch.edge_index
            inputs = (x, edge_index)
        elif 'linear' in model_component_name.lower():
            message = torch.randn((input_batch.x.size(0), feature_size))
            if current_layer_idx == 0:
                x = input_batch.x
            elif current_layer_idx == 1:
                x = torch.randn((input_batch.x.size(0), feature_size))
            inputs = (message, x)
        else:
            raise ValueError(f"Unsupported model type: {model_component_name}")
    else:
        raise ValueError(f"Unsupported device type: {device}")

    # run inference
    start_time = time.time()

    # Perform linear transformation on cpu using OpenVINO
    inputs = {
        compiled_model.inputs[0].any_name: inputs[0].numpy(),
        compiled_model.inputs[1].any_name: inputs[1].numpy()
    }
    infer_request = compiled_model.create_infer_request()
    infer_request.start_async(inputs=inputs)
    infer_request.wait()
    output_tensor = infer_request.get_output_tensor(0).data
    output = torch.tensor(output_tensor)
    # print(f"output_tensor {output}")
    timing = (time.time() - start_time)/10 * 1000
    return timing

def find_combinations(idx):
    combinations = []
    value_range = [1, 9]
    cnt = 0
    for x in range(value_range[0], value_range[1] + 1):
        for y in range(value_range[0], value_range[1] + 1):
            z = 10 - x - y
            if value_range[0] <= z <= value_range[1]:
                combinations = [x, y, z]
                if idx == cnt:
                    return [x / 10.0 for x in combinations]
                cnt += 1
    return None
         
def get_pipeline_time(model_name, model_component_name, input_batch, feature_size, ir_dir, csv_path, current_layer_idx, current_batch_idx, current_ratio_idx):

    device_sets = ["CPU", "NPU", "GPU"]

    split_rate = find_combinations(current_ratio_idx)
    if split_rate == None:
        raise ValueError(f"Unsupported rate idx {current_ratio_idx}")
        return
 
    print(f"layer {current_layer_idx} batch {current_batch_idx} current ratio {current_ratio_idx}: {split_rate}")
    
    subgraphs = split_graph_by_edge_proportion(input_batch, 3, split_rate)

    time0 = run_inference(model_name, model_component_name, ir_dir, csv_path, device_sets[0], subgraphs[0], feature_size, current_layer_idx)
    time1 = run_inference(model_name, model_component_name, ir_dir, csv_path, device_sets[1], subgraphs[1], feature_size, current_layer_idx)
    time2 = run_inference(model_name, model_component_name, ir_dir, csv_path, device_sets[2], subgraphs[2], feature_size, current_layer_idx)
    print(f"{device_sets[0]} time:{time0:.4f}ms; {device_sets[1]} time:{time1:.4f}ms; {device_sets[2]} time:{time2:.4f}ms")
    return 

 

# Parse command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process layer and batch information.")
    parser.add_argument('--current_layer', type=int, help='The index of the current layer to process.')
    parser.add_argument('--current_batch', type=int, help='The index of the current batch to process.')
    parser.add_argument('--current_rate', type=int, help='The index of the current device to process.')

     # Parse arguments
    args = parser.parse_args()

    current_layer_idx = args.current_layer
    current_batch_idx = args.current_batch
    current_ratio_idx = args.current_rate

    ################################
    # setting
    ################################
    model_name = "sage"
    dataset_name = "Reddit2"
    csv_path = f"utils/{dataset_name}_unique_padded_sizes.csv"
    current_layer = 0

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

    # Define the NeighborLoader
    subgraph_loader = NeighborLoader(
        data,
        num_neighbors=[-1],
        batch_size=4096,  # Adjust as needed
        input_nodes=None
    )
    total_batch = sum(1 for _ in subgraph_loader)
    print(f"{dataset_name} total node {data.x.size(0)} total edge {data.edge_index.size(1)} batch {total_batch} feature size {data.x.size(1)}")
      
    feature_size_list = [dataset.num_features, 256, dataset.num_classes]
    in_channel_size = feature_size_list[current_layer]
    out_channel_size = feature_size_list[current_layer + 1]

    
    if not os.path.exists(csv_path):
        # Collect unique padded sizes
        unique_sizes = collect_unique_padded_sizes(subgraph_loader, multiple)
        # Save unique sizes to CSV
        df_unique_sizes = pd.DataFrame(list(unique_sizes), columns=['Number of Nodes', 'Number of Edges'])
        df_unique_sizes.to_csv(csv_path, index=False)
        print(f"Exported unique padded sizes to {csv_path}.")
    else:
        print(f"{csv_path} already exists")
    
    # Define directories
    onnx_dir = f"Onnx/{model_name}_{dataset_name}"
    ir_dir = f"Ir/{model_name}_{dataset_name}"
    if not os.path.exists(onnx_dir):
        os.makedirs(onnx_dir)
    if not os.path.exists(onnx_dir):
        os.makedirs(onnx_dir)
    
    if model_name == "sage":
        model_components = {
            # "propagate": PropagateModel_10(in_channels=in_channel_size, out_channels=out_channel_size),
            # "linear": LinearModel_10(in_channels=in_channel_size, out_channels=out_channel_size),
            "sagefull": SageFullModel_10(in_channels=in_channel_size, out_channels=out_channel_size)
        ###################################
        # add here
        ##########
        }
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
        
    for model_component_name, model in model_components.items():
        last_batch, current_batch = handle_batches(subgraph_loader, current_layer_idx, current_batch_idx)
        get_pipeline_time(model_name, model_component_name, current_batch, in_channel_size, ir_dir, csv_path, current_layer_idx, current_batch_idx, current_ratio_idx)
   
