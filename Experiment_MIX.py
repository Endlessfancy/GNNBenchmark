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


# Inference function
def run_inference_with_dynamic_loading(model, subgraph_loader, multiple, ir_dir, current_layer_cnt, current_batch_cnt):
    # Load the compiled model
    timings = []
    core = Core()
    x_all = None
    xs = []
    in_channels = model.convs[current_layer_cnt].in_channels # !!! need to change herer
    out_channels = model.convs[current_layer_cnt].out_channels 
    last_batch, current_batch = handle_batches(subgraph_loader, current_layer_cnt, current_batch_cnt)
    cpu_input = last_batch
    gpu_npu_input = current_batch
       

    ##################################    
    # Pipeline stage1 (CPU)
    ##################################
    # Prepare inputs0
    model_cpu = "propagate"
    ir_model_dir_cpu = f"Ir/sage_{model_cpu}_layer_{current_layer}_dynamic.xml"
    if not os.path.exists(ir_model_dir_cpu):
        raise FileNotFoundError(f"IR model not found for layer {current_layer} with {current_batch - 1}.")

    # Load the corresponding compiled model for CPU
    compiled_model_cpu = core.compile_model(ir_model_dir_cpu, "CPU")  # CPU model

    start_time = time.time()
    if current_batch_cnt == 0 or cpu_input == None:
        cpu_input.x = torch.randn((cpu_input.x.size(0), in_channels))
        # Perform linear transformation on CPU using OpenVINOs
        inputs_cpu = {
            compiled_model_cpu.inputs[0].any_name: cpu_input.x.numpy(),
            compiled_model_cpu.inputs[1].any_name: cpu_input.edge_index.numpy()
        }
        infer_request_cpu = compiled_model_cpu.create_infer_request()
        infer_request_cpu.start_async(inputs=inputs_cpu)
        infer_request_cpu.wait()
        output_tensor_cpu = infer_request_cpu.get_output_tensor(0).data
        output_cpu = torch.tensor(output_tensor_cpu)
        # print(f"output_tensor_cpu {output_tensor_cpu}")
        end_time = time.time() - start_time
        print(f"CPU time layer {current_layer_cnt} batch {current_batch_cnt}: 0 seconds")
    else:
        cpu_input.x = torch.randn((cpu_input.x.size(0), in_channels))
        # Perform linear transformation on CPU using OpenVINO
        inputs_cpu = {
            compiled_model_cpu.inputs[0].any_name: cpu_input.x.numpy(),
            compiled_model_cpu.inputs[1].any_name: cpu_input.edge_index.numpy()
        }
        infer_request_cpu = compiled_model_cpu.create_infer_request()
        infer_request_cpu.start_async(inputs=inputs_cpu)
        infer_request_cpu.wait()
        output_tensor_cpu = infer_request_cpu.get_output_tensor(0).data
        output_cpu = torch.tensor(output_tensor_cpu)
        # print(f"output_tensor_cpu {output_tensor_cpu}")
        end_time = time.time() - start_time
        print(f"CPU time layer {current_layer_cnt} batch {current_batch_cnt}: {end_time:.4f} seconds")
        timings.append(end_time)
        xs.append(output_cpu)
    

    split_rate_init = [0.1, 0.9]
    timings_stage2 = []
    max_timing_stage2 = 100
    max_rate = []
    if gpu_npu_input == None:
        print(f"GPU time layer {current_layer_cnt} batch {current_batch_cnt}: 0 seconds")
        print(f"NPU time layer {current_layer_cnt} batch {current_batch_cnt}: 0 seconds")
    elif current_batch_cnt == 0:
        split_rate = [0.99, 0.01]
        subgraphs = split_graph_by_edge_proportion(gpu_npu_input, 2, split_rate)
        ##################################    
        # Pipeline stage2 (GPU)
        ##################################
        gpu_input = subgraphs[0]
        gpu_input.x = torch.randn((gpu_input.num_nodes, in_channels))
        npu_input = subgraphs[1]
        npu_input.x = torch.randn((npu_input.num_nodes, in_channels))
        npu_input_padding = pad_to_smallest_matching_model(npu_input, 'unique_padded_sizes.csv')
        model_gpu = "linear"
        ir_model_dir_gpu = f"Ir/sage_{model_gpu}_layer_{current_layer}_dynamic.xml"
        if not os.path.exists(ir_model_dir_gpu):
            raise FileNotFoundError(f"IR model not found for layer {current_layer} with {current_batch}.")
        # Load the corresponding compiled model for GPU
        compiled_model_gpu = core.compile_model(ir_model_dir_gpu, "GPU")  # GPU model

        start_time = time.time()
        # Perform linear transformation on GPU using OpenVINO
        inputs_gpu = {
            compiled_model_gpu.inputs[0].any_name: gpu_input.x.numpy(),
            compiled_model_gpu.inputs[1].any_name: gpu_input.x.numpy()
        }
        infer_request_gpu = compiled_model_gpu.create_infer_request()
        infer_request_gpu.start_async(inputs=inputs_gpu)
        infer_request_gpu.wait()
        output_tensor_gpu = infer_request_gpu.get_output_tensor(0).data
        output_gpu = torch.tensor(output_tensor_gpu)
        # print(f"output_tensor_gpu {output_tensor_gpu}")
        gpu_end_time = time.time() - start_time
        # print(f"GPU:NPU workload={round(split_rate[0], 2)}:{round(split_rate[1], 2)} GPU time layer {current_layer_cnt} batch {current_batch_cnt}: {gpu_end_time:.4f} seconds")
        timings_stage2.append(gpu_end_time)

        ##################################    
        # Pipeline stage2 (NPU)
        ##################################
        # Padding the input & load the model for NPU
        num_nodes_npu = npu_input_padding.num_nodes
        num_edges_npu = npu_input_padding.edge_index.size(1)
        x_npu = npu_input_padding.x
        edge_index_npu = npu_input_padding.edge_index
        feature_size_list = [500, 256]
        feature_size = feature_size_list[current_layer_cnt]

        model_name = "linear"
        # Create dummy inputs based on model input type
        if 'propagate' in model_name.lower() or 'sage' in model_name.lower():
            # For GNN models with 'propagate' or 'sage'
            x_npu = npu_input_padding.x
            edge_index_npu = npu_input_padding.edge_index
            inputs = (x_npu, edge_index_npu)
            ir_path_npu = os.path.join(ir_dir, f"test/{model_name}_nodes_{num_nodes_npu}_features_{feature_size}_edges_{num_edges_npu}.xml")
        elif 'linear' in model_name.lower():
            # For linear models
            message_npu = torch.randn((num_nodes_npu, feature_size))
            x_npu = npu_input_padding.x
            inputs = (message_npu, x_npu)
            ir_path_npu = os.path.join(ir_dir, f"test/{model_name}_nodes_{num_nodes_npu}_features_{feature_size}.xml")
            # Find the smallest model that satisfies the node and edge requirements
       
        # xml_path_npu = os.path.join(ir_model_dir_npu, f"propagate_layer_{current_layer}_nodes_{num_nodes_npu}_edges_{num_edges_npu}.xml")  # Adjust as needed
        if not os.path.exists(ir_path_npu):
            raise FileNotFoundError(f"IR model not found for layer {current_layer} with {num_nodes_npu} nodes and {num_edges_npu} edges.")
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
        timings_stage2.append(npu_end_time)

        max_end_time = max([gpu_end_time, npu_end_time])
        print(f"GPU:NPU workload={round(split_rate[0], 2)}:{round(split_rate[1], 2)} GPU: {gpu_end_time:.4f}s, NPU: {npu_end_time:.4f}s, max: {max_end_time:.4f}s")
        timings.append(timings_stage2[0])  
        timings.append(timings_stage2[1]) 
        print(f"Stage2: GPU:NPU workload={round(split_rate[0], 2)}:{round(split_rate[1], 2)}")
        print(f"GPU time layer {current_layer_cnt} batch {current_batch_cnt}: {timings_stage2[0]:.4f} seconds")
        print(f"NPU time layer {current_layer_cnt} batch {current_batch_cnt}: {timings_stage2[1]:.4f} seconds")
    else:
        for i in range(5):
            split_rate = [split_rate_init[0]+0.2*i, split_rate_init[1]-0.2*i]
            ##################################    
            # Pipeline stage2 (GPU)
            ##################################
            subgraphs = split_graph_by_edge_proportion(gpu_npu_input, 2, split_rate)
            gpu_input = subgraphs[0]
            gpu_input.x = torch.randn((gpu_input.num_nodes, in_channels))
            npu_input = subgraphs[1]
            npu_input.x = torch.randn((npu_input.num_nodes, in_channels))
            npu_input_padding = pad_to_smallest_matching_model(npu_input, 'unique_padded_sizes.csv')
            model_gpu = "linear"
            ir_model_dir_gpu = f"Ir/sage_{model_gpu}_layer_{current_layer}_dynamic.xml"
            if not os.path.exists(ir_model_dir_gpu):
                raise FileNotFoundError(f"IR model not found for layer {current_layer} with {current_batch}.")
            # Load the corresponding compiled model for GPU
            compiled_model_gpu = core.compile_model(ir_model_dir_gpu, "CPU")  # GPU model

            start_time = time.time()
            # Perform linear transformation on GPU using OpenVINO
            inputs_gpu = {
                compiled_model_gpu.inputs[0].any_name: gpu_input.x.numpy(),
                compiled_model_gpu.inputs[1].any_name: gpu_input.x.numpy()
            }
            infer_request_gpu = compiled_model_gpu.create_infer_request()
            infer_request_gpu.start_async(inputs=inputs_gpu)
            infer_request_gpu.wait()
            output_tensor_gpu = infer_request_gpu.get_output_tensor(0).data
            output_gpu = torch.tensor(output_tensor_gpu)
            # print(f"output_tensor_gpu {output_tensor_gpu}")
            gpu_end_time = time.time() - start_time
            # print(f"GPU:NPU workload={round(split_rate[0], 2)}:{round(split_rate[1], 2)} GPU time layer {current_layer_cnt} batch {current_batch_cnt}: {gpu_end_time:.4f} seconds")
            timings_stage2.append(gpu_end_time)
        

            ######################
            # Pipeline stage3 (NPU)
            ######################
            # Padding the input & load the model for NPU
            num_nodes_npu = npu_input_padding.num_nodes
            num_edges_npu = npu_input_padding.edge_index.size(1)
            x_npu = npu_input_padding.x
            edge_index_npu = npu_input_padding.edge_index
            # Find the smallest model that satisfies the node and edge requirements
            feature_size_list = [500, 256]
            feature_size = feature_size_list[current_layer_cnt]

            model_name = "linear"
            # Create dummy inputs based on model input type
            if 'propagate' in model_name.lower() or 'sage' in model_name.lower():
                # For GNN models with 'propagate' or 'sage'
                x_npu = npu_input_padding.x
                edge_index_npu = npu_input_padding.edge_index
                inputs = (x_npu, edge_index_npu)
                ir_path_npu = os.path.join(ir_dir, f"tset/{model_name}_nodes_{num_nodes_npu}_features_{feature_size}_edges_{num_edges_npu}.xml")
            elif 'linear' in model_name.lower():
                # For linear models
                message_npu = torch.randn((num_nodes_npu, feature_size))
                x_npu = npu_input_padding.x
                inputs = (message_npu, x_npu)
                ir_path_npu = os.path.join(ir_dir, f"test/{model_name}_nodes_{num_nodes_npu}_features_{feature_size}.xml")
            # Find the smallest model that satisfies the node and edge requirements
       
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
            npu_end_time = (time.time() - start_time)/10
            max_end_time = max([gpu_end_time, npu_end_time])
            print(f"GPU:NPU workload={round(split_rate[0], 2)}:{round(split_rate[1], 2)} GPU: {gpu_end_time:.4f}s, NPU: {npu_end_time:.4f}s, max: {max_end_time:.4f}s")
          
            if max_timing_stage2 > max_end_time:
                max_timing_stage2 = max_end_time
                timings_stage2 = [gpu_end_time, npu_end_time]
                max_rate = split_rate
        timings.append(timings_stage2[0])  
        timings.append(timings_stage2[1]) 
        print(f"Stage2: GPU:NPU workload={round(max_rate[0], 2)}:{round(max_rate[1], 2)}")
        print(f"GPU time layer {current_layer_cnt} batch {current_batch_cnt}: {timings_stage2[0]:.4f} seconds")
        print(f"NPU time layer {current_layer_cnt} batch {current_batch_cnt}: {timings_stage2[1]:.4f} seconds")

    # xs.append(output_npu)
    # x_all = torch.cat(xs, dim=0)
    print(f"Max time layer {current_layer_cnt} batch {current_batch_cnt}: {max(timings):.4f} seconds\n")
    return x_all, timings

# Parse command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process layer and batch information.")
    parser.add_argument('--current_layer', type=int, help='The index of the current layer to process.')
    parser.add_argument('--current_batch', type=int, help='The index of the current batch to process.')
    
    # Parse arguments
    args = parser.parse_args()

    current_layer = args.current_layer
    current_batch = args.current_batch

    # Load the dataset
    dataset = Flickr(root="data/Flickr")
    data = dataset[0]

    # Define the NeighborLoader
    subgraph_loader = NeighborLoader(
        data,
        num_neighbors=[-1],
        batch_size=4096,  # Adjust as needed
        input_nodes=None
    )

    # Define the SAGE model
    model = SAGE(
        in_channels=dataset.num_features,
        hidden_channels=256,
        out_channels=dataset.num_classes,
        num_layers=2
    )

    # Collect unique padded sizes
    multiple = 5000
    unique_sizes = collect_unique_padded_sizes(subgraph_loader, multiple)

    # Save unique sizes to CSV
    df_unique_sizes = pd.DataFrame(list(unique_sizes), columns=['Number of Nodes', 'Number of Edges'])
    df_unique_sizes.to_csv('unique_padded_sizes.csv', index=False)

    # Define directories
    onnx_dir = "Onnx"
    if not os.path.exists(onnx_dir):
        os.makedirs(onnx_dir)
    ir_dir = "Ir"

    # # Export models
    # export_models_for_unique_sizes(model, df_unique_sizes, onnx_dir)

    # # Compile IR models
    # compile_ir_models_for_unique_sizes(df_unique_sizes, model.num_layers, onnx_dir, ir_dir)

    # Run inference
    output, timings = run_inference_with_dynamic_loading(model, subgraph_loader, multiple, ir_dir, current_layer, current_batch)
    
    # write to file
    with open('mix_execution_times.txt', 'a') as f:
        f.write(f"Layer: {current_layer}, Batch: {current_batch}, Time: {max(timings):.4f} seconds\n")
