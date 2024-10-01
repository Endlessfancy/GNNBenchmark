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
import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Flickr
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, Size, SparseTensor
import torch.onnx
import subprocess
from openvino.runtime import Core
from openvino.runtime import Tensor  # Import Tensor class from OpenVINO
import time

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
        for i in range(10):
            for conv in self.convs:
                x = conv(x, edge_index)
                x = x.relu()
        return x
    
class PropagateModel_10(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr="mean"):
        super().__init__(aggr=aggr)

    def forward(self, x: Tensor, edge_index: Adj, size: Size = None) -> Tensor:
        return self.propagate(edge_index, x=x, size=size)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)
    
class LinearModel_10(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias_l=True, bias_r=False):
        super().__init__()
        self.lin_l = Linear(in_channels, out_channels, bias=bias_l)  # For neighbor features
        self.lin_r = Linear(in_channels, out_channels, bias=bias_r)  # For root node features
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, message: Tensor, x: Tensor) -> Tensor:
        for i in range(10):
            # Apply linear transformation to aggregated neighbor features
            out = self.lin_l(message)
            
            # # Add the transformed root node (self-node) features
            out += self.lin_r(x)

        return out
    
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


# Function to export a single ONNX model for given nodes, edges, and feature size
# export the onnx, detect the different model and input
def export_model_to_onnx(model_name, model, num_nodes, num_edges, feature_size, onnx_path, onnx_ir):
    # Skip exporting if ONNX file already exists
    if os.path.exists(onnx_path):
        print(f"ONNX model already exists at {onnx_path}. Skipping export.")
        return
    
    # Create the IR output directory if it doesn't exist
    if not os.path.exists(onnx_dir):
        os.makedirs(onnx_dir)

    # Create dummy inputs based on model input type
    if 'propagate' in model_name.lower() or 'sage' in model_name.lower():
        # For GNN models with 'propagate' or 'sage'
        dummy_x = torch.randn((num_nodes, feature_size))
        dummy_edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
        input_names = ['x', 'edge_index']
        inputs = (dummy_x, dummy_edge_index)
    elif 'linear' in model_name.lower():
        # For linear models
        dummy_out = torch.randn((num_nodes, feature_size))
        dummy_x = torch.randn((num_nodes, feature_size))
        input_names = ['out', 'x']
        inputs = (dummy_out, dummy_x)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    
    # Export the model
    print(f"Exporting ONNX model to {onnx_path} for {model_name} with {num_nodes} nodes, {feature_size} feature_size and {num_edges} edges.")
    torch.onnx.export(
        model,
        inputs,
        onnx_path,
        input_names=input_names,
        output_names=['output'],
        dynamic_axes=None,  # Static shapes
        opset_version=11
    )

# Function to compile ONNX file to IR using OpenVINO Model Optimizer
def compile_onnx_to_ir(onnx_path, ir_path, ir_dir):
    if os.path.exists(ir_path):
        print(f"IR model {ir_path} already exists. Skipping IR compilation.")
        return

    # Create the IR output directory if it doesn't exist
    if not os.path.exists(ir_dir):
        os.makedirs(ir_dir)

    # Compile ONNX to IR using the OpenVINO Model Optimizer
    print(f"Compiling IR model for {onnx_path} to {ir_dir}.")
    try:
        subprocess.run(
            ['mo', '--input_model', onnx_path, '--output_dir', ir_dir],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error during OpenVINO model optimization: {e}")

# Function to process combinations of node/edge pairs from df_unique_sizes and call export_model_to_onnx
def export_onnx_combinations(model_name, model, df_unique_sizes, onnx_dir, ir_dir, feature_size):
    # for feature_size in feature_size_list:
    for index, row in df_unique_sizes.iterrows():
        num_nodes = int(row['Number of Nodes'])
        num_edges = int(row['Number of Edges'])

        # Construct ONNX filename based on model type and size combinations
        if 'propagate' in model_name.lower() or 'sage' in model_name.lower():
            onnx_path = os.path.join(
                onnx_dir,
                f"{model_name}_nodes_{num_nodes}_features_{feature_size}_edges_{num_edges}.onnx"
            )
            ir_path = os.path.join(    
                ir_dir,
                f"{model_name}_nodes_{num_nodes}_features_{feature_size}_edges_{num_edges}.xml"
            )
        elif 'linear' in model_name.lower():
            onnx_path = os.path.join(
                onnx_dir,
                f"{model_name}_nodes_{num_nodes}_features_{feature_size}.onnx"
            )
            ir_path = os.path.join(    
                ir_dir,
                f"{model_name}_nodes_{num_nodes}_features_{feature_size}.xml"
            )
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

        # Call the function to export the ONNX model
        export_model_to_onnx(model_name, model, num_nodes, num_edges, feature_size, onnx_path, onnx_dir)

        # Compile the ONNX model to IR format
        compile_onnx_to_ir(onnx_path, ir_path, ir_dir)


# Main execution
if __name__ == "__main__":
    ######################################################
    # Input Parameter 
    #####################################################
    model_name = "linear"
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

    # Save unique sizes to CSV
    df_unique_sizes = pd.DataFrame(list(unique_sizes), columns=['Number of Nodes', 'Number of Edges'])
    df_unique_sizes.to_csv('unique_padded_sizes.csv', index=False)
    print("Exported unique padded sizes to 'unique_padded_sizes.csv'.")

    for model, feature_size in zip(model_list, feature_size_list):
        # print(f"model input size : {model.in_channels}")
        export_onnx_combinations(model_name, model, df_unique_sizes, onnx_dir, ir_dir, feature_size)

    # model_name = "linear"
    # # load the model
    # model_list = []
    # if model_name == "propagate":
    #     model_1 = PropagateModel_10(in_channels=dataset.num_features, out_channels=256)
    #     model_2 = PropagateModel_10(in_channels=256, out_channels=dataset.num_classes)
    #     model_list.append(model_1)
    #     model_list.append(model_2)
    # elif model_name == "linear":
    #     model_1 = LinearModel_10(in_channels=dataset.num_features, out_channels=256)
    #     model_2 = LinearModel_10(in_channels=256, out_channels=dataset.num_classes)
    #     model_list.append(model_1)
    #     model_list.append(model_2)
    # for model, feature_size in zip(model_list, feature_size_list):
    #     export_onnx_combinations(model_name, model, df_unique_sizes, onnx_dir, ir_dir, feature_size)

    
