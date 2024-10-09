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
from torch_geometric.datasets import Reddit2
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
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
import onnx
from torch_geometric.nn import GATConv
from sage_split import PropagateModel_10, LinearModel_10
from gat import *

# Define the function to check ONNX model's dynamic axes
def check_onnx_dynamic_axes(onnx_path):
    model = onnx.load(onnx_path)
    print(f"Checking ONNX model: {onnx_path}")
    for input in model.graph.input:
        dims = input.type.tensor_type.shape.dim
        dim_strs = [str(dim.dim_param or dim.dim_value) for dim in dims]
        print(f"Input '{input.name}' has shape: {', '.join(dim_strs)}")

# Function to export the GAT components to ONNX dynamically
def export_onnx_dynamic(model, model_name, feature_size, num_nodes, num_heads, out_channels, onnx_path_dynamic):
    num_nodes = data.x.size(0)
    num_edges = data.edge_index.size(1)
    # Set up dummy inputs based on the component
    if model_name == "gat_propagate": 
        x_dummy = torch.randn((num_nodes, num_heads, out_channels))
        edge_index_dummy = torch.randint(0, num_nodes, (2, (num_nodes + num_edges)))
        alpha_dummy = torch.randn(((num_nodes + num_edges), num_heads))
        input_names = ['x', 'edge_index', 'alpha']
        dynamic_axes_list = {'x': {0: 'num_nodes', 2: 'num_features'}, 'edge_index': {1: 'num_edges'}, 'alpha': {0: 'num_nodes'}}
        inputs = (x_dummy, edge_index_dummy, alpha_dummy)
    elif model_name == "gat_full": 
        x_dummy = torch.randn((num_nodes, feature_size))
        edge_index_dummy = torch.randint(0, num_nodes, (2, (num_edges)))
        input_names = ['x', 'edge_index']
        dynamic_axes_list = {'x': {0: 'num_nodes'}, 'edge_index': {1: 'num_edges'}}
        inputs = (x_dummy, edge_index_dummy)
    elif model_name == "gat_linear":
        x_dummy = torch.randn((num_nodes, feature_size))
        input_names = ['x']
        dynamic_axes_list = {'x': {0: 'num_nodes', 1: 'num_features'}}
        inputs = (x_dummy)
    elif model_name == "GAT_Transform":
        x_dummy = torch.randn((num_nodes, feature_size))
        input_names = ['x']
        dynamic_axes_list = {'x': {0: 'num_nodes', 1: 'num_features'}}
        inputs = (x_dummy)
    elif model_name == "GAT_Attention":
        x_dummy = torch.randn((num_nodes, num_heads, out_channels))
        input_names = ['x']
        dynamic_axes_list = {'x': {0: 'num_nodes', 1: 'num_features'}}
        inputs = (x_dummy)
    elif model_name == "GAT_SelfLoop":
        num_nodes_dummy = num_nodes
        edge_index_dummy = torch.randint(0, num_nodes, (2, num_nodes * 2))
        input_names = ['num_nodes', 'edge_index']
        dynamic_axes_list = {'edge_index': {1: 'num_edges'}}
        inputs = (num_nodes_dummy, edge_index_dummy)
    elif model_name == "GAT_EdgeWeightUpdate":
        alpha_dummy = torch.randn(((num_nodes + num_edges), num_heads))
        edge_index_dummy = torch.randint(0, num_nodes, (2, (num_nodes + num_edges)))
        print(f"size {alpha_dummy.size()}, {edge_index_dummy.size()}")
        input_names = ['alpha', 'edge_index']
        dynamic_axes_list = {'alpha': {0: 'num_nodes'}, 'edge_index': {1: 'num_edges'}}
        # edge_index_dummy = torch.randint(0, num_nodes, (2, num_nodes * 2))
        inputs = (alpha_dummy, edge_index_dummy)
    elif model_name == "GAT_MessagePassing":
        x_dummy = torch.randn((num_nodes, num_heads, out_channels))
        edge_index_dummy = torch.randint(0, num_nodes, (2, (num_nodes + num_edges)))
        alpha_dummy = torch.randn(((num_nodes + num_edges), num_heads))
        input_names = ['x', 'edge_index', 'alpha']
        dynamic_axes_list = {'x': {0: 'num_nodes', 2: 'num_features'}, 'edge_index': {1: 'num_edges'}, 'alpha': {0: 'num_nodes'}}
        inputs = (x_dummy, edge_index_dummy, alpha_dummy)
    elif model_name == "GAT_output":
        out_dummy = torch.randn((num_nodes, num_heads, out_channels))
        print(f"out_dummy : {out_dummy.size()}")
        input_names = ['out']
        dynamic_axes_list = {'out': {0: 'num_nodes', 2: 'num_features'}}
        inputs = (out_dummy)

    # Export to ONNX
    # dummy_inputs = (locals()[f"{name}_dummy"] for name in input_names)
    torch.onnx.export(
        model,
        inputs,
        onnx_path_dynamic,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=input_names,
        output_names=['output'],
        dynamic_axes=dynamic_axes_list
    )

# Function to export to OpenVINO IR format
def export_ir(onnx_path_dynamic, ir_model_path):
    ir_dir = os.path.dirname(ir_model_path)
    if not os.path.exists(ir_dir):
        os.makedirs(ir_dir)
    
    try:
        subprocess.run(['mo', '--input_model', onnx_path_dynamic, '--output_dir', ir_dir], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during OpenVINO model optimization: {e}")

    # if not os.path.exists(ir_model_path):
    #     print(f"Exporting OpenVINO IR model to {ir_model_path}...")
    #     try:
    #         subprocess.run(['mo', '--input_model', onnx_path_dynamic, '--output_dir', ir_dir], check=True)
    #     except subprocess.CalledProcessError as e:
    #         print(f"Error during OpenVINO model optimization: {e}")
    # else:
    #     print(f"IR model {ir_model_path} already exists. Skipping export.")

# Main function for exporting
def export(model, model_name, feature_size, data, num_heads, out_channels, onnx_dynamic_path, ir_model_path):
    # Export ONNX model
    export_onnx_dynamic(model, model_name, feature_size, data, num_heads, out_channels, onnx_dynamic_path)
    # Export OpenVINO IR model
    export_ir(onnx_dynamic_path, ir_model_path)

# Example usage
if __name__ == "__main__":
    dataset_name = "Product"
    num_nodes = 1000  # Example node count
    current_layer = 0
    num_heads = 4
    if dataset_name == "Flickr":
        dataset = Flickr(root="data/Flickr")
        data = dataset[0]

    elif dataset_name == "Reddit2":
        dataset = Reddit2(root="data/Reddit2")
        data = dataset[0]

    elif dataset_name == "Product":
        dataset = PygNodePropPredDataset('ogbn-products', 'data/Product')
        data = dataset[0]
    
    feature_size_list = [dataset.num_features, 256, dataset.num_classes]
    feature_size = feature_size_list[current_layer]
    out_channel_size = feature_size_list[current_layer + 1]
    # Define paths for the models
    model_components = {
        # "GAT_Transform": GAT_Transform(in_channels=feature_size, out_channels=out_channel_size, heads=4),
        # "GAT_Attention": GAT_Attention(in_channels=feature_size, out_channels=out_channel_size, heads=4),
        # "GAT_SelfLoop": GAT_SelfLoop(in_channels=feature_size, out_channels=out_channel_size, heads=4),
        # "GAT_EdgeWeightUpdate": GAT_EdgeWeightUpdate(in_channels=feature_size, out_channels=out_channel_size, heads=4),
        # "GAT_MessagePassing": GAT_MessagePassing(in_channels=feature_size, out_channels=out_channel_size, heads=4),
        # "GAT_output": GAT_output(in_channels=feature_size, out_channels=out_channel_size, heads=4),
        # "gat_propagate": gat_propagate(in_channels=feature_size, out_channels=out_channel_size, heads=4),
        "gat_linear": gat_linear(in_channels=feature_size, out_channels=out_channel_size, heads=4),
        # "gat_full": GATConv(in_channels=dataset.num_features, out_channels=128, heads=4)

    }

    for model_name, model in model_components.items():
        onnx_dynamic_path = f"Onnx/gat_{dataset_name}/{model_name}_features_{feature_size}_dynamic.onnx"
        ir_model_path = f"Ir/gat_{dataset_name}/{model_name}_features_{feature_size}_dynamic.xml"
        export(model, model_name, feature_size, data, num_heads, out_channel_size, onnx_dynamic_path, ir_model_path)
