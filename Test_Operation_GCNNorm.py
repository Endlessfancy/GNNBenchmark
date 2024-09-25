import torch
from torch.nn import Module
from torch_geometric.datasets import Flickr
from torch_geometric.utils import to_undirected
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import onnxruntime as ort
import numpy as np
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.datasets import Flickr
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import to_undirected
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, Size, SparseTensor
import onnxruntime as ort
import onnx
import openvino as ov

class GCNNorm(Module):
    def __init__(self, improved=False, add_self_loops=True):
        super(GCNNorm, self).__init__()
        self.improved = improved
        self.add_self_loops = add_self_loops
        self.node_dim = 0  # Node dimension is 0 for 2D tensors

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        # edge_index = edge_index.to(torch.int64)

        # Calculate num_nodes from x.size(self.node_dim)
        num_nodes = x.size(self.node_dim)

        # Directly use gcn_norm from PyG with default edge weight
        edge_index, edge_weight = gcn_norm(
            edge_index=edge_index, 
            edge_weight=None,  # Use default edge weight of 1
            num_nodes=num_nodes, 
            improved=self.improved, 
            add_self_loops=self.add_self_loops
        )
        return x, edge_index, edge_weight


# Load the Flickr dataset
dataset = Flickr(root="data/Flickr")
data = dataset[0]

# Ensure the edge index is undirected
data.edge_index = to_undirected(data.edge_index)

# Get node features (x) and edge index from the dataset
x = data.x  # Node features
edge_index = data.edge_index  # Edge index

# Instantiate the GCN class
gcn_model = GCNNorm(improved=True, add_self_loops=True)

# Pass x and edge_index to the GCN forward function
x_out, norm_edge_index, norm_edge_weight = gcn_model(x, edge_index)

# Print results
print(f"Normalized Edge Index: {norm_edge_index.size()}")
print(f"Normalized Edge Weight: {norm_edge_weight.size()}\n")

# Dummy inputs for export (edge_index is 2D and x is the node feature matrix)
dummy_x = x  # Use the actual input size
dummy_edge_index = edge_index  # Use the actual edge_index size
edge_index = data.edge_index.to(torch.int64)

# Static ONNX export
onnx_static_path = "Onnx/gcn_norm_static2.onnx"
torch.onnx.export(
    gcn_model,  # Model instance
    (dummy_x, dummy_edge_index),  # Inputs
    onnx_static_path,  # Output ONNX file path
    input_names=["x", "edge_index"],  # Name of the inputs
    output_names=['x_out', 'norm_edge_index', 'norm_edge_weight'],  # output names，
    dynamic_axes=None,  # Disable dynamic axes to ensure static shapes
    opset_version=13  # ONNX ss
)

print(f"Static model exported to {onnx_static_path}!")


# Dynamic ONNX export
onnx_dynamic_path = "Onnx/gcn_norm_dynamic2.onnx"
torch.onnx.export(
    gcn_model,  # Model instance
    (dummy_x, dummy_edge_index),  # Inputs
    onnx_dynamic_path,  # Output ONNX file path
    input_names=["x", "edge_index"],  # Name of the inputs
    output_names=['x_out', 'norm_edge_index', 'norm_edge_weight'],  # output names
    dynamic_axes={  # Dynamic dimensions for variable-sized inputs
        'x': {0: 'num_nodes', 1: 'num_features'},  # Dynamically sized node features
        'x_out': {0: 'num_nodes', 1: 'num_features'},  # Dynamically sized node features
        'edge_index': {1: 'num_edges'},  # Dynamically sized edge index
        'norm_edge_index': {1: 'num_edges'},  # Dynamically sized output edge index
        'norm_edge_weight': {0: 'num_edges'}  # Dynamically sized edge weight
    },
    opset_version=13  # ONNX version
)

print(f"Dynamic model exported to {onnx_dynamic_path}!\n")


# Convert the input data to NumPy arrays as ONNX expects NumPy inputs
# Prepare input data for the model (example using NumPy)
x_numpy = data.x.numpy().astype(np.float32)
edge_index_numpy = data.edge_index.numpy().astype(np.int64)


# Prepare input for ONNX model
# Prepare inputs (assuming the ONNX model expects 'x' and 'edge_index' as inputs)
inputs = {
    'x': x_numpy,
    'edge_index': edge_index_numpy
}


# Load the ONNX model
onnx_model_path = "Onnx/gcn_norm_static2.onnx"  # or the path to your model
ort_session = ort.InferenceSession(onnx_model_path)

# Print the input names of the ONNX model
for input_meta in ort_session.get_inputs():
    print(f"Input name: {input_meta.name}, Input shape: {input_meta.shape}")

# Run inference
x_out, norm_edge_index, norm_edge_weight = ort_session.run(None, inputs)

# Print the results of inference
print(f"x_out Index: {x_out.shape}")
print(f"Normalized Edge Index: {norm_edge_index.shape}")
print(f"Normalized Edge Weight: {norm_edge_weight.shape}\n")


# # Test a specific model
device = "NPU"
model_ir_path = "Ir/gcn_norm_dynamic2.xml"
core = ov.Core()

# Load the Combined Model with 1 (GCNNorm) and 2 (MLPUpdate)
compiled_model = core.compile_model(model=model_ir_path, device_name=device)
print(f"Model {model_ir_path} loaded to {device}!")

# Prepare input data
input_data = data.x.numpy()  # Node features
edge_index = data.edge_index.numpy()  # Edge index
dummy_edge_weight = np.ones(data.edge_index.size(1))  # Example edge weight

# Create OpenVINO tensors
input_tensor = ov.Tensor(array=input_data)
edge_index_tensor = ov.Tensor(array=edge_index)
dummy_edge_weight_tensor = ov.Tensor(array=dummy_edge_weight)

# Create inference request and set input tensors
infer_request = compiled_model.create_infer_request()
infer_request.set_tensor(compiled_model.input(0), input_tensor)  # x
infer_request.set_tensor(compiled_model.input(1), edge_index_tensor)  # edge_index
# infer_request_12.set_tensor(compiled_model_12.input(2), dummy_edge_weight_tensor)  # edge_weight

# Perform inference
infer_request.start_async()
infer_request.wait()

# Get and print the result
output_tensor = infer_request.get_output_tensor(0).data
# res = infer_request.get_output_tensor(0).data
output = output_tensor.data
print(f"Output of combined model: {output.shape}\n")
