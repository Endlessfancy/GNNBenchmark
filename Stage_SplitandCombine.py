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



class GCNNorm(torch.nn.Module):
    def __init__(self, improved=False, cached=False, add_self_loops=True, normalize=True):
        super(GCNNorm, self).__init__()
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x, edge_index, edge_weight=None):
        # Compute num_nodes from edge_indesx
        num_nodes = torch.max(edge_index) + 1

        # Part 1: Normalize (gcn_norm)
        if self.normalize:
            if isinstance(edge_index, torch.Tensor):
                if self._cached_edge_index is None:
                    # Ensure edge_weight is a tensor
                    if edge_weight is None:
                        edge_weight = torch.ones(
                            (edge_index.size(1),), dtype=torch.float32 #, device=edge_index.device
                        )
                    # Apply gcn_norm
                    
                    edge_index, edge_weight = gcn_norm(
                        edge_index, edge_weight, num_nodes, self.improved, self.add_self_loops
                    )
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = self._cached_edge_index
            elif isinstance(edge_index, SparseTensor):
                if self._cached_adj_t is None:
                    # Apply gcn_norm for SparseTensor
                    edge_index = gcn_norm(
                        edge_index, edge_weight, num_nodes, self.improved, self.add_self_loops
                    )
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = self._cached_adj_t

        return x, edge_index, edge_weight

class MLPUpdate(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLPUpdate, self).__init__()
        self.lin = Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        return self.lin(x), edge_index
    
class MessageGeneration(torch.nn.Module):
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        # Get source node indices
        source_nodes = edge_index[0]  # Shape: [num_edges]
        # Obtain source node features
        x_j = x[source_nodes]  # Shape: [num_edges, num_features]
        # Messages are x_j
        messages = self.message(x_j)
        print(f"Message Generation operation: messages: {messages.shape}")
        return x, edge_index, messages

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j

class MessageAggregation(MessagePassing):
    def __init__(self, aggr: str = 'add', **kwargs):
        super().__init__(aggr=aggr, **kwargs)

    def forward(self, x_pre_message: torch.Tensor, edge_index: torch.Tensor, messages: torch.Tensor):
        index = edge_index[1]  # Assuming edge_index follows the [source, target] format
        # Perform aggregation
        out = self.aggregate(messages, index, dim_size=x_pre_message.size(0))
        # Update node embeddings after aggregation
        out = self.update(out)
        return out

    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor, ptr: torch.Tensor = None, dim_size: int = None):
        return scatter_add(inputs, index, dim=self.node_dim, dim_size=dim_size)
    
    def update(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.relu(inputs)

# -----------------------
# test inference for each stage
# ----------------------
# # Load the Flickr dataset
# dataset = Flickr(root='data/Flickr')
# data = dataset[0]

# # Ensure the edge index is undirected
# data.edge_index = to_undirected(data.edge_index)
# # Prepare the Flickr data
# # x = data.x  # Node features
# # edge_index = data.edge_index  # Edge index
# edge_weight = None
# # Step 1: Apply GCN normalization
# gcn_norm_model = GCNNorm(cached=False)
# with torch.no_grad():
#     for param in gcn_norm_model.parameters():
#         param.fill_(1)
# data.x, norm_edge_index, norm_edge_weight = gcn_norm_model(data.x, data.edge_index, edge_weight)

# print(f"Normalized Edge Index: {norm_edge_index.size()}")
# print(f"Normalized Edge Weight: {norm_edge_weight.size()}")

# # Step 2: Apply linear transformation
# linear_model = MLPUpdate(in_channels=dataset.num_node_features, out_channels=16)
# with torch.no_grad():
#     for param in linear_model.parameters():
#         param.fill_(1)
# x_linear = linear_model(data.x)
# print(f"Linear operation: Input size: {data.x.size()}")
# print(f"Linear operation: Output size: {x_linear.size()}")

# # Model 3: MessageGeneration
# before_msg_model = MessageGeneration()
# x_pre_message, norm_edge_index, messages = before_msg_model(x_linear, data.edge_index)
# print(f"Message Generation operation: x_pre_message: {x_pre_message.size()}")
# print(f"Message Generation operation: norm_edge_index: {norm_edge_index.size()}")
# print(f"Message Generation operation: msg_kwargs: {messages}")

# # Model 4: MessageAggregation
# after_msg_model = MessageAggregation()
# output = after_msg_model(x_pre_message, norm_edge_index, messages)

# print("Final output shape:", output.shape)

# -----------------------
# define model combinatio class
# ----------------------
class CombinedModel(nn.Module):
    def __init__(self, selected_models):
        super(CombinedModel, self).__init__()
        self.models = nn.ModuleList(selected_models)

    def forward(self, *inputs):
        # Check the type of the first layer and set inputs accordingly
        if isinstance(self.models[0], GCNNorm):
            x, edge_index, edge_weight = inputs
            x, edge_index, edge_weight = self.models[0](x, edge_index, edge_weight)
        elif isinstance(self.models[0], MLPUpdate):
            x, edge_index = inputs
            x, edge_index = self.models[0](x, edge_index)
        elif isinstance(self.models[0], MessageGeneration):
            x, edge_index = inputs
            x, edge_index, messages = self.models[0](x, edge_index)
        elif isinstance(self.models[0], MessageAggregation):
            x_pre_message, edge_index, messages = inputs
            x = self.models[0](x_pre_message, edge_index, messages)

        # Pass through the remaining models
        for model in self.models[1:]:
            if isinstance(model, GCNNorm):
                x, edge_index, edge_weight = model(x, edge_index, edge_weight)
            elif isinstance(model, MLPUpdate):
                x, edge_index = model(x, edge_index)
            elif isinstance(model, MessageGeneration):
                x, edge_index, messages = model(x, edge_index)
            elif isinstance(model, MessageAggregation):
                x = model(x, edge_index, messages)

        # Check the type of the last layer and return outputs accordingly
        if isinstance(self.models[-1], GCNNorm):
            return x, edge_index, edge_weight
        elif isinstance(self.models[-1], MLPUpdate):
            return x, edge_index
        elif isinstance(self.models[-1], MessageGeneration):
            return x, edge_index, messages
        elif isinstance(self.models[-1], MessageAggregation):
            return x
# -----------------------
# test model combinatio class inference
# ----------------------
# Load the Flickr dataset
dataset = Flickr(root='data/Flickr')
data = dataset[0]
  
# Define individual models
gcn_norm_model = GCNNorm(cached=False)
linear_model = MLPUpdate(in_channels=dataset.num_node_features, out_channels=16)
before_msg_model = MessageGeneration()
after_msg_model = MessageAggregation()

# Test forward pass with combined models
def test_combined_model_output(combined_model, selected_model_list, data):
    # define the first layer and last layer:
    first_layer = selected_model_list[0]
    last_layer = selected_model_list[-1]

    # Define input for the first layer
    dummy_inputs = None
    
    # Set dummy input based on the first layer
    if isinstance(first_layer, GCNNorm):
        dummy_x = data.x
        dummy_edge_index = data.edge_index
        dummy_edge_weight = torch.ones(data.edge_index.size(1))  # Example edge weight
        dummy_inputs = (dummy_x, dummy_edge_index, dummy_edge_weight)
        print("Testing combined model with GCNNorm as the first layer")
        
    elif isinstance(first_layer, MLPUpdate):
        dummy_x = data.x
        dummy_edge_index = data.edge_index
        dummy_inputs = (dummy_x, dummy_edge_index)
        print("Testing combined model with MLPUpdate as the first layer")
        
    elif isinstance(first_layer, MessageGeneration):
        dummy_x = data.x
        dummy_edge_index = data.edge_index
        dummy_inputs = (dummy_x, dummy_edge_index)
        print("Testing combined model with MessageGeneration as the first layer")
        
    elif isinstance(first_layer, MessageAggregation):
        dummy_x = data.x
        dummy_edge_index = data.edge_index
        dummy_messages = torch.randn(dummy_edge_index.size(1), dummy_x.size(1))  # Example messages
        dummy_inputs = (dummy_x, dummy_edge_index, dummy_messages)
        print("Testing combined model with MessageAggregation as the first layer")
    
    # Run the forward pass on the combined model
    if dummy_inputs:
        try:
            output = combined_model(*dummy_inputs)
            print(f"Successfully tested the combined model with output: {output}")
        except Exception as e:
            print(f"Error encountered during model testing: {str(e)}")
    else:
        print("Invalid inputs for the combined model")

# model_sequence = []
# for i in range(1, 5):
#     for j in range(i, 5):
#         model_sequence = list(range(i, j + 1))
#         print(model_sequence)
#         # Map the numbers to actual models
#         model_map = {
#             1: gcn_norm_model,
#             2: linear_model,
#             3: before_msg_model,
#             4: after_msg_model
#         }
#         selected_model_list = []
#         for model_number in model_sequence:
#             selected_model_list.append(model_map[model_number])
#         # print(selected_model_list)

#         # Combine models into one sequence
#         combined_model = CombinedModel(selected_model_list)

#         test_combined_model_output(combined_model, selected_model_list, data)

# -------------------------------
# Export the model
# -------------------------------
def export_combined_model_to_onnx(combined_model, selected_model_list, data, model_onnx_path):
    # define the first layer and last layer:
    first_layer = selected_model_list[0]
    last_layer = selected_model_list[-1]

    # Define input for the first layer
    dummy_inputs = None
    input_names = []
    output_names = []
    dynamic_axes = {}
    
    # Set dummy input and input names based on the first layer
    if isinstance(first_layer, GCNNorm):
        dummy_x = data.x
        dummy_edge_index = data.edge_index
        dummy_edge_weight = torch.ones(data.edge_index.size(1))  # Example edge weight
        dummy_inputs = (dummy_x, dummy_edge_index, dummy_edge_weight)
        input_names = ['x', 'edge_index', 'edge_weight']
        dynamic_axes = {
            'x': {0: 'num_nodes', 1: 'features'},
            'edge_index': {1: 'num_edges'},
            'norm_edge_index': {1: 'num_edges'},
            'norm_edge_weight': {0: 'num_edges'}
        }
    elif isinstance(first_layer, MLPUpdate):
        dummy_x = data.x
        dummy_edge_index = data.edge_index
        dummy_inputs = (dummy_x, dummy_edge_index)
        input_names = ['x', 'num_edges']
        dynamic_axes = {
            'x': {0: 'num_nodes', 1: 'features'},
            'edge_index': {1: 'num_edges'},
            'out_edge_index': {1: 'num_edges'},
            'x_linear': {0: 'num_nodes', 1: 'output_features'}
        }
    elif isinstance(first_layer, MessageGeneration):
        dummy_x = data.x
        dummy_edge_index = data.edge_index
        dummy_inputs = (dummy_x, dummy_edge_index)
        input_names = ['x', 'edge_index']
        dynamic_axes = {
            'x': {0: 'num_nodes', 1: 'features'},
            'edge_index': {1: 'num_edges'},
            'x_pre_message': {0: 'num_nodes', 1: 'features'},
            'messages': {0: 'num_edges'}
        }
    elif isinstance(first_layer, MessageAggregation):
        dummy_x = data.x
        dummy_edge_index = data.edge_index
        dummy_messages = torch.randn(dummy_edge_index.size(1), dummy_x.size(1))  # Example messages
        dummy_inputs = (dummy_x, dummy_edge_index, dummy_messages)
        input_names = ['x', 'edge_index', 'messages']
        dynamic_axes = {
            'x': {0: 'num_nodes', 1: 'features'},
            'edge_index': {1: 'num_edges'},
            'messages': {0: 'num_edges'},
            'output': {0: 'num_nodes'}
        }

    # Set output names based on the last layer
    if isinstance(last_layer, GCNNorm):
        output_names = ['x_out', 'norm_edge_index', 'norm_edge_weight']
        dynamic_axes.update({
            'x_out': {0: 'num_nodes', 1: 'features'},
            'norm_edge_index': {1: 'num_edges'},
            'norm_edge_weight': {0: 'num_edges'}
        })
    elif isinstance(last_layer, MLPUpdate):
        output_names = ['x_linear', 'out_edge_index']
        dynamic_axes.update({
            'out_edge_index': {1: 'num_edges'},
            'x_linear': {0: 'num_nodes', 1: 'output_features'}
        })
    elif isinstance(last_layer, MessageGeneration):
        output_names = ['x_pre_message', 'edge_index', 'messages']
        dynamic_axes.update({
            'x_pre_message': {0: 'num_nodes', 1: 'features'},
            'messages': {0: 'num_edges'}
        })
    elif isinstance(last_layer, MessageAggregation):
        output_names = ['output']
        dynamic_axes.update({
            'output': {0: 'num_nodes'}
        })

    # Export the model to ONNX
    torch.onnx.export(
        combined_model,
        dummy_inputs,
        model_onnx_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=11,
        dynamic_axes=dynamic_axes
    )

    print(f"Exported combined model with {first_layer.__class__.__name__} as the first layer and {last_layer.__class__.__name__} as the last layer.")


# Export combined model with dynamic input/output based on first and last layers
# export_combined_model_to_onnx(combined_model, selected_model_list, data)
# model_sequence = []
# for i in range(1, 5):
#     for j in range(i, 5):
#         model_sequence = list(range(i, j + 1))
#         print(model_sequence)
#         # Map the numbers to actual models
#         model_map = {
#             1: gcn_norm_model,
#             2: linear_model,
#             3: before_msg_model,
#             4: after_msg_model
#         }
#         selected_model_list = []
#         for model_number in model_sequence:
#             selected_model_list.append(model_map[model_number])
#         # print(selected_model_list)

#         # Combine models into one sequence
#         combined_model = CombinedModel(selected_model_list)

#         export_combined_model_to_onnx(combined_model, selected_model_list, data)

# Export the [1, 2], [3], [4]
model_sequence = [[1, 2], [3], [4]]
model_map = {
    1: gcn_norm_model,
    2: linear_model,
    3: before_msg_model,
    4: after_msg_model
}
for models in model_sequence:
    selected_model_list = []
    model_onnx_path = "Onnx/Combined_model_"
    for model_number in models:
        selected_model_list.append(model_map[model_number])
        model_onnx_path += str(model_number)
    model_onnx_path += ".onnx"
    print(model_onnx_path)

    # Combine models into one sequence
    combined_model = CombinedModel(selected_model_list)
    export_combined_model_to_onnx(combined_model, selected_model_list, data, model_onnx_path)
