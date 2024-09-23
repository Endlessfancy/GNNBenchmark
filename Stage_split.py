import torch
from torch_geometric.datasets import Flickr
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import to_undirected
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, Size, SparseTensor


# class GCNNorm(torch.nn.Module):
#     def __init__(self, improved=False, cached=False, add_self_loops=True):
#         super(GCNNorm, self).__init__()
#         self.improved = improved
#         self.cached = cached
#         self.add_self_loops = add_self_loops

#     def forward(self, edge_index, edge_weight=None, num_nodes=None):
#         edge_index, edge_weight = gcn_norm(edge_index, edge_weight, num_nodes, self.improved, self.add_self_loops)
#         return edge_index, edge_weight

# class GCNNorm(torch.nn.Module):
#     def __init__(self, improved=False, cached=False, add_self_loops=True, normalize=True):
#         super(GCNNorm, self).__init__()
#         self.improved = improved
#         self.cached = cached
#         self.add_self_loops = add_self_loops
#         self.normalize = normalize
#         self._cached_edge_index = None
#         self._cached_adj_t = None

#     def forward(self, edge_index, edge_weight=None, num_nodes=None):
#         # Part 1: Normalize (gcn_norm), similar to customGCNConv
#         if self.normalize:
#             if isinstance(edge_index, torch.Tensor):
#                 if self._cached_edge_index is None:
#                     # Ensure edge_weight is a tensor for ONNX export
#                     if edge_weight is None:
#                         edge_weight = torch.ones((edge_index.size(1),), dtype=torch.float32, device=edge_index.device)
#                     # Apply gcn_norm
#                     edge_index, edge_weight = gcn_norm(
#                         edge_index, edge_weight, num_nodes, self.improved, self.add_self_loops
#                     )
#                     if self.cached:
#                         self._cached_edge_index = (edge_index, edge_weight)
#                 else:
#                     edge_index, edge_weight = self._cached_edge_index
#             elif isinstance(edge_index, SparseTensor):
#                 if self._cached_adj_t is None:
#                     # Apply gcn_norm for SparseTensor
#                     edge_index = gcn_norm(
#                         edge_index, edge_weight, num_nodes, self.improved, self.add_self_loops
#                     )
#                     if self.cached:
#                         self._cached_adj_t = edge_index
#                 else:
#                     edge_index = self._cached_adj_t

#         return edge_index, edge_weight    

class GCNNorm(torch.nn.Module):
    def __init__(self, improved=False, cached=False, add_self_loops=True, normalize=True):
        super(GCNNorm, self).__init__()
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, edge_index, edge_weight=None):
        # Compute num_nodes from edge_index
        num_nodes = torch.max(edge_index) + 1

        # Part 1: Normalize (gcn_norm)
        if self.normalize:
            if isinstance(edge_index, torch.Tensor):
                if self._cached_edge_index is None:
                    # Ensure edge_weight is a tensor
                    if edge_weight is None:
                        edge_weight = torch.ones(
                            (edge_index.size(1),), dtype=torch.float32, device=edge_index.device
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

        return edge_index, edge_weight


class MLPUpdate(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLPUpdate, self).__init__()
        self.lin = Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        return self.lin(x)
    
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


# Step 2: Define AfterMessageGeneration model
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


#------------------------------
# Inference Test
#------------------------------
# Load the Flickr dataset
dataset = Flickr(root='data/Flickr')
data = dataset[0]

# Ensure the edge index is undirected
data.edge_index = to_undirected(data.edge_index)
# Prepare the Flickr data
# x = data.x  # Node features
# edge_index = data.edge_index  # Edge index
edge_weight = None
# Step 1: Apply GCN normalization
gcn_norm_model = GCNNorm(cached=False)
with torch.no_grad():
    for param in gcn_norm_model.parameters():
        param.fill_(1)
norm_edge_index, norm_edge_weight = gcn_norm_model(data.edge_index, edge_weight)

print(f"Normalized Edge Index: {norm_edge_index.size()}")
print(f"Normalized Edge Weight: {norm_edge_weight.size()}")

# Step 2: Apply linear transformation
linear_model = MLPUpdate(in_channels=dataset.num_node_features, out_channels=16)
with torch.no_grad():
    for param in linear_model.parameters():
        param.fill_(1)
x_linear = linear_model(data.x)
print(f"Linear operation: Input size: {data.x.size()}")
print(f"Linear operation: Output size: {x_linear.size()}")

# Model 3: MessageGeneration
before_msg_model = MessageGeneration()
x_pre_message, norm_edge_index, messages = before_msg_model(x_linear, data.edge_index)
print(f"Message Generation operation: x_pre_message: {x_pre_message.size()}")
print(f"Message Generation operation: norm_edge_index: {norm_edge_index.size()}")
print(f"Message Generation operation: msg_kwargs: {messages}")

# Model 4: MessageAggregation
after_msg_model = MessageAggregation()
output = after_msg_model(x_pre_message, norm_edge_index, messages)

print("Final output shape:", output.shape)


# #------------------------------
# # Export Test
# #------------------------------
# # Static export (fixed input sizes)
# # Static export for GCNNormModel
# # Example of how to export to ONNX, removing 'edge_weight' from the input names
# # Export the model to ONNX

# # Export the model to ONNX
# torch.onnx.export(
#     gcn_norm_model,
#     (data.edge_index, edge_weight),  # Only 2 inputs now
#     "Onnx/gcn_norm_static.onnx",
#     input_names=['edge_index'],  # Match the 2 inputs
#     output_names=['norm_edge_index', 'norm_edge_weight'],
#     opset_version=11
# )

# print("GCN norm static model exported successfully!")
# # torch.onnx.export(
# #     gcn_norm_model,
# #     (edge_index, edge_weight, data.num_nodes),
# #     "gcn_norm_static.onnx",
# #     input_names=['edge_index', 'edge_weight', 'num_nodes'],
# #     output_names=['norm_edge_index', 'norm_edge_weight'],
# #     opset_version=11
# # )
# # print("gcn_norm_model static model exported successfully!")

# # Static export for LinearModel
# # model_onnx_path = "Onnx/linear_static.onnx"
# # if not osp.exists(model_onnx_path):
# torch.onnx.export(linear_model,
#                   data.x,
#                   "Onnx/linear_static.onnx",
#                   input_names=['x'],
#                   output_names=['x_linear'],
#                   opset_version=11)
# print("linear_model static model exported successfully!")

# torch.onnx.export(before_msg_model, (data.x, data.edge_index),
#                   "Onnx/MessageGeneration_static.onnx",
#                   input_names=['x', 'edge_index'],
#                   output_names=['x_pre_message', 'edge_index', 'messages'],
#                   opset_version=11)
# print("before_message_static static models exported successfully!")

# torch.onnx.export(after_msg_model, (x_pre_message, norm_edge_index, messages),
#                   "Onnx/MessageAggregation_static.onnx",
#                   input_names=['x_pre_message', 'edge_index', 'messages'],
#                   output_names=['output'],
#                   opset_version=11)

# print("after_message_static static models exported successfully!")

# # Dynamic export (dynamic input sizes)

# torch.onnx.export(
#     gcn_norm_model,
#     (data.edge_index,),  # Only 1 input here
#     "Onnx/gcn_norm_dynamic.onnx",
#     input_names=['edge_index'],
#     output_names=['norm_edge_index', 'norm_edge_weight'],
#     opset_version=11,
#     dynamic_axes= {
#         'edge_index': {1: 'num_edges'},  # Set dynamic axis for edge_index
#         'norm_edge_index': {1: 'num_edges'},  # Dynamic for output
#         'norm_edge_weight': {0: 'num_edges'}  # Dynamic for output
#     }
# )
# print("GCN norm dynamic model exported successfully!")

# # torch.onnx.export(gcn_norm_model,
# #                   (data.edge_index, None, data.x.size(0)),
# #                   "gcn_norm_dynamic.onnx",
# #                   input_names=['edge_index', 'num_nodes'],
# #                   output_names=['norm_edge_index', 'norm_edge_weight'],
# #                   opset_version=11,
# #                   dynamic_axes={
# #                     'edge_index': {1: 'num_edges'},
# #                     'norm_edge_index': {1: 'num_edges'},
# #                     'norm_edge_weight': {0: 'num_edges'}
# #                   }
# # )
# print("gcn_norm_model dynamic model exported successfully!")

# # Dynamic export for LinearModel
# torch.onnx.export(linear_model,
#                   data.x,
#                   "Onnx/linear_dynamic.onnx",
#                   input_names=['x'],
#                   output_names=['x_linear'],
#                   opset_version=11,
#                   dynamic_axes={
#                     'x': {0: 'num_nodes', 1: 'features'},
#                     'x_linear': {0: 'num_nodes', 1: 'output_features'}
#                   }
# )
# print("linear_model dynamic model exported successfully!")


# torch.onnx.export(before_msg_model, 
#                   (data.x, data.edge_index), 
#                   "Onnx/MessageGeneration_dynamic.onnx",
#                   input_names=['x', 'edge_index'],            # match with the model's input names
#                   output_names=['x_pre_message', 'edge_index', 'messages'],  # match with the model's output names
#                   opset_version=11,
#                   dynamic_axes={
#                     'x': {0: 'num_nodes', 1: 'features'},      # dynamic batch size and feature dimension for 'x'
#                     'edge_index': {1: 'num_edges'},            # dynamic number of edges for 'edge_index'
#                     'x_pre_message': {0: 'num_nodes', 1: 'features'},  # dynamic batch size and feature dimension for 'x_pre_message'
#                     'messages': {0: 'num_edges'}               # dynamic number of edges for 'messages'
#                   }
# )


# print("before_message_dynamic models exported successfully!")
# torch.onnx.export(after_msg_model, (x_pre_message, norm_edge_index, messages),
#                   "Onnx/MessageAggregation_dynamic.onnx",
#                   input_names=['x_pre_message', 'norm_edge_index', 'messages'],
#                   output_names=['output'],
#                   dynamic_axes={
#                       'x_pre_message': {0: 'batch_size'},
#                       'norm_edge_index': {1: 'num_edges'},
#                       'messages': {0: 'num_edges'},
#                       'output': {0: 'batch_size'}
#                   },
#                   opset_version=13)

# print("after_message_dynamic models exported successfully!")

class CombinedModel(torch.nn.Module):
    def __init__(self, selected_models):
        super(CombinedModel, self).__init__() 
        self.models = torch.nn.ModuleList(selected_models)

    def forward(self, *inputs):
        x = inputs[0]  # Initial input
        edge_index = inputs[1]  # Edge index (shared for all GCN-like layers)
        edge_weight = inputs[2] if len(inputs) > 2 else None  # Optional edge weight
        for model in self.models:
            if isinstance(model, GCNNorm) or isinstance(model, MessageGeneration):
                x, edge_index, edge_weight = model(x, edge_index)
            else:
                x = model(x)

        return x
  
# Define individual models
gcn_norm_model = GCNNorm(cached=False)
linear_model = MLPUpdate(in_channels=dataset.num_node_features, out_channels=16)
before_msg_model = MessageGeneration()
after_msg_model = MessageAggregation()

# Combine models into one sequence
combined_model = CombinedModel([gcn_norm_model, linear_model])

# Test forward pass with combined models
x = data.x
edge_index = data.edge_index
edge_weight = None
output = combined_model(x, edge_index, edge_weight)

print("Combined Model output shape:", output.shape)
