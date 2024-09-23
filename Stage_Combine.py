import torch
import torch.nn as nn
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
        # Compute num_nodes from edge_index
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


class CombinedModel(nn.Module):
    def __init__(self, selected_models):
        super(CombinedModel, self).__init__()
        self.models = nn.ModuleList(selected_models)

    def forward(self, x, edge_index, edge_weight=None):
        # Ensure edge_index is always int64
        edge_index = edge_index.to(torch.int64)  # Add this line to ensure correct type

        for model in self.models:
            if isinstance(model, GCNNorm):
                x, edge_index, edge_weight = model(x, edge_index, edge_weight)
            elif isinstance(model, MessageGeneration):
                x, edge_index, messages = model(x, edge_index)
            elif isinstance(model, MessageAggregation):
                x = model(x, edge_index, messages)
            else:
                x = model(x)
        return x

# Load the Flickr dataset
dataset = Flickr(root='data/Flickr')
data = dataset[0]
  
# Define individual models
gcn_norm_model = GCNNorm(cached=False)
linear_model = MLPUpdate(in_channels=dataset.num_node_features, out_channels=16)
before_msg_model = MessageGeneration()
after_msg_model = MessageAggregation()

model_sequence = [2, 3]
# Map the numbers to actual models
model_map = {
    1: gcn_norm_model,
    2: linear_model,
    3: before_msg_model,
    4: after_msg_model
}
selected_models_list = []
for model_number in model_sequence:
    selected_models_list.append(model_map[model_number])
print(selected_models_list)

# Combine models into one sequence
combined_model = CombinedModel(selected_models_list)

# Test forward pass with combined models
x = data.x
edge_index = data.edge_index
edge_index = edge_index.to(torch.int64)
edge_weight = None
output = combined_model(x, edge_index, edge_weight)

print("Combined Model output shape:", output.shape)

# Export ONNX based on the first model's input
if isinstance(selected_models_list[0], GCNNorm):
    # Input for GCNNorm
    dummy_x = None  # No node features needed here
    dummy_edge_index = data.edge_index
    dummy_edge_weight = torch.ones(data.edge_index.size(1))  # Example edge weight
    input_names = ['edge_index', 'edge_weight']
    dummy_inputs = (dummy_edge_index, dummy_edge_weight)

elif isinstance(selected_models_list[0], MLPUpdate):
    # Input for MLPUpdate (linear layer)
    dummy_x = data.x
    dummy_edge_index = None
    dummy_edge_weight = None
    input_names = ['x']
    dummy_inputs = (dummy_x,)

elif isinstance(selected_models_list[0], MessageGeneration):
    # Input for MessageGeneration (requires x and edge_index)
    dummy_x = data.x
    dummy_edge_index = data.edge_index
    dummy_edge_weight = None
    input_names = ['x', 'edge_index']
    dummy_inputs = (dummy_x, dummy_edge_index)

elif isinstance(selected_models_list[0], MessageAggregation):
    # Input for MessageAggregation (requires x, edge_index, and messages)
    dummy_x = data.x
    dummy_edge_index = data.edge_index
    dummy_edge_weight = torch.ones(data.edge_index.size(1))  # Example edge weight
    messages = torch.randn(dummy_edge_index.size(1), dummy_x.size(1))  # Generated messages
    input_names = ['x', 'edge_index', 'messages']
    dummy_inputs = (dummy_x, dummy_edge_index, messages)

# Export the combined model to ONNX
torch.onnx.export(
    combined_model,  # The combined model
    dummy_inputs,  # Inputs to the model
    "combined_model.onnx",  # Output ONNX file name
    input_names=input_names,  # Input names
    output_names=['output'],  # Output name
    opset_version=11,  # ONNX opset version
    dynamic_axes={  # Define dynamic axes for flexibility
        'x': {0: 'num_nodes', 1: 'features'} if dummy_x is not None else {},
        'edge_index': {1: 'num_edges'} if dummy_edge_index is not None else {},
        'output': {0: 'num_nodes', 1: 'output_features'}
    }
)

print("Combined model exported to combined_model.onnx successfully!")
