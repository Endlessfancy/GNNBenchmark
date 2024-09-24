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

import openvino as ov

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
        # print(f"Message Generation operation: messages: {messages.shape}")
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
# define model combination class
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


# ----------------------------
# Test Inference with openvino api
# ----------------------------
# ----------------------------
# # Test all combination

# Test forward pass with OpenVINO inference for combined models
def test_combined_model_ir_inference(model_ir_path, selected_model_list, data, device="CPU"):
    # define the first layer and last layer:
    first_layer = selected_model_list[0]

    # Create OpenVINO Core
    core = ov.Core()

    # Load the Combined Model in IR format
    compiled_model = core.compile_model(model=model_ir_path, device_name=device)
    print(f"Model {model_ir_path} loaded to {device}!")

    # Define input based on the first layer
    dummy_inputs = None

    # Set dummy input based on the first layer
    if isinstance(first_layer, GCNNorm):
        input_data = data.x.numpy().astype(np.float32)
        edge_index = data.edge_index.numpy().astype(np.int64)
        dummy_edge_weight = np.ones(data.edge_index.size(1), dtype=np.float32)  # Example edge weight
        print("Testing combined model with GCNNorm as the first layer")

        # Create OpenVINO tensors
        input_data_tensor = ov.Tensor(array=input_data)
        edge_index_tensor = ov.Tensor(array=edge_index)
        dummy_edge_weight_tensor = ov.Tensor(array=dummy_edge_weight)

        # Create inference request and set input tensors
        infer_request = compiled_model.create_infer_request()
        infer_request.set_tensor(compiled_model.input(0), input_data_tensor)  # edge_index
        infer_request.set_tensor(compiled_model.input(1), edge_index_tensor)  # edge_weight
        # infer_request.set_tensor(compiled_model.input(2), dummy_edge_weight_tensor)  # edge_weight

    elif isinstance(first_layer, MLPUpdate):
        input_data = data.x.numpy().astype(np.float32)
        edge_index = data.edge_index.numpy()  # Edge index
        print("Testing combined model with MLPUpdate as the first layer")

        # Create OpenVINO tensor
        input_tensor = ov.Tensor(array=input_data)
        edge_index_tensor = ov.Tensor(array=edge_index)

        # Create inference request and set input tensors
        infer_request = compiled_model.create_infer_request()
        infer_request.set_tensor(compiled_model.input(0), input_tensor)  # x
        infer_request.set_tensor(compiled_model.input(1), edge_index_tensor)  # edge_we

    elif isinstance(first_layer, MessageGeneration):
        input_data = data.x.numpy().astype(np.float32)
        edge_index = data.edge_index.numpy().astype(np.int64)
        print("Testing combined model with MessageGeneration as the first layer")

        # Create OpenVINO tensors
        input_tensor = ov.Tensor(array=input_data)
        edge_index_tensor = ov.Tensor(array=edge_index)

        # Create inference request and set input tensors
        infer_request = compiled_model.create_infer_request()
        infer_request.set_tensor(compiled_model.input(0), input_tensor)  # x
        infer_request.set_tensor(compiled_model.input(1), edge_index_tensor)  # edge_index

    elif isinstance(first_layer, MessageAggregation):
        input_data = data.x.numpy().astype(np.float32)
        edge_index = data.edge_index.numpy().astype(np.int64)
        dummy_messages = np.random.randn(edge_index.shape[1], input_data.shape[1]).astype(np.float32)  # Example messages
        print("Testing combined model with MessageAggregation as the first layer")

        # Create OpenVINO tensors
        input_tensor = ov.Tensor(array=input_data)
        edge_index_tensor = ov.Tensor(array=edge_index)
        dummy_messages_tensor = ov.Tensor(array=dummy_messages)

        # Create inference request and set input tensors
        infer_request = compiled_model.create_infer_request()
        infer_request.set_tensor(compiled_model.input(0), input_tensor)  # x
        infer_request.set_tensor(compiled_model.input(1), edge_index_tensor)  # edge_index
        infer_request.set_tensor(compiled_model.input(2), dummy_messages_tensor)  # messages

    # Perform inference
    infer_request.start_async()
    infer_request.wait()

    # Get and print the result
    output_tensor = infer_request.get_output_tensor(0).data
    output = output_tensor.data
    print(f"Output of combined model {selected_model_list}: {output.shape}\n")

device = "CPU"  # Or "CPU"

for i in range(1, 5):
    for j in range(i, 5):
        selected_model_list = []
        model_sequence = list(range(i, j + 1))
        print(f"Model sequence: {model_sequence}")
        
        # Map the numbers to actual models
        model_map = {
            1: gcn_norm_model,
            2: linear_model,
            3: before_msg_model,
            4: after_msg_model
        }
        model_ir_path = "Ir/Combined_model_Ir/Combined_model_"
    
        for model_number in model_sequence:
            selected_model_list.append(model_map[model_number])
            model_ir_path += str(model_number)
        model_ir_path += ".xml"

        # Run the test for the selected model sequence
        test_combined_model_ir_inference(model_ir_path, selected_model_list, data, device)


# device = "CPU"
# for i in range(1, 5):
#     for j in range(i, 5):
#         selected_model_list = []
#         model_sequence = list(range(i, j + 1))
#         print(f"Model sequence: {model_sequence}")
        
#         # Map the numbers to actual models
#         model_map = {
#             1: gcn_norm_model,
#             2: linear_model,
#             3: before_msg_model,
#             4: after_msg_model
#         }
#         model_ir_path = "Ir/Combined_model_Ir/Combined_model_"
    
#         for model_number in model_sequence:
#             selected_model_list.append(model_map[model_number])
#             model_ir_path += str(model_number)
#         model_ir_path += ".xml"

#         # Create OpenVINO Core
#         core = ov.Core()

#         # Load the Combined Model
#         compiled_model = core.compile_model(model=model_ir_path, device_name=device)
#         print(f"Model {model_ir_path} loaded to {device}!")

#         # Prepare input data based on the first model in the sequence
#         first_layer = selected_model_list[0]

#         if isinstance(first_layer, GCNNorm):
#             input_data = data.x.numpy()  # Node features
#             edge_index = data.edge_index.numpy()  # Edge index
#             dummy_edge_weight = np.ones(data.edge_index.size(1), dtype=np.float32)  # Example edge weight in float32


#             # Create OpenVINO tensors
#             input_data_tensor = ov.Tensor(array=input_data)
#             edge_index_tensor = ov.Tensor(array=edge_index)
#             dummy_edge_weight_tensor = ov.Tensor(array=dummy_edge_weight)

#             # Create inference request and set input tensors
#             infer_request = compiled_model.create_infer_request()
#             infer_request.set_tensor(compiled_model.input(0), input_data_tensor)  # edge_index
#             infer_request.set_tensor(compiled_model.input(1), edge_index_tensor)  # edge_weight
#             infer_request.set_tensor(compiled_model.input(2), dummy_edge_weight_tensor)  # edge_weight

#         elif isinstance(first_layer, MLPUpdate):
#             input_data = data.x.numpy()  # Node features
#             edge_index = data.edge_index.numpy()  # Edge index

#             # Create OpenVINO tensor
#             input_tensor = ov.Tensor(array=input_data)
#             edge_index_tensor = ov.Tensor(array=edge_index)

#             # Create inference request and set input tensors
#             infer_request = compiled_model.create_infer_request()
#             infer_request.set_tensor(compiled_model.input(0), input_tensor)  # x
#             infer_request.set_tensor(compiled_model.input(1), edge_index_tensor)  # edge_weight

#         elif isinstance(first_layer, MessageGeneration):
#             input_data = data.x.numpy()  # Node features
#             edge_index = data.edge_index.numpy()  # Edge index

#             # Create OpenVINO tensors
#             input_tensor = ov.Tensor(array=input_data)
#             edge_index_tensor = ov.Tensor(array=edge_index)

#             # Create inference request and set input tensors
#             infer_request = compiled_model.create_infer_request()
#             infer_request.set_tensor(compiled_model.input(0), input_tensor)  # x
#             infer_request.set_tensor(compiled_model.input(1), edge_index_tensor)  # edge_index

#         elif isinstance(first_layer, MessageAggregation):
#             input_data = data.x.numpy()  # Node features
#             edge_index = data.edge_index.numpy()  # Edge index
#             dummy_messages = np.random.randn(edge_index.shape[1], input_data.shape[1])  # Example messages

#             # Create OpenVINO tensors
#             input_tensor = ov.Tensor(array=input_data)
#             edge_index_tensor = ov.Tensor(array=edge_index)
#             dummy_messages_tensor = ov.Tensor(array=dummy_messages)

#             # Create inference request and set input tensors
#             infer_request = compiled_model.create_infer_request()
#             infer_request.set_tensor(compiled_model.input(0), input_tensor)  # x
#             infer_request.set_tensor(compiled_model.input(1), edge_index_tensor)  # edge_index
#             infer_request.set_tensor(compiled_model.input(2), dummy_messages_tensor)  # messages

#         # Perform inference
#         infer_request.start_async()
#         infer_request.wait()

#         # Get and print the result
#         output_tensor = infer_request.get_output_tensor(0).data
#         output = output_tensor.data
#         print(f"Output of combined model {model_sequence}: {output.shape}\n")

# # ----------------------------
# # Test a specific model
# device = "GPU"
# model_ir_path = "Ir/Combined_model_Ir/Combined_model_12.xml"
# core = ov.Core()

# # Load the Combined Model with 1 (GCNNorm) and 2 (MLPUpdate)
# compiled_model = core.compile_model(model=model_ir_path, device_name=device)
# print(f"Model {model_ir_path} loaded to {device}!")

# # Prepare input data
# input_data = data.x.numpy()  # Node features
# edge_index = data.edge_index.numpy()  # Edge index
# dummy_edge_weight = np.ones(data.edge_index.size(1))  # Example edge weight

# # Create OpenVINO tensors
# input_tensor = ov.Tensor(array=input_data)
# edge_index_tensor = ov.Tensor(array=edge_index)
# dummy_edge_weight_tensor = ov.Tensor(array=dummy_edge_weight)

# # Create inference request and set input tensors
# infer_request = compiled_model.create_infer_request()
# infer_request.set_tensor(compiled_model.input(0), input_tensor)  # x
# infer_request.set_tensor(compiled_model.input(1), edge_index_tensor)  # edge_index
# # infer_request_12.set_tensor(compiled_model_12.input(2), dummy_edge_weight_tensor)  # edge_weight

# # Perform inference
# infer_request.start_async()
# infer_request.wait()

# # Get and print the result
# output_tensor = infer_request.get_output_tensor(0).data
# # res = infer_request.get_output_tensor(0).data
# output = output_tensor.data
# print(f"Output of combined model: {output.shape}\n")
