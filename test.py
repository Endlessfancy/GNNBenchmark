import os.path as osp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Flickr
# from torch_geometric.utils import to_dense_adj
# from torch_geometric.utils import scatter_add
import time
import openvino.runtime as ov
from openvino.tools.mo import convert_model

# Define the GCNLayer without converting edge_index to adj_matrix
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0, use_layernorm=False):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.ones(in_features, out_features))
        self.bias = nn.Parameter(torch.ones(out_features))
        self.dropout = dropout
        self.use_layernorm = use_layernorm

        if self.use_layernorm:
            self.layernorm = nn.LayerNorm(out_features)

    def forward(self, x, edge_index):
        row, col = edge_index

        if self.training and self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Message Passing: Aggregate neighbor features
        out = torch.matmul(x, self.weight)
        out = torch.zeros_like(out).scatter_add_(0, row.unsqueeze(-1).expand(-1, out.size(1)), out[col]) + self.bias

        if self.use_layernorm:
            out = self.layernorm(out)

        return out
# Define the GCN model
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.0, use_layernorm=False):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNLayer(in_channels, hidden_channels, dropout, use_layernorm))
        for _ in range(num_layers - 2):
            self.convs.append(GCNLayer(hidden_channels, hidden_channels, dropout, use_layernorm))
        self.convs.append(GCNLayer(hidden_channels, out_channels, dropout, use_layernorm))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(x, dim=1)

# # Define the GCNLayer
# class GCNLayer(nn.Module):
#     def __init__(self, in_features, out_features, dropout=0.0, use_layernorm=False):
#         super(GCNLayer, self).__init__()
#         self.weight = nn.Parameter(torch.ones(in_features, out_features))
#         self.bias = nn.Parameter(torch.ones(out_features))
#         self.dropout = dropout
#         self.use_layernorm = use_layernorm

#         if self.use_layernorm:
#             self.layernorm = nn.LayerNorm(out_features)

#     def forward(self, x, adj_matrix):
#         adj_matrix = adj_matrix.float()

#         if self.training and self.dropout > 0:
#             x = F.dropout(x, p=self.dropout, training=self.training)

#         out = torch.matmul(adj_matrix, x)
#         out = torch.matmul(out, self.weight) + self.bias

#         if self.use_layernorm:
#             out = self.layernorm(out)

#         return out

# # Define the GCN model
# class GCN(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.0, use_layernorm=False):
#         super(GCN, self).__init__()
#         self.num_layers = num_layers
#         self.convs = nn.ModuleList()
#         self.convs.append(GCNLayer(in_channels, hidden_channels, dropout, use_layernorm))
#         for _ in range(num_layers - 2):
#             self.convs.append(GCNLayer(hidden_channels, hidden_channels, dropout, use_layernorm))
#         self.convs.append(GCNLayer(hidden_channels, out_channels, dropout, use_layernorm))

#     def forward(self, x, edge_index):
#         for i, conv in enumerate(self.convs):
#             x = conv(x, edge_index)
#             if i != self.num_layers - 1:
#                 x = F.relu(x)
#                 x = F.dropout(x, p=0.5, training=self.training)
#         return F.log_softmax(x, dim=1)

#     # def forward(self, x, edge_index):
#     #     num_nodes = x.size(0)
#     #     adj_matrix = self.edge_index_to_adj(edge_index, num_nodes)
#     #     adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)

#     #     for i, conv in enumerate(self.convs):
#     #         x = conv(x, adj_matrix)
#     #         if i != self.num_layers - 1:
#     #             x = F.relu(x)
#     #             x = F.dropout(x, p=0.5, training=self.training)
#     #     return F.log_softmax(x, dim=1)

#     # def edge_index_to_adj(self, edge_index, num_nodes):
#     #     row, col = edge_index
#     #     adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
#     #     indices = torch.stack([row, col], dim=0)
#     #     adj_matrix.index_put_(indices, torch.ones_like(row, dtype=torch.float32), accumulate=True)
#     #     return adj_matrix

# # Define the GCN model
# class GCN(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.0, use_layernorm=False):
#         super(GCN, self).__init__()
#         self.num_layers = num_layers
#         self.convs = nn.ModuleList()
#         self.convs.append(GCNLayer(in_channels, hidden_channels, dropout, use_layernorm))
#         for _ in range(num_layers - 2):
#             self.convs.append(GCNLayer(hidden_channels, hidden_channels, dropout, use_layernorm))
#         self.convs.append(GCNLayer(hidden_channels, out_channels, dropout, use_layernorm))

#     def forward(self, x, edge_index):
#         num_nodes = x.size(0)
#         adj_matrix = self.edge_index_to_adj(edge_index, num_nodes)
#         adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)

#         for i, conv in enumerate(self.convs):
#             x = conv(x, adj_matrix)
#             if i != self.num_layers - 1:
#                 x = F.relu(x)
#                 x = F.dropout(x, p=0.5, training=self.training)
#         return F.log_softmax(x, dim=1)

#     def edge_index_to_adj(self, edge_index, num_nodes):
#         adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
#         for src, dest in edge_index.t():
#             adj_matrix[src, dest] = 1
#         return adj_matrix

# Load the Flickr dataset
def load_data():
    dataset = Flickr(root='data/Flickr')
    return dataset

# Set all parameters to 1 and print output
def main():
    # Load data
    dataset = load_data()
    data = dataset[0]

    # Define model
    model = GCN(dataset.num_features, 64, dataset.num_classes, num_layers=2, dropout=0.5, use_layernorm=True)

    # Set all parameters to 1 (already done in the initialization)
    for name, param in model.named_parameters():
        param.data.fill_(1.0)
        # print(f'Parameter {name} set to: \n{param}\n')

    # Run the model on the data
    model.eval()  # Set model to evaluation mode
    out = model(data.x, data.edge_index)
    print(f'Output of the model: \n{out}')

    # Compile the model for NPU
    model_ir_path = "Ir/" +"test_opt14.xml"
    model_onnx_path = "Onnx/" +"test_opt14.onnx"
    # export new format:

    if not osp.exists(model_onnx_path):
        print(f"Sampled ONNX model exported to {model_onnx_path}.")
        dummy_input = (data.x, data.edge_index)

        torch.onnx.export(
            model,
            dummy_input,
            model_onnx_path,
            # input_names=['node_features', 'edge_index'],
            # output_names=['output'],
            # dynamic_axes=None,
            # dynamic_axes={'node_features': {0: 'num_nodes'},  # 将 num_nodes 设为动态
            #     'edge_index': {1: 'num_edges'}  # 将 num_edges 设为动态
            #     },
            opset_version=13  # 根据需要调整 opset 版本
            )

    else:
        print(f"Sampled ONNX smodel {model_onnx_path} already exists.")
    if not osp.exists(model_ir_path):
        print("Exporting ONNX model to IR... This may take a few minutes.")
        # os.system('mo --input_model Onnx/Sampled_Flickr_SAGE.onnx --output_dir Ir')
    else:
        print(f"IR model {model_ir_path} already exists.")
        core = ov.Core()
        # Load the model
        # model_ir_path = "Ir/" +DATASET +"_" +MODEL +".xml"
        model = core.read_model(model_ir_path)
        # model.reshape([-1, -1, -1])
        compiled_model = core.compile_model(model=model_ir_path, device_name="NPU")
        print("Covert to NPU!")
        input_data = data.x.numpy()
        edge_index = data.edge_index.numpy()
        _ = compiled_model([input_data, edge_index])
        # infer_request = compiled_model.create_infer_request()
        # input_data = data.x.numpy()
        # edge_index = data.edge_index.numpy()
        # #    _ = compiled_model([input_data, edge_index])

        # # Assuming input_data and edge_index are numpy arrays
        # input_tensor = ov.Tensor(array=input_data)

        # # If you need to pass multiple tensors, you would do something like:
        # edge_index_tensor = ov.Tensor(array=edge_index)

        # # Then, use these tensors in your infer_request
        # infer_request.set_tensor(compiled_model.input(0), input_tensor)
        # infer_request.set_tensor(compiled_model.input(1), edge_index_tensor)

        # timings = []
        # start_time = time.time()
        # infer_request.start_async()
        # infer_request.wait()
        # res = infer_request.get_output_tensor(0).data
        # end_time = time.time()
        # timings.append(end_time - start_time)
        # print(res)
        # print(f'Average Inference Time: {sum(timings)} ms')

if __name__ == "__main__":
    main()
