import torch
# import intel_extension_for_pytorch as ipex
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from typing import Optional, Tuple, Union, overload

from torch_geometric.datasets import Flickr
from torch_geometric.nn import GATConv, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils.sparse import set_sparse_value
import time
from torch_geometric.nn import GATConv



class GAT_Transform(GATConv):
    def forward(self, x):
        for i in range(10):
            # Linear transformation
            x_lin = self.lin(x)
            x_lin_reshape = x_lin.view(-1, self.heads, self.out_channels)
        # print("Shape of x after linear transformation:", x_lin.shape)
        # print("Shape of x after reshaping:", x_lin_reshape.shape)
        return x_lin_reshape

class GAT_Attention(GATConv):
    def forward(self, x):
        for i in range(10):
            # Compute attention coefficients
            alpha_src = (x * self.att_src).sum(dim=-1)
            alpha_dst = (x * self.att_dst).sum(dim=-1)
            alpha = alpha_src + alpha_dst

        return alpha

class gat_linear(GATConv):
    def forward(self, x):
        # print("Shape of x:", x.shape)
        # print("Shape of self.att_src:", self.att_src.shape)
        for i in range(10):
            x_lin = self.lin(x)
            x_lin_reshape = x_lin.view(-1, self.heads, self.out_channels)
            # Compute attention coefficients
            alpha_src = (x_lin_reshape * self.att_src).sum(dim=-1)
            alpha_dst = (x_lin_reshape * self.att_dst).sum(dim=-1)
            alpha = alpha_src + alpha_dst
        # print("Shape of alpha_src:", alpha_src.shape)
        # print("Shape of alpha_dst:", alpha_dst.shape)
        # print("Shape of alpha after combining source and target:", alpha.shape)

        return alpha

class GAT_SelfLoop(GATConv):
    def forward(self, num_nodes, edge_index):
        for i in range(10):
            # num_nodes = x.size(0)
            edge_index_out, edge_attr  = remove_self_loops(edge_index)
            edge_index_out, edge_attr  = add_self_loops(edge_index_out, num_nodes=num_nodes)
        # print("Shape of edge_index after adding self-loops:", edge_index_out.shape)
        return edge_index_out

class GAT_EdgeWeightUpdate(GATConv):
    def forward(self, alpha, edge_index): 
        for i in range(10):
            alpha_out = self.edge_updater(edge_index, alpha=alpha, edge_attr=None)
        # print("Shape of out after edge updata:", alpha_out.shape)
        return alpha_out
    # def edge_update(self, alpha: Tensor, 
    #                 edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
    #                 dim_size: Optional[int]) -> Tensor:
    #     alpha = F.leaky_relu(alpha, self.negative_slope)
    #     # alpha = softmax(alpha, index, ptr, dim_size)
    #     # alpha = softmax(alpha, index)
    #     return alpha
    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                dim_size: Optional[int]) -> Tensor:
        # Print initial input sizes
        # print("Initial alpha_j size:", alpha_j.size())
        # if alpha_i is not None:
            # print("Initial alpha_i size:", alpha_i.size())
        # else:
            # print("Initial alpha_i: None")
        
        # if edge_attr is not None:
            # print("Initial edge_attr size:", edge_attr.size())
        # else:
            # print("Initial edge_attr: None")
        
        # print("Initial index size:", index.size())
        # if ptr is not None:
            # print("Initial ptr size:", ptr.size())
        # else:
            # print("Initial ptr: None")
        # print("Initial dim_size:", dim_size)

        # Sum up source and target attention coefficients
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        # print("Summed alpha size (alpha_j + alpha_i):", alpha.size())

        # Skip if index is empty
        if index.numel() == 0:
            # print("Index is empty, returning alpha size:", alpha.size())
            return alpha

        # Apply linear transformation to edge attributes if they exist
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            # print("Edge_attr size after reshape (if necessary):", edge_attr.size())
            
            edge_attr = self.lin_edge(edge_attr)
            # print("Edge_attr size after lin_edge transformation:", edge_attr.size())

            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            # print("Edge_attr reshaped size for multi-head attention:", edge_attr.size())

            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            # print("Alpha_edge size (edge_attr weighted by att_edge):", alpha_edge.size())
            
            alpha = alpha + alpha_edge
            # print("Alpha size after adding edge attention:", alpha.size())

        # Apply Leaky ReLU
        alpha = F.leaky_relu(alpha, self.negative_slope)
        # print("Alpha size after Leaky ReLU:", alpha.size())

        # Apply softmax
        alpha = softmax(alpha, index, ptr, dim_size)
        # print("Alpha size after softmax:", alpha.size())

        # Apply dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        # print("Alpha size after dropout:", alpha.size())
        
        return alpha

class gat_propagate(GATConv):
    def forward(self, x, edge_index, alpha):
        for i in range(10):
            # Start propagating messages
            out = self.propagate(edge_index, x=x, alpha=alpha)
        # print("Shape of out after Message Passing:", out.shape)
        return out

    
class GAT_MessagePassing(GATConv):
    def forward(self, x, edge_index, alpha):
        for i in range(10):
            # Start propagating messages
            out = self.propagate(edge_index, x=x, alpha=alpha)
        # print("Shape of out after Message Passing:", out.shape)
        return out

class GAT_output(GATConv):
    def forward(self, out):
        for i in range(10):
           
            out_concat = out.view(-1, self.heads * self.out_channels)
        
            # Add bias if applicable
            if self.bias is not None:
                out_bias = out_concat + self.bias
        # print("Shape of out after concatenation/averaging:", out_concat.shape)
        # print("Shape of out after adding bias:", out_bias.shape)
        return out_bias


class gat_full(GATConv):
    def forward(self, x, edge_index):
        for i in range(10):
            x_lin = self.lin(x)
            x_lin_reshape = x_lin.view(-1, self.heads, self.out_channels)
            # Compute attention coefficients
            alpha_src = (x_lin_reshape * self.att_src).sum(dim=-1)
            alpha_dst = (x_lin_reshape * self.att_dst).sum(dim=-1)
            alpha = alpha_src + alpha_dst
            out = self.propagate(edge_index, x=x, alpha=alpha)
        # print("Shape of out after Message Passing:", out.shape)
        return out

if __name__ == "__main__":
    # Load the Flickr dataset
    dataset = Flickr(root='data/Flickr')
    data = dataset[0]

    # Define the NeighborLoader
    subgraph_loader = NeighborLoader(
        data,
        num_neighbors=[-1],
        batch_size=4096,  # Adjust as needed
        input_nodes=None
    )

    # Instantiate models
    linear_model = GAT_Transform(in_channels=dataset.num_features, out_channels=128, heads=4)
    attention_model = GAT_Attention(in_channels=dataset.num_features, out_channels=128, heads=4)
    edge_index_model = GAT_SelfLoop(in_channels=dataset.num_features, out_channels=128, heads=4)
    edge_weight_model = GAT_EdgeWeightUpdate(in_channels=dataset.num_features, out_channels=128, heads=4)
    message_passing_model = GAT_MessagePassing(in_channels=dataset.num_features, out_channels=128, heads=4)
    output_model = GAT_output(in_channels=dataset.num_features, out_channels=128, heads=4)
    linear = gat_linear(in_channels=dataset.num_features, out_channels=128, heads=4)
    gat_full = gat_full(in_channels=dataset.num_features, out_channels=128, heads=4, add_self_loops=False)

    # Perform the forward passes and time each step
    with torch.no_grad():
        # output = gat_full(data.x, data.edge_index)
        # print(f"full Output size: {output.size()}")

        start_time = time.time()
        
        # Linear transformation
        print(f"Input size for Linear Transformation: {data.x.size()}")
        x = linear_model(data.x)
        print(f"Output size after Linear Transformation: {x.size()}")
        linear_time = time.time()
        
        x_linear = linear(data.x)
        print(f"Output size after Linear Transformation: {x_linear.size()}")

        # Compute attention coefficients
        print(f"Input size for Attention Coefficients: {x.size()}")
        alpha = attention_model(x)
        print(f"Output size after Attention Coefficients: {alpha.size()}")
        attention_time = time.time()
        
        # Add self-loops
        print(f"Input size for Self-Loop Addition (edge_index): {data.edge_index.size()}")
        num_nodes = data.x.size(0)
        edge_index = edge_index_model(num_nodes, data.edge_index)
        print(f"Output size after Self-Loop Addition: {edge_index.size()}")
        self_loop_time = time.time()
        edge_index = data.edge_index
        
        # Update edge weights
        print(f"Input size for Edge Weight Update (alpha): {alpha.size()}, (edge_index): {edge_index.size()}")
        alpha = edge_weight_model(alpha, edge_index)
        print(f"Output size after Edge Weight Update: {alpha.size()}")
        edge_weight_time = time.time()
        
        # Message passing
        print(f"Input sizes for Message Passing (x): {x.size()}, (edge_index): {edge_index.size()}, (alpha): {alpha.size()}")
        out = message_passing_model(x, edge_index, alpha)
        print(f"Output size after Message Passing: {out.size()}")
        message_passing_time = time.time()
        
        # Output aggregation
        print(f"Input size for Output Aggregation: {out.size()}")
        out = output_model(out)
        print(f"Output size after Output Aggregation: {out.size()}")
        output_time = time.time()
        # Total and individual step times
        # print(f"Total inference time: {(output_time - start_time) * 100:.4f} ms")
        # print(f"Linear Transformation Time: {(linear_time - start_time) * 100:.4f} ms")
        # print(f"Attention Coefficients Time: {(attention_time - linear_time) * 100:.4f} ms")
        # print(f"Self-Loop Addition Time: {(self_loop_time - attention_time) * 100:.4f} ms")
        # print(f"Edge Weight Update Time: {(edge_weight_time - self_loop_time) * 100:.4f} ms")
        # print(f"Message Passing Time: {(message_passing_time - edge_weight_time) * 100:.4f} ms")
        # print(f"Output Aggregation Time: {(output_time - message_passing_time) * 100:.4f} ms")


    # # Move models to XPU and optimize with IPEX
    # device = torch.device("xpu")
    # linear_model = linear_model.to("xpu")
    # attention_model = attention_model.to("xpu")
    # edge_index_model = edge_index_model.to("xpu")
    # edge_weight_model = edge_weight_model.to("xpu")
    # message_passing_model = message_passing_model.to("xpu")
    # output_model = output_model.to("xpu")

    # # Optimize models for inference
    # linear_model = ipex.optimize(linear_model, dtype=torch.float32, level='O1', inplace=False)
    # attention_model = ipex.optimize(attention_model, dtype=torch.float32, level='O1', inplace=False)
    # edge_index_model = ipex.optimize(edge_index_model, dtype=torch.float32, level='O1', inplace=False)
    # edge_weight_model = ipex.optimize(edge_weight_model, dtype=torch.float32, level='O1', inplace=False)
    # message_passing_model = ipex.optimize(message_passing_model, dtype=torch.float32, level='O1', inplace=False)
    # output_model = ipex.optimize(output_model, dtype=torch.float32, level='O1', inplace=False)

    # data = data.to(device)

    # # Perform the forward passes and time each step
    # with torch.no_grad():
    #     start_time = time.time()
        
    #     # Linear transformation
    #     x = linear_model(data.x)
    #     linear_time = time.time()
        
    #     # Compute attention coefficients
    #     alpha = attention_model(x)
    #     attention_time = time.time()
        
    #     # Add self-loops
    #     edge_index = edge_index_model(data.edge_index)
    #     self_loop_time = time.time()
        
    #     # Update edge weights
    #     alpha = edge_weight_model(alpha, edge_index)
    #     edge_weight_time = time.time()
        
    #     # Message passing
    #     out = message_passing_model(x, edge_index, alpha)
    #     message_passing_time = time.time()
        
    #     # Output aggregation
    #     out = output_model(out)
    #     output_time = time.time()

    # # Total and individual step times
    # print(f"Total inference time: {(output_time - start_time) * 100:.4f} ms")
    # print(f"Linear Transformation Time: {(linear_time - start_time) * 100:.4f} ms")
    # print(f"Attention Coefficients Time: {(attention_time - linear_time) * 100:.4f} ms")
    # print(f"Self-Loop Addition Time: {(self_loop_time - attention_time) * 100:.4f} ms")
    # print(f"Edge Weight Update Time: {(edge_weight_time - self_loop_time) * 100:.4f} ms")
    # print(f"Message Passing Time: {(message_passing_time - edge_weight_time) * 100:.4f} ms")
    # print(f"Output Aggregation Time: {(output_time - message_passing_time) * 100:.4f} ms")
