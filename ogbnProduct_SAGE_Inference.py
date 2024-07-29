import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Reddit2
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.transforms import NormalizeFeatures
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from tqdm import tqdm
import time
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader


# load dataset
dataset = PygNodePropPredDataset('ogbn-products', root="/tmp/ogbnProduct")
data = dataset[0]

# Ensure labels are on the correct device and not None
data.y = data.y if data.y is not None else torch.zeros((data.num_nodes,), dtype=torch.long)

kwargs = {'batch_size': 1024, 'num_workers': 6, 'persistent_workers': True}

# Measure the time it takes to initialize the subgraph loader
start_time = time.time()
subgraph_loader = NeighborLoader(data, input_nodes=None,
                                 num_neighbors=[-1], shuffle=False, **kwargs)
loader_init_time = (time.time() - start_time)*1000
print(f"Subgraph Loader Initialization Time: {loader_init_time:.4f} ms")

# # # No need to maintain these features during evaluation:
# # del subgraph_loader.data.x, subgraph_loader.data.y
# # Add global node index information.
# subgraph_loader.data.num_nodes = data.num_nodes
# subgraph_loader.data.n_id = torch.arange(data.num_nodes)

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id]
                x = conv(x, batch.edge_index)
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
            x_all = torch.cat(xs, dim=0)
        return x_all

# laod
model = SAGE(dataset.num_features, 256, dataset.num_classes)
model.load_state_dict(torch.load('Weights/ogbnProduct_SAGE_weights.pth'))

# Measure inference time
def measure_inference_time(model, data, runs):
    model.eval()
    timings = []
    # for i in range(2):
    #     model(data.x, data.edge_index)
    # with torch.no_grad():
    for i in range(runs):
        start_time = time.time()
        model(data.x, data.edge_index)  # Inference on the whole graph
        end_time = time.time()
        timings.append(end_time - start_time)
    inference_time = sum(timings) / runs * 1000
    return inference_time

avg_inference_time = measure_inference_time(model, data, 2)
print(f'Model: SAGE, Dataset: ogbn-product, Average inference time: {avg_inference_time:.3f} ms')  
 
# def inference(graph, batch_size, model):
#     model.eval()
#     start_time = time.time()
#     with torch.no_grad():
#         for i, (input_nodes, output_nodes, subgraph) in enumerate(dgl.dataloading.NodeDataLoader(
#                 graph, torch.arange(graph.number_of_nodes()), dgl.dataloading.MultiLayerFullNeighborSampler(2), 
#                 batch_size=batch_size, shuffle=True, drop_last=False)):
#             subgraph = subgraph.to('cuda')
#             inputs = subgraph.ndata['feat']
#             model(subgraph, inputs)
#         end_time = time.time()
#         # Calculate time for the whole graph and average per node in the test mask
#     inference_time = (end_time - start_time) * 1000
#     return inference_time


# avg_inference_time = inference(graph, batch_size=1024, model=model
# # avg_inference_time = measure_inference_time(model, data)
# print(f'Average inference time per node: {avg_inference_time:.3f} ms')

# # Measure inference time
# def measure_inference_time_withoutSampling(model, data):
#     model.eval()
#     with torch.no_grad():
#         start_time = time.time()
#         model(data.x, data.edge_index)  # Inference on the whole graph
#         end_time = time.time()
#     # Calculate time for the whole graph and average per node in the test mask
#     inference_time = (end_time - start_time) * 1000
#     return inference_time

# avg_inference_time = measure_inference_time_withoutSampling(model, data)
# print(f'Average inference time without sampling: {avg_inference_time:.3f} ms')

# # Measure inference time
# def measure_inference_time_withSampling(model, data):
#     model.eval()
#     batch_count = 0
#     with torch.no_grad():
#         start_time = time.time()
#         for batch in subgraph_loader:
#           model(batch.x, batch.edge_index)  # Inference on the whole graph
#           batch_count += 1
#         end_time = time.time()
#     # Calculate time for the whole graph and average per node in the test mask
#     inference_time = (end_time - start_time) * 1000
#     print(f"inference with sampling: number of target nodes {batch_count}")
#     return inference_time, batch_count

# avg_inference_time, batch_count = measure_inference_time_withSampling(model, data)
# print(f'Average inference time with sampling: {avg_inference_time:.3f} ms')
# print(f'Average inference time with sampling per target node: {avg_inference_time/batch_count:.3f} ms')


