import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Reddit2
from torch_geometric.transforms import NormalizeFeatures
import time
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

# load dataset
dataset = PygNodePropPredDataset('ogbn-products', root="/tmp/ogbnProduct")
data = dataset[0]


# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


model = GCN(hidden_channels=16)

# load
model = GCN(hidden_channels=16)
model.load_state_dict(torch.load('Weights/ogbnProduct_GCN_weights.pth'))

# Measure inference time
def measure_inference_time(model, data):
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        model(data.x, data.edge_index)  # Inference on the whole graph
        end_time = time.time()
    # Calculate time for the whole graph and average per node in the test mask
    inference_time = (end_time - start_time) * 1000
    return inference_time
    
 
# Measure inference time
def measure_inference_time(model, data, runs):
    model.eval()
    timings = []
    for i in range(2):
        model(data.x, data.edge_index)
    # with torch.no_grad():
    for i in range(runs):
        start_time = time.time()
        model(data.x, data.edge_index)  # Inference on the whole graph
        end_time = time.time()
        timings.append(end_time - start_time)
    inference_time = sum(timings) / runs * 1000
    return inference_time

avg_inference_time = measure_inference_time(model, data, 3)
print(f'Model: GCN, Dataset: ogbn-product, Average inference time: {avg_inference_time:.3f} ms')

