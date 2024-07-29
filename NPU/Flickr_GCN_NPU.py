import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Flickr
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import time
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import intel_npu_acceleration_library
import torch._dynamo

# Suppress errors to fall back to eager execution if needed
torch._dynamo.config.suppress_errors = True


# load dataset
dataset = Flickr(root='/tmp/Flickr')
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
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# criterion = torch.nn.CrossEntropyLoss()
#
#
# def train():
#     model.train()
#     optimizer.zero_grad()
#     out = model(data.x, data.edge_index)
#     loss = criterion(out[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer.step()
#     return loss
#
#
# # Run training cycles
# for epoch in range(1, 101):
#     train()
#
# # save weight
# torch.save(model.state_dict(), 'Flickr_GCN_weights.pth')

# load
model = GCN(hidden_channels=16)
model.load_state_dict(torch.load('../Weights/Flickr_GCN_weights.pth'))

# Compile the model for NPU
optimized_model = torch.compile(model, backend="npu")

def measure_inference_time(model, data, runs):
    model.eval()
    timings = []
    model(data.x, data.edge_index)
    # with torch.no_grad():
    for i in range(runs):
        start_time = time.time()
        model(data.x, data.edge_index)  # Inference on the whole graph
        end_time = time.time()
        timings.append(end_time - start_time)
    inference_time = sum(timings) / runs * 1000
    return inference_time


avg_inference_time = measure_inference_time(optimized_model, data, 10)
print(f'Average inference time Cora: {avg_inference_time:.3f} ms')
