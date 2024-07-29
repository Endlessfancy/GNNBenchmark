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
dataset = Reddit2(root='/tmp/Reddit2')
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
# torch.save(model.state_dict(), 'Reddit2_GCN_weights.pth')

# load
model = GCN(hidden_channels=16)
model.load_state_dict(torch.load('Weights/Reddit2_GCN_weights.pth'))

# Dummy input for the model
dummy_input = (data.x, data.edge_index)

# Export the model to ONNX format
torch.onnx.export(model, dummy_input, 'Onnx/Reddit2_GCN.onnx', opset_version=11,
                  input_names=['x', 'edge_index'],
                  output_names=['output'])
                 

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


avg_inference_time = measure_inference_time(model, data)
print(f'Average inference time Reddit2: {avg_inference_time:.3f} ms')
