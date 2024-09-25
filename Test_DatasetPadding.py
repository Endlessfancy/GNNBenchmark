###
# para: N=5000,
###
import torch
import pandas as pd
import torch.nn as nn
from torch_geometric.loader import NeighborLoader
from torch_geometric.datasets import Flickr
from torch_geometric.nn import GCNConv
import os
import onnxruntime as ort
import torch.onnx

# Load dataset, with the path 'data/Flickr'
dataset = Flickr(root='data/Flickr')
data = dataset[0]

# Define the sampler using NeighborLoader with full neighbor sampling and a batch size of 512
subgraph_loader = NeighborLoader(
    data,
    input_nodes=None,
    num_neighbors=[-1],
    batch_size=512,
    # num_workers=12, 
    # persistent_workers=True
)

# Function to pad nodes and edges to the nearest multiple of 5000
def pad_to_multiple_of_5000(batch):
    batch_num_nodes = batch.num_nodes
    batch_num_edges = batch.edge_index.size(1)

    # Calculate the number of nodes and edges to pad to the next multiple of 5000
    padding_nodes = (5000 - batch_num_nodes % 5000) % 5000
    padding_edges = (5000 - batch_num_edges % 5000) % 5000

    # Pad nodes by adding zero features
    if padding_nodes > 0:
        padding_node_features = torch.zeros((padding_nodes, batch.x.size(1)))
        batch.x = torch.cat([batch.x, padding_node_features], dim=0)

    # Pad edges by adding (0, 0) self-loop edges
    if padding_edges > 0:
        padding_edge_index = torch.tensor([[0] * padding_edges, [0] * padding_edges])
        batch.edge_index = torch.cat([batch.edge_index, padding_edge_index], dim=1)

    # Update the number of nodes
    batch.num_nodes = batch.x.size(0)

    return batch

# Set to store all unique (node, edge) combinations after padding
unique_nodes_edges_combinations = set()

# Use the loader to iterate over batches, pad them, and store the unique results
for batch in subgraph_loader:
    print(f"Padded batch to: {batch.num_nodes} nodes, {batch.edge_index.size(1)} edges.")
    batch = pad_to_multiple_of_5000(batch)
    # Add the combination of (number of nodes, number of edges) to the set
    unique_nodes_edges_combinations.add((batch.num_nodes, batch.edge_index.size(1)))
    print(f"Padded batch to: {batch.num_nodes} nodes, {batch.edge_index.size(1)} edges.")

# Convert the set to a list and export the unique (node, edge) combinations to a CSV file
df = pd.DataFrame(list(unique_nodes_edges_combinations), columns=['Number of Nodes', 'Number of Edges'])
df.to_csv('unique_nodes_edges_combinations.csv', index=False)
print("Exported the unique padded (node, edge) combinations to 'unique_nodes_edges_combinations.csv'.")

# Define a single-layer GCN network
class GCNNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x

# Load the unique (node, edge) combinations from the CSV file
df = pd.read_csv('unique_nodes_edges_combinations.csv')

# Feature size is set to 16 as per the requirement
feature_size = 16
output_size = 16  # You can adjust the output size of the GCN as needed

# Iterate through each unique combination of (node, edge) sizes
for index, row in df.iterrows():
    num_nodes = int(row['Number of Nodes'])
    num_edges = int(row['Number of Edges'])

    # Create a dummy input for nodes and edges
    dummy_x = torch.randn((num_nodes, feature_size), requires_grad=True)  # Node feature matrix
    dummy_edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)  # Random edge index

    # Initialize the GCN model
    model = GCNNet(in_channels=feature_size, out_channels=output_size)

    # Set the model to evaluation mode
    model.eval()

    # Create a filename based on the number of nodes and edges
    onnx_filename = f'Onnx/padding/gcn_model_nodes_{num_nodes}_edges_{num_edges}.onnx'

    # Export the model to ONNX
    torch.onnx.export(
        model,                         # Model to export
        (dummy_x, dummy_edge_index),    # Inputs (node features and edge index)
        onnx_filename,                  # Output ONNX file name
        input_names=['node_features', 'edge_index'],  # Input names for ONNX model
        output_names=['output'],        # Output names for ONNX model
        # dynamic_axes={
        #     'node_features': {0: 'num_nodes', 1: 'num_features'},  # Dynamic node and feature sizes
        #     'edge_index': {1: 'num_edges'}                         # Dynamic edge size
        # },
        opset_version=12                # Specifying the ONNX opset version
    )

    print(f"Exported ONNX model with {num_nodes} nodes and {num_edges} edges to {onnx_filename}.")

     # Check if the ONNX file exists
    if not os.path.exists(onnx_filename):
        print(f"ONNX file {onnx_filename} does not exist.")
        continue

    # Load the ONNX model using onnxruntime
    try:
        session = ort.InferenceSession(onnx_filename)

        # Generate random dummy inputs (same as during the export)
        dummy_x = torch.randn((num_nodes, feature_size)).numpy()  # Node feature matrix
        dummy_edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long).numpy()  # Random edge index

        # Prepare inputs for the ONNX model
        input_feed = {
            'node_features': dummy_x,
            'edge_index': dummy_edge_index
        }

        # Perform inference
        outputs = session.run(None, input_feed)

        # Output the result to ensure it's working
        print(f"ONNX inference successful for {onnx_filename}. Output shape: {outputs[0].shape}")

    except Exception as e:
        print(f"Failed to run inference for {onnx_filename}. Error: {str(e)}")


   
