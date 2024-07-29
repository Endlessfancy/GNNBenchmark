import numpy as np
import openvino.runtime as ov
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

# Load the Cora dataset

# load dataset
dataset = Flickr(root='/tmp/Flickr')
data = dataset[0]

# Convert data to numpy arrays
input_data = data.x.numpy()
edge_index = data.edge_index.numpy()

# Initialize OpenVINO runtime
ie = ov.Core()

# Load the model
model_ir_path = "../model_ir/Flickr_GCN_Onnx.xml"
compiled_model = ie.compile_model(model=model_ir_path, device_name="GPU")

# Prepare input and output
input_layer_x = compiled_model.input(0)
input_layer_edge_index = compiled_model.input(1)
output_layer = compiled_model.output(0)

# Perform inference
result = compiled_model([input_data, edge_index])

# Get the output
output = result[output_layer]
pred = np.argmax(output, axis=1)

print(f'Inference result: {pred}')

# Measure the average inference time
def measure_inference_time(compiled_model, input_data, edge_index, runs=100):
    import time
    timings = []
    for _ in range(runs):
        start_time = time.time()
        _ = compiled_model([input_data, edge_index])
        end_time = time.time()
        timings.append(end_time - start_time)
    avg_time = sum(timings) / runs
    return avg_time

avg_inference_time = measure_inference_time(compiled_model, input_data, edge_index)
print(f'Average Inference Time: {avg_inference_time:.6f} seconds')

