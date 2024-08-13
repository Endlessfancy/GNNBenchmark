import os.path as osp
import os
import torch
import torch.nn.functional as F
import time
import argparse
import warnings

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GCNConv

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Flickr
from torch_geometric.datasets import Reddit2

# GPU
import openvino.runtime as ov
from openvino.tools.mo import convert_model
# NPU
# import intel_npu_acceleration_library
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
import torch._dynamo
# Suppress errors to fall back to eager execution if needed
torch._dynamo.config.suppress_errors = True

class SAGE(torch.nn.Module):
   def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
      super().__init__()
      self.num_layers = num_layers
      self.convs = torch.nn.ModuleList()
      self.convs.append(SAGEConv(in_channels, hidden_channels))
      for _ in range(num_layers - 2):
          self.convs.append(SAGEConv(hidden_channels, hidden_channels))
      self.convs.append(SAGEConv(hidden_channels, out_channels))

   def reset_parameters(self):
      for conv in self.convs:
          conv.reset_parameters()

   def forward(self, x, edge_index):
      for i, conv in enumerate(self.convs):
          x = conv(x, edge_index)
          if i != self.num_layers - 1:
              x = x.relu()
              x = F.dropout(x, p=0.5, training=self.training)
      return x

   def samplingInference(self, x_all, subgraph_loader):
      for i in range(self.num_layers):
          xs = []
          for batch in subgraph_loader:
              x = x_all[batch.n_id]
              edge_index = batch.edge_index
              x = self.convs[i](x, edge_index)
              x = x[:batch.batch_size]
              if i != self.num_layers - 1:
                  x = x.relu()
              xs.append(x.cpu())
          x_all = torch.cat(xs, dim=0)
      return x_all

   def inference(self, x, edge_index):
       for i, conv in enumerate(self.convs):
           x = conv(x, edge_index)
           if i != self.num_layers - 1:
               x = x.relu()
       return x


# Define the GCN model
class GCN(torch.nn.Module):
   def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
       super(GCN, self).__init__()
       self.num_layers = num_layers
       self.convs = torch.nn.ModuleList()
       self.convs.append(GCNConv(in_channels, hidden_channels))
       for _ in range(num_layers - 2):
          self.convs.append(GCNConv(hidden_channels, hidden_channels))
       self.convs.append(GCNConv(hidden_channels, out_channels))

   def reset_parameters(self):
      for conv in self.convs:
          conv.reset_parameters()

   def forward(self, x, edge_index):
      for i, conv in enumerate(self.convs):
          x = conv(x, edge_index)
          if i != self.num_layers - 1:
              x = x.relu()
              x = F.dropout(x, p=0.5, training=self.training)
      return x
  
   def inference(self, x, edge_index):
       for i, conv in enumerate(self.convs):
           x = conv(x, edge_index)
           if i != self.num_layers - 1:
               x = x.relu()
       return x


@torch.no_grad()
def samplingInference(loader, compiled_model, runs=2):
    import time
    timings = []
    # Warm up
    for _ in range(1):
        for batch in loader:
            input_data = batch.x.numpy()
            edge_index = batch.edge_index.numpy()
            _ = compiled_model([input_data, edge_index])
    
    for _ in range(runs):
        start_time = time.time()
        for batch in loader:
            input_data = batch.x.numpy()
            edge_index = batch.edge_index.numpy()
            _ = compiled_model([input_data, edge_index])
        end_time = time.time()
        timings.append(end_time - start_time)
    
    avg_time = sum(timings) * 1000 / runs
    print(f'Average Inference Time: {avg_time:.6f} ms')
    return avg_time


@torch.no_grad()
def fullInference(compiled_model, data, runs=2):
    import time
    input_data = data.x.numpy()
    edge_index = data.edge_index.numpy()
    timings = []
    for _ in range(1):
        _ = compiled_model([input_data, edge_index])
    for _ in range(runs):
        start_time = time.time()
        _ = compiled_model([input_data, edge_index])
        end_time = time.time()
        timings.append(end_time - start_time)
    avg_time = sum(timings) *1000 / runs
    print(f'Average Inference Time: {avg_time:.6f} ms')
    return avg_time

def main():
   parser = argparse.ArgumentParser(description="Example script to pass model and dataset names.")
   parser.add_argument('--model', type=str, required=True, help='The name of the model.')
   parser.add_argument('--data', type=str, required=True, help='The name of the dataset.')
   parser.add_argument('--mode', type=str, required=True, help='The mode of the Inference.')
   parser.add_argument('--runs', type=str, required=True, help='The number of runs.')
   parser.add_argument('--device', type=str, required=True, help='The target device.')
   args = parser.parse_args()

   print(f"Model Name: {args.model}")
   print(f"Dataset Name: {args.data}")
   print(f"Inference Mode Name: {args.mode}")
   print(f"Run Times: {args.runs}")
   print(f"Target Device: {args.device}")

   # set
   MODEL = args.model
   DATASET = args.data
   MODE = args.mode
   RUN = int(args.runs)
   DEVICE =args.device

   # load dataset
   if DATASET == "Cora":
      dataset = Planetoid(root='data/Cora', name='Cora', transform=NormalizeFeatures())
      data = dataset[0]
   elif DATASET == "Flickr":
      dataset = Flickr(root='data/Flickr', transform=NormalizeFeatures())
      data = dataset[0]
   elif DATASET == "Reddit2":
      dataset = Reddit2(root='data/Reddit2', transform=NormalizeFeatures())
      data = dataset[0]    
   elif DATASET == "Products":
      dataset = PygNodePropPredDataset('ogbn-products', root="data/products", transform=NormalizeFeatures())
      data = dataset[0]   

   # sample the dataset
   if MODE == "samplingInference":
       subgraph_loader = NeighborLoader(
       data,
       input_nodes=None,
       num_neighbors=[-1],
       batch_size=256,
       num_workers=1,
       persistent_workers=True,
       pin_memory=False
       )
       print("sampled!")

   # create model
   if MODEL == "GCN":
       model = GCN(dataset.num_features, 16, dataset.num_classes, 2)
       # Initialize the model weights to 1.
       with torch.no_grad():
          for param in model.parameters():
            param.fill_(1)
   elif MODEL == "SAGE":
       model = SAGE(dataset.num_features, 256, dataset.num_classes, num_layers=2)
       # Initialize the model weights to 1.
       with torch.no_grad():
          for param in model.parameters():
            param.fill_(1)
   else:
       print("model error!")
   if DEVICE == "NPU":
       # Compile the model for NPU
       model = torch.compile(model, backend="npu")
       print("Covert to NPU!")
   elif DEVICE == "GPU":
       # Initialize OpenVINO runtime
       core = ov.Core()
       # Load the model
       model_ir_path = "Onnx/model_ir/" +DATASET +"_" +MODEL +".xml"
       model_onnx_path = "Onnx/" +DATASET +"_" +MODEL +".onnx"
    #    # Dummy input for the model
    #    dummy_input = (data.x, data.edge_index)
    #    # Export the model to ONNX format
    #    if not osp.exists(model_ir_path):
    #       torch.onnx.export(model, dummy_input, model_onnx_path, opset_version=11,
    #                         input_names=['x', 'edge_index'],
    #                         output_names=['output'])
    #       print("Onxx dosen't exits, generated")
       with warnings.catch_warnings():
          warnings.filterwarnings("ignore")
          if not osp.exists(model_onnx_path):
              dummy_input = (data.x, data.edge_index)
              torch.onnx.export(
                  model,
                  dummy_input,
                  model_onnx_path,
              )
              print(f"ONNX model exported to {model_onnx_path}.")
          else:
              print(f"ONNX model {model_onnx_path} already exists.")
       if not osp.exists(model_ir_path):
          print("Exporting ONNX model to IR... This may take a few minutes.")
        #   os.system('mo --input_model Onnx/Reddit2_SAGE.onnx --output_dir Onnx/model_ir --input data.x, data.edge_index')
        #   ov_model = ov.convert_model(model_onnx_path)
          # ov_model = convert_model(model)
        #   ov.save_model(ov_model, model_ir_path)
       else:
          print(f"IR model {model_ir_path} already exists.")
          model = core.compile_model(model=model_ir_path, device_name="GPU")
       print("Covert to GPU!")
        
        # # Convert data to numpy arrays
        # input_data = data.x.numpy()
        # edge_index = data.edge_index.numpy()

        # # Perform inference
        # result = compiled_model([input_data, edge_index])


   # run inference
   if args.model == "GCN":
       if args.mode == "fullInference":
           fullInference(model, data, RUN)
       elif args.mode == "samplingInference":
           samplingInference(subgraph_loader, model, RUN)
   elif args.model  == "SAGE":
       if args.mode == "fullInference":
           fullInference(model, data, RUN)
       elif args.mode == "samplingInference":
           samplingInference(subgraph_loader, model, RUN)
   print("Finished!!")


if __name__ == "__main__":
   main()
