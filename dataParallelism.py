import os.path as osp
import torch
import torch.nn.functional as F
import time
import argparse

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GCNConv

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Flickr
from torch_geometric.datasets import Reddit2

# GPU
import openvino.runtime as ov
# NPU
# import intel_npu_acceleration_library
# from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
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

#    def samplingInference(self, x_all, subgraph_loader):
#       for i in range(self.num_layers):
#           xs = []
#           for batch in subgraph_loader:
#               x = x_all[batch.n_id]
#               edge_index = batch.edge_index
#               x = self.convs[i](x, edge_index)
#               x = x[:batch.batch_size]
#               if i != self.num_layers - 1:
#                   x = x.relu()
#               xs.append(x.cpu())
#           x_all = torch.cat(xs, dim=0)
#       return x_all

#    def inference(self, x, edge_index):
#        for i, conv in enumerate(self.convs):
#            x = conv(x, edge_index)
#            if i != self.num_layers - 1:
#                x = x.relu()
#        return x

class SAGELayer1(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return x

class SAGELayer2(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv2(x, edge_index)
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

#    def inference(self, x, edge_index):
#        for i, conv in enumerate(self.convs):
#            x = conv(x, edge_index)
#            if i != self.num_layers - 1:
#                x = x.relu()
#        return x

    def forward_layer(self, x, edge_index, layer_index):
        x = self.convs[layer_index](x, edge_index)
        if layer_index != self.num_layers - 1:
            x = x.relu()
        return x

# Define two separate models for exporting the layers
class GCNLayer(torch.nn.Module):
    def __init__(self, conv):
        super(GCNLayer, self).__init__()
        self.conv = conv

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

@torch.no_grad()
def samplingInference(loader, compiled_model, runs=2):
    import time
    timings = []
    # Warm up
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
    _ = compiled_model([input_data, edge_index])
    for _ in range(runs):
        start_time = time.time()
        _ = compiled_model([input_data, edge_index])
        end_time = time.time()
        timings.append(end_time - start_time)
    avg_time = sum(timings) *1000 / runs
    print(f'Average Inference Time: {avg_time:.6f} ms')
    return avg_time

def callback(request, userdata):
            # Process the output
            result = request.get_output_tensor().data
            # Do something with the result
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

    ##############################
    # load dataset
    ##############################
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
        batch_size=4096,
        num_workers=12,
        persistent_workers=True,
        )
        print("sampled!")
    # for batch in subgraph_loader:
    #     print(f"node shape {batch.x.shape} edge {batch.edge_index.shape} #node {batch.num_nodes} #edges {batch.num_edges}")
    # print(f"full graph shape {data.x.shape} edge {data.edge_index.shape} #node {data.num_nodes} #edges {data.num_edges}")


    ##############################
    # create model
    ##############################
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

    ##############################
    # select device
    ##############################
    if DEVICE == "NPU":
        # Compile the model for NPU
        #   model = torch.compile(model, backend="npu")
        # Initialize OpenVINO runtime
        core = ov.Core()
        # Load the model
        model_ir_path = "Ir/" +DATASET +"_" +MODEL +".xml"
        model = core.compile_model(model=model_ir_path, device_name="NPU")
        print("Covert to NPU!")
    
    elif DEVICE == "GPU":
        # Initialize OpenVINO runtime
        core = ov.Core()
        # Load the model
        model_ir_path = "Ir/" +DATASET +"_" +MODEL +".xml"
        model = core.compile_model(model=model_ir_path, device_name="GPU")
        print("Covert to GPU!")

    elif DEVICE == "dataParallelism" and MODE == "fullInference":
        # Initialize OpenVINO runtime
        core = ov.Core()
        # Load the model
        model_ir_path = "Ir/" +DATASET +"_" +MODEL +".xml"
    #    model = ie.compile_model(model=model_ir_path, device_name="NPU")
        print("Covert to GPU!")
        
        # Asynchronous inference example
        compiled_model = core.compile_model(model=model_ir_path, device_name="GPU")
        infer_request = compiled_model.create_infer_request()
        input_data = data.x.numpy()
        edge_index = data.edge_index.numpy()
        #    _ = compiled_model([input_data, edge_index])

        # Assuming input_data and edge_index are numpy arrays
        input_tensor = ov.Tensor(array=input_data)

        # If you need to pass multiple tensors, you would do something like:
        edge_index_tensor = ov.Tensor(array=edge_index)

        # Then, use these tensors in your infer_request
        infer_request.set_tensor(compiled_model.input(0), input_tensor)
        infer_request.set_tensor(compiled_model.input(1), edge_index_tensor)

        timings = []
        start_time = time.time()
        infer_request.start_async()
        infer_request.wait()
        res = infer_request.get_output_tensor(0).data
        end_time = time.time()
        timings.append(end_time - start_time)
        print(res)
        print(f'Average Inference Time: {sum(timings)} ms')

    elif DEVICE == "dataParallelism" and MODE == "samplingInference":
        model_ir_path = "Ir/" +"dynaSampled_" +DATASET +"_" +MODEL +".xml"
        model_onnx_path = "Onnx/" +"dynaSampled_" +DATASET +"_" +MODEL +".onnx"
        # export new format:
    
        if not osp.exists(model_onnx_path):
            dummy_input = (data.x, data.edge_index)

            torch.onnx.export(
                model,
                dummy_input,
                model_onnx_path,
                input_names=['node_features', 'edge_index'],
                output_names=['output'],
                dynamic_axes={'node_features': {0: 'num_nodes'},  # 将 num_nodes 设为动态
                    'edge_index': {1: 'num_edges'}  # 将 num_edges 设为动态
                    },
                opset_version=11  # 根据需要调整 opset 版本
                )


            print(f"Sampled ONNX model exported to {model_onnx_path}.")
        else:
            print(f"Sampled ONNX model {model_onnx_path} already exists.")
        if not osp.exists(model_ir_path):
            print("Exporting ONNX model to IR... This may take a few minutes.")
            # os.system('mo --input_model Onnx/Sampled_Flickr_SAGE.onnx --output_dir Ir')
        else:
            print(f"IR model {model_ir_path} already exists.")
    
        # Warm up
        timings = []
        cnt = 0
        buffer = None
        # Initialize OpenVINO runtime
        core = ov.Core()
     
        # Asynchronous inference example
        compiled_model = core.compile_model(model=model_ir_path, device_name="GPU")
        infer_request = compiled_model.create_infer_request()
        for batch in subgraph_loader:
            # print(len(batch.x))
            # print(len(batch.edge_index))
            if buffer == None:
                buffer = batch
            else:
                subbatch_1 = buffer
                subbatch_2 = batch

                input_data_1 = subbatch_1.x.numpy()
                edge_index_1 = subbatch_1.edge_index.numpy()
                input_data_2 = subbatch_2.x.numpy()
                edge_index_2 = subbatch_2.edge_index.numpy()
            
                # Assuming input_data and edge_index are numpy arrays
                input_tensor_2 = ov.Tensor(array=input_data_2)
                edge_index_tensor_2 = ov.Tensor(array=edge_index_2)
                # Then, use these tensors in your infer_request
                infer_request.set_tensor(compiled_model.input(0), input_tensor_2)
                infer_request.set_tensor(compiled_model.input(1), edge_index_tensor_2)
            
                start_time = time.time()
                infer_request.start_async()
                infer_request.wait()
                res = infer_request.get_output_tensor(0).data
                end_time = time.time()
                timings.append(end_time - start_time)
               
                buffer = None

        print(f'count {cnt} :Average Inference Time: {sum(timings)*1000} ms')
    elif DEVICE == "ratioDataParallelism" and MODE == "samplingInference":    
        core = ov.Core()
        # Load the model
        model_ir_path = "Ir/" +"dynaSampled_" +DATASET +"_" +MODEL +".xml"
        compiled_model1 = core.compile_model(model=model_ir_path, device_name="GPU")
        compiled_model2 = core.compile_model(model=model_ir_path, device_name="CPU")

        # warm up
        for batch in subgraph_loader:
            infer_request1 = compiled_model1.create_infer_request()
            infer_request2 = compiled_model2.create_infer_request()
            input_data = batch.x.numpy()
            edge_index = batch.edge_index.numpy()
            # Assuming input_data and edge_index are numpy arrays
            input_tensor = ov.Tensor(array=input_data)
            edge_index_tensor = ov.Tensor(array=edge_index)
            # Then, use these tensors in your infer_request
            infer_request1.set_tensor(compiled_model1.input(0), input_tensor)
            infer_request1.set_tensor(compiled_model1.input(1), edge_index_tensor)
            infer_request2.set_tensor(compiled_model2.input(0), input_tensor)
            infer_request2.set_tensor(compiled_model2.input(1), edge_index_tensor)
            infer_request1.start_async()
            infer_request2.start_async()
            infer_request1.wait()
            infer_request2.wait()         
        print("warm up finished!")

        buffer1 = []
        buffer2 = []
        timings = []
        infer_queue1 = []
        infer_queue2 = []
        cnt = 0
        batchCnt = 0

        a = 5  # example value for buffer1 size
        b = 1  # example value for buffer2 size

        for batch in subgraph_loader:
            if len(buffer1) < a:
                buffer1.append(batch)
                batchCnt = batchCnt + 1
            elif len(buffer2) < b:
                buffer2.append(batch)
                batchCnt = batchCnt + 1
            if len(buffer1) == a and len(buffer2) == b and batchCnt < len(subgraph_loader):
                for subbatch_1 in buffer1:
                    input_data_1 = subbatch_1.x.numpy()
                    edge_index_1 = subbatch_1.edge_index.numpy()
                  
                    # Assuming input_data and edge_index are numpy arrays
                    input_tensor_1 = ov.Tensor(array=input_data_1)
                    edge_index_tensor_1 = ov.Tensor(array=edge_index_1)
                    # Then, use these tensors in your infer_request
                   
                    infer_request = compiled_model1.create_infer_request()
                    infer_request.set_tensor(compiled_model1.input(0), input_tensor_1)
                    infer_request.set_tensor(compiled_model1.input(1), edge_index_tensor_1)
                    
                    infer_queue1.append(infer_request)

                for subbatch_2 in buffer2:
                    input_data_2 = subbatch_2.x.numpy()
                    edge_index_2 = subbatch_2.edge_index.numpy()
                  
                    # Assuming input_data and edge_index are numpy arrays
                    input_tensor_2 = ov.Tensor(array=input_data_2)
                    edge_index_tensor_2 = ov.Tensor(array=edge_index_2)
                    # Then, use these tensors in your infer_request
                   
                    infer_request = compiled_model2.create_infer_request()
                    infer_request.set_tensor(compiled_model2.input(0), input_tensor_2)
                    infer_request.set_tensor(compiled_model2.input(1), edge_index_tensor_2)
                    
                    infer_queue2.append(infer_request)
                
                start_time = time.time()
                for infer_request in infer_queue1:
                    infer_request.start_async()
                for infer_request in infer_queue2:
                    infer_request.start_async()
       
                for infer_request in infer_queue1:
                    infer_request.wait()
                for infer_request in infer_queue2:
                    infer_request.wait()

                # res = infer_request.get_output_tensor(0).data
                end_time = time.time()
                timings.append(end_time - start_time)
                cnt = cnt + 1
                # print(f"cnt {cnt}: time {end_time - start_time}")
                
                # clear buffer
                buffer1.clear()
                buffer2.clear()
                infer_queue1.clear()
                infer_queue2.clear()
            elif batchCnt == len(subgraph_loader):
                for subbatch_1 in buffer1:
                    input_data_1 = subbatch_1.x.numpy()
                    edge_index_1 = subbatch_1.edge_index.numpy()
                  
                    # Assuming input_data and edge_index are numpy arrays
                    input_tensor_1 = ov.Tensor(array=input_data_1)
                    edge_index_tensor_1 = ov.Tensor(array=edge_index_1)
                    # Then, use these tensors in your infer_request
                   
                    infer_request = compiled_model1.create_infer_request()
                    infer_request.set_tensor(compiled_model1.input(0), input_tensor_1)
                    infer_request.set_tensor(compiled_model1.input(1), edge_index_tensor_1)
                    
                    infer_queue1.append(infer_request)

                for subbatch_2 in buffer2:
                    input_data_2 = subbatch_2.x.numpy()
                    edge_index_2 = subbatch_2.edge_index.numpy()
                  
                    # Assuming input_data and edge_index are numpy arrays
                    input_tensor_2 = ov.Tensor(array=input_data_2)
                    edge_index_tensor_2 = ov.Tensor(array=edge_index_2)
                    # Then, use these tensors in your infer_request
                   
                    infer_request = compiled_model2.create_infer_request()
                    infer_request.set_tensor(compiled_model2.input(0), input_tensor_2)
                    infer_request.set_tensor(compiled_model2.input(1), edge_index_tensor_2)
                    
                    infer_queue2.append(infer_request)
                
                start_time = time.time()
                for infer_request in infer_queue1:
                    infer_request.start_async()
                for infer_request in infer_queue2:
                    infer_request.start_async()
       
                for infer_request in infer_queue1:
                    infer_request.wait()
                for infer_request in infer_queue2:
                    infer_request.wait()

                # res = infer_request.get_output_tensor(0).data
                end_time = time.time()
                timings.append(end_time - start_time)
                cnt = cnt + 1
                # print(f"cnt {cnt}: time {end_time - start_time}")
                   
                # clear buffer
                buffer1.clear()
                buffer2.clear()
                infer_queue1.clear()
                infer_queue2.clear()

        print(f'count {cnt} :Average Inference Time: {sum(timings)*1000/cnt} ms')
        print(f'count {cnt} :  Total Inference Time: {sum(timings)*1000} ms')

    elif DEVICE == "pipelineParallelism" and MODE == "samplingInference": 
        # path
        layer1_onnx_path = "Onnx/SAGE_layer1.onnx"
        layer1_ir_path = "Ir/SAGE_layer1.xml"
        layer2_onnx_path = "Onnx/SAGE_layer2.onnx"
        layer2_ir_path = "Ir/SAGE_layer2.xml"

        # Assuming in_channels, hidden_channels, and out_channels are already defined
        sage_layer1 = SAGELayer1(dataset.num_features, 256)
        sage_layer2 = SAGELayer2(256, dataset.num_classes)

        # Dummy input for export
        dummy_input_layer1 = (data.x, data.edge_index)

        # Export the first layer
        if not osp.exists(layer1_onnx_path):
            torch.onnx.export(
                sage_layer1,
                dummy_input_layer1,
                layer1_onnx_path,
                input_names=['node_features', 'edge_index'],
                output_names=['output'],
                dynamic_axes={'node_features': {0: 'num_nodes'},
                            'edge_index': {1: 'num_edges'}},
                opset_version=11
            )
            print(f"Sampled ONNX model exported to {layer1_onnx_path}.")
        else:
            print(f"Sampled ONNX model {layer1_onnx_path} already exists.")
        if not osp.exists(layer1_ir_path):
            print("Exporting ONNX model to IR... This may take a few minutes.")
            # os.system('mo --input_model Onnx/Sampled_Flickr_SAGE.onnx --output_dir Ir')
        else:
            print(f"IR model {layer1_ir_path} already exists.")
        
        # mo --input_model Onnx/Sampled_Flickr_SAGE.onnx --output_dir Ir

        # Dummy input for the second layer (output dimensions of the first layer)
        with torch.no_grad():
            output_from_layer1 = sage_layer1(data.x, data.edge_index)

        dummy_input_layer2 = (output_from_layer1, data.edge_index)

        # Export the second layer
        if not osp.exists(layer2_onnx_path):
            torch.onnx.export(
                sage_layer2,
                dummy_input_layer2,
                layer2_onnx_path,
                input_names=['node_features', 'edge_index'],
                output_names=['output'],
                dynamic_axes={'node_features': {0: 'num_nodes'},
                            'edge_index': {1: 'num_edges'}},
                opset_version=11
            )
            print(f"Sampled ONNX model exported to {layer2_onnx_path}.")
        else:
            print(f"Sampled ONNX model {layer2_onnx_path} already exists.")
        if not osp.exists(layer2_ir_path):
            print("Exporting ONNX model to IR... This may take a few minutes.")
            # os.system('mo --input_model Onnx/Sampled_Flickr_SAGE.onnx --output_dir Ir')
        else:
            print(f"IR model {layer2_ir_path} already exists.")

        # mo --input_model Onnx/Sampled_Flickr_SAGE.onnx --output_dir Ir

        # Initialize OpenVINO runtime
        core = ov.Core()
     
        # Asynchronous inference example
        compiled_sage_layer1 = core.compile_model(model=layer1_ir_path, device_name="GPU")
        compiled_sage_layer2 = core.compile_model(model=layer2_ir_path, device_name="CPU")

        # Warm up
        for batch in subgraph_loader:
            infer_request1 = compiled_sage_layer1.create_infer_request()
            infer_request2 = compiled_sage_layer2.create_infer_request()
            input_data_1 = batch.x.numpy()
            edge_index_1 = batch.edge_index.numpy()

            # Assuming input_data and edge_index are numpy arrays
            input_tensor_1 = ov.Tensor(array=input_data_1)
            edge_index_tensor_1 = ov.Tensor(array=edge_index_1)
            # Then, use these tensors in your infer_request
            infer_request1.set_tensor(compiled_sage_layer1.input(0), input_tensor_1)
            infer_request1.set_tensor(compiled_sage_layer1.input(1), edge_index_tensor_1)
            input_data_2 = infer_request1.get_output_tensor(0).data
            edge_index_2 = edge_index_1
            input_tensor_2 = ov.Tensor(array=input_data_2)
            edge_index_tensor_2 = ov.Tensor(array=edge_index_2)
            infer_request2.set_tensor(compiled_sage_layer2.input(0), input_tensor_2)
            infer_request2.set_tensor(compiled_sage_layer2.input(1), edge_index_tensor_2)
            infer_request1.start_async()
            infer_request2.start_async()
            infer_request1.wait()
            infer_request2.wait()         
        print("warm up finished!")


        timings = []
        cnt = 0
        input_data_1 = None
        input_data_2 = None
        edge_index_1 = None
        edge_index_2 = None
        for batch in subgraph_loader:
            if cnt == 0:
                infer_request1 = compiled_sage_layer1.create_infer_request()

                input_data_1 = batch.x.numpy()
                edge_index_1 = batch.edge_index.numpy()
            
                # Assuming input_data and edge_index are numpy arrays
                input_tensor_1 = ov.Tensor(array=input_data_1)
                edge_index_tensor_1 = ov.Tensor(array=edge_index_1)
                # Then, use these tensors in your infer_request
                infer_request1.set_tensor(compiled_sage_layer1.input(0), input_tensor_1)
                infer_request1.set_tensor(compiled_sage_layer1.input(1), edge_index_tensor_1)
  
                start_time = time.time()
                infer_request1.start_async()
                infer_request1.wait()
                input_data_2 = infer_request1.get_output_tensor(0).data
                edge_index_2 = edge_index_1
                end_time = time.time()
                timings.append(end_time - start_time)
    
                cnt += 1
            else:
                infer_request1 = compiled_sage_layer1.create_infer_request()
                infer_request2 = compiled_sage_layer2.create_infer_request()

                input_data_1 = batch.x.numpy()
                edge_index_1 = batch.edge_index.numpy()
            
                # Assuming input_data and edge_index are numpy arrays
                input_tensor_1 = ov.Tensor(array=input_data_1)
                edge_index_tensor_1 = ov.Tensor(array=edge_index_1)
                # Then, use these tensors in your infer_request
                infer_request1.set_tensor(compiled_sage_layer1.input(0), input_tensor_1)
                infer_request1.set_tensor(compiled_sage_layer1.input(1), edge_index_tensor_1)

                # Assuming input_data and edge_index are numpy arrays
                input_tensor_2 = ov.Tensor(array=input_data_2)
                edge_index_tensor_2 = ov.Tensor(array=edge_index_2)
                # Then, use these tensors in your infer_request
                infer_request2.set_tensor(compiled_sage_layer2.input(0), input_tensor_2)
                infer_request2.set_tensor(compiled_sage_layer2.input(1), edge_index_tensor_2)
            
                start_time = time.time()
                infer_request1.start_async()
                infer_request2.start_async()
                infer_request1.wait()
                infer_request2.wait()
                input_data_2 = infer_request1.get_output_tensor(0).data
                edge_index_2 = edge_index_1
                end_time = time.time()
                timings.append(end_time - start_time)
                
                cnt += 1

        infer_request2 = compiled_sage_layer2.create_infer_request()
        # Assuming input_data and edge_index are numpy arrays
        input_tensor_2 = ov.Tensor(array=input_data_2)
        edge_index_tensor_2 = ov.Tensor(array=edge_index_2)
        # Then, use these tensors in your infer_request
        infer_request2.set_tensor(compiled_sage_layer2.input(0), input_tensor_2)
        infer_request2.set_tensor(compiled_sage_layer2.input(1), edge_index_tensor_2)
    
        start_time = time.time()
        infer_request2.start_async()
        infer_request2.wait()
        input_data_2 = infer_request1.get_output_tensor(0).data
        edge_index_2 = edge_index_1
        end_time = time.time()
        timings.append(end_time - start_time)    
        cnt += 1     
        print(f'count {cnt} :Average Inference Time: {sum(timings)*1000/cnt} ms')
        print(f'count {cnt} :  Total Inference Time: {sum(timings)*1000} ms')


    print("Finished!!")


if __name__ == "__main__":
   main()
