import os.path as osp
import time
import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.datasets import Flickr
from torch_geometric.datasets import Reddit2
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv


# load dataset
dataset = Reddit2(root='/tmp/Reddit2')
data = dataset[0]


# Ensure labels are on the correct device and not None
data.y = data.y if data.y is not None else torch.zeros((data.num_nodes,), dtype=torch.long)

kwargs = {'batch_size': 1024, 'num_workers': 6, 'persistent_workers': True}
train_loader = NeighborLoader(data, input_nodes=data.train_mask,
                              num_neighbors=[25, 10], shuffle=True, **kwargs)

# Measure the time it takes to initialize the subgraph loader
start_time = time.time()
subgraph_loader = NeighborLoader(data, input_nodes=None,
                                 num_neighbors=[-1], shuffle=False, **kwargs)
loader_init_time = time.time() - start_time
print(f"Subgraph Loader Initialization Time: {loader_init_time:.4f} seconds")

# # No need to maintain these features during evaluation:
# del subgraph_loader.data.x, subgraph_loader.data.y
# Add global node index information.
subgraph_loader.data.num_nodes = data.num_nodes
subgraph_loader.data.n_id = torch.arange(data.num_nodes)

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

model = SAGE(dataset.num_features, 256, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(epoch):
    model.train()

    pbar = tqdm(total=int(len(train_loader.dataset)))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = total_examples = 0
    for batch in train_loader:
        optimizer.zero_grad()
        y = batch.y[:batch.batch_size]
        y_hat = model(batch.x, batch.edge_index)[:batch.batch_size]
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch.batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch.batch_size
        pbar.update(batch.batch_size)
    pbar.close()

    return total_loss / total_examples, total_correct / total_examples

@torch.no_grad()
def test():
    model.eval()
    y_hat = model.inference(data.x, subgraph_loader).argmax(dim=-1)
    y = data.y

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((y_hat[mask] == y[mask]).sum()) / int(mask.sum()))
    return accs

times = []
for epoch in range(1, 2):
    start = time.time()
    loss, acc = train(epoch)
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
    train_acc, val_acc, test_acc = test()
    # print(f'Epoch: {epoch:02d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
    #       f'Test: {test_acc:.4f}')
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")

# save weight
torch.save(model.state_dict(), 'Reddit2_SAGE_weights.pth')
print("model saved!")