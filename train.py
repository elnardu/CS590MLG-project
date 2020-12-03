import numpy as np
import torch
import models
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

epochs = 10
hidden_dim = 32
num_classes = 7
num_layers = 2
learning_rate = 0.005
weight_decay = 5e-3
model_type = "GCN" # only GCN or GraphSage

def train(dataset):
    model = models.GNNStack(dataset.num_node_features, hidden_dim, num_classes, model_type, num_layers)
    filter_fn = filter(lambda p : p.requires_grad,  model.parameters())
    opt = optim.Adam(filter_fn, lr=learning_rate, weight_decay=weight_decay)
    writer = SummaryWriter()
    for epoch in range(epochs):
        model.train()
        for batch in dataset:
            opt.zero_grad()
            prob = model(batch)
            pred = prob.argmax(axis=1)
#             label = batch.y
            label = np.zeros(len(batch.y))
            label[batch.attacked_nodes] = 1
            label = torch.tensor(label, dtype=torch.long)
#             print((pred==label).double().mean())
            loss = model.loss(prob, label)
            print(loss.item())
            loss.backward()
            opt.step()
        true_pos = (pred[batch.attacked_nodes] == 1).sum().double()
        precision = true_pos/pred.sum()
        recall = true_pos/label.sum()
        print(2*precision*recall/(precision+recall))
#             writer.add_scalar("Precision", 1/pred.sum(), 
#         total_loss /= len(dataset)
    
if __name__ == '__main__':
    from pathlib import Path
    from mlg import NETTACKDataset
    from torch_geometric.data import DataLoader
    
    nc = NETTACKDataset(root=Path('data'), combine_n=1000)
    train(nc)
