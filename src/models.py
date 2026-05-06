import torch
import torch.nn as nn
from torch.nn import Linear, BatchNorm1d, LeakyReLU
from torch_geometric.nn import ChebConv, global_mean_pool

class Eeg_GNN(nn.Module):
    def __init__(
        self,
        num_node_features,
        num_classes=2,
        num_layers=2,
        hidden_dim=128,
        K=3,
        dropout=0.1
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        in_channels = num_node_features

        for _ in range(num_layers):
            self.convs.append(ChebConv(in_channels, hidden_dim, K=K))
            self.bns.append(BatchNorm1d(hidden_dim))
            in_channels = hidden_dim

        self.fc1 = Linear(hidden_dim, 128)
        self.fc2 = Linear(128, 64)
        self.fc_out = Linear(64, num_classes)

        self.act = LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch, edge_weight=None):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = bn(x)
            x = self.act(x)
            x = self.dropout(x)

        x = global_mean_pool(x, batch)

        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc_out(x)

        return x
    