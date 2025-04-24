# 2. GCN (Graph Convolutional Network)
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool
from ..configure import Config

class GCNModel(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super().__init__()
        self.conv1 = GCNConv(input_dim, Config.hidden_dim)
        self.conv2 = GCNConv(Config.hidden_dim, Config.hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(Config.hidden_dim, Config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(Config.dropout),
            nn.Linear(Config.hidden_dim, output_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_max_pool(x, batch)
        return F.log_softmax(self.fc(x), dim=1)