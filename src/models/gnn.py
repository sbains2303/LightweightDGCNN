# 3. Fixed BasicGNN implementation
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_max_pool
from ..configure import Config

class BasicGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j):
        # x_j has shape [E, out_channels]
        return self.lin(x_j)

class BasicGNN(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super().__init__()
        self.conv1 = BasicGNNLayer(input_dim, Config.hidden_dim)
        self.conv2 = BasicGNNLayer(Config.hidden_dim, Config.hidden_dim)
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