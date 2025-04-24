# Model 2 HighwayEdgeConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, knn_graph
from torch_geometric.nn import global_max_pool
from ..configure import Config

class HighwayEdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels, k=5):
        super().__init__(aggr='mean')  # Mean aggregation of messages
        self.k = k
        
        # Transformation network
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels + 1, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(2 * in_channels + 1, out_channels),
            nn.Sigmoid()  
        )
        
        # Skip connection projection if dimensions don't match
        self.skip_lin = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x, edge_index, feature):
        # Propagate messages
        out = self.propagate(edge_index, x=x, feature=feature)
        # Skip connection with potential projection
        return out + self.skip_lin(x)

    def message(self, x_i, x_j, feature_i, feature_j):
        feature_diff = feature_i - feature_j
        combined = torch.cat([x_i, x_j, feature_diff], dim=-1)
        
        h = self.mlp(combined)
        g = self.gate(combined)
        return g * h + (1 - g) * x_j

class HighwayDGCNN(nn.Module):
    def __init__(self, input_dim, output_dim=2, k=5, key_feature_idx=0):
        super().__init__()
        self.k = k
        self.key_feature_idx = key_feature_idx if key_feature_idx is not None else input_dim - 1
        self.feature_dim = input_dim - 1 
        
        # Initial feature processing
        self.initial_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, Config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(Config.dropout)
        )
        
        # Graph attention layers
        self.conv1 = HighwayEdgeConv(Config.hidden_dim, Config.hidden_dim, k=k)
        self.conv2 = HighwayEdgeConv(Config.hidden_dim, Config.hidden_dim, k=k)
        
        self.classifier = nn.Sequential(
            nn.Linear(Config.hidden_dim * 2, Config.hidden_dim),  # Input matches pooled features
            nn.ReLU(),
            nn.Dropout(Config.dropout),
            nn.Linear(Config.hidden_dim, output_dim)
        )

    def forward(self, data):
        x, batch = data.x, data.batch
        key_feature = x[:, self.key_feature_idx].unsqueeze(-1)
        
        if self.key_feature_idx == x.size(1) - 1:
            features = x[:, :-1]
        else:
            features = torch.cat([x[:, :self.key_feature_idx], 
                                  x[:, self.key_feature_idx+1:]], dim=1)
        x = self.initial_mlp(features)
        
        # Create dynamic graph
        edge_index = knn_graph(x, k=self.k, batch=batch, loop=False)
        x1 = F.leaky_relu(self.conv1(x, edge_index, key_feature), negative_slope=0.2)
        x2 = F.leaky_relu(self.conv2(x1, edge_index, key_feature), negative_slope=0.2)
        
        x_combined = torch.cat([x1, x2], dim=1)
        out = global_max_pool(x_combined, batch)
        return F.log_softmax(self.classifier(out), dim=1)