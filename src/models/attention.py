# AttentionEdgeConv 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, knn_graph, global_max_pool
from ..configure import Config, create_spatiotemporal_edges

class AttentionEdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels, k=5):
        super().__init__(aggr='add')
        self.k = k
        
        # Attention weights computation
        self.att = nn.Sequential(
            nn.Linear(2 * in_channels + 1, out_channels),
            nn.Tanh(),
            nn.Linear(out_channels, 1, bias=False)
        )
        
        # Feature transformation
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels + 1, out_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(out_channels, out_channels),
            nn.Dropout(Config.dropout)
        )
        
        # Skip connection handling
        self.skip_lin = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x, edge_index, key_feature, batch):
        edge_index = create_spatiotemporal_edges(x, batch, self.k, Config.temporal_window)
        out = self.propagate(edge_index, x=x, key_feature=key_feature)
        return out + self.skip_lin(x)

    def message(self, x_i, x_j, key_feature_i, key_feature_j):
        feature_diff = key_feature_i - key_feature_j
        combined = torch.cat([x_i, x_j, feature_diff], dim=-1)
        
        # Compute attention weights
        att_weight = F.softmax(self.att(combined), dim=0)
        
        # Transform features
        transformed = self.mlp(combined)
        
        return att_weight * transformed

class AttentionDGCNN(nn.Module):
    def __init__(self, input_dim, output_dim=2, k=5, key_feature_idx=0):
        super().__init__()
        self.k = k
        self.key_feature_idx = key_feature_idx if key_feature_idx is not None else input_dim - 1
        self.feature_dim = input_dim - 1  # Total features minus key feature
        
        # Initial feature processing
        self.initial_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, Config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(Config.dropout)
        )
        
        # Graph attention layers
        self.conv1 = AttentionEdgeConv(Config.hidden_dim, Config.hidden_dim, k=k)
        self.conv2 = AttentionEdgeConv(Config.hidden_dim, Config.hidden_dim, k=k)
        
        # Final classifier - adjusted dimensions
        self.classifier = nn.Sequential(
            nn.Linear(Config.hidden_dim * 2, Config.hidden_dim),  # Input matches pooled features
            nn.ReLU(),
            nn.Dropout(Config.dropout),
            nn.Linear(Config.hidden_dim, output_dim)
        )

    def forward(self, data):
        x, batch = data.x, data.batch
        
        # Extract the discriminative feature
        key_feature = x[:, self.key_feature_idx].unsqueeze(-1)
        
        # Create a new feature tensor without the key feature
        if self.key_feature_idx == x.size(1) - 1:
            features = x[:, :-1]
        else:
            # Concatenate all columns except the key feature
            features = torch.cat([x[:, :self.key_feature_idx], 
                                  x[:, self.key_feature_idx+1:]], dim=1)
        
        # Initial feature transformation
        x = self.initial_mlp(features)
        
        # Apply attention convolutions
        x1 = F.leaky_relu(self.conv1(x, batch, key_feature), negative_slope=0.2)
        x2 = F.leaky_relu(self.conv2(x1, batch, key_feature), negative_slope=0.2)
        
        # Combine features from both layers
        x_combined = torch.cat([x1, x2], dim=1)
        
        # Global pooling
        out = global_max_pool(x_combined, batch)
        
        # Final classification
        return F.log_softmax(self.classifier(out), dim=1)
