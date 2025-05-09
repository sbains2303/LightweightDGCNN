# Final model using TSGConv (Temporal Graph Convolution) - acc_tr=  93, acc_te=  93, lat = 1917.08 ms
from ..configure import Config, create_spatiotemporal_edges
from torch_geometric.nn import MessagePassing, knn_graph, global_max_pool, global_mean_pool
import torch
import torch.nn as nn
import torch.nn.functional as F

class TempEdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels, k=5):
        super().__init__(aggr='max')  
        self.k = k
        # Pre-allocate memory for edge_index
        self.register_buffer('dummy_edge_index', torch.empty(2, 0, dtype=torch.long))
        
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels + 1, out_channels),
            nn.LeakyReLU(0.1, inplace=True)  ,  
            nn.Linear(out_channels, out_channels)
        )
    
    def forward(self, x, batch, key_feature):
        edge_index = create_spatiotemporal_edges(x, batch, self.k, Config.temporal_window)
        return self.propagate(edge_index, x=x, key_feature=key_feature) + x
        
    def message(self, x_i, x_j, key_feature_i, key_feature_j):
        # Compute difference in features and discriminative key
        h_diff = x_j - x_i
        key_diff = key_feature_j - key_feature_i

        # Concatenate [h_j - h_i, h_i, k_j - k_i]
        h = torch.cat([h_diff, x_i, key_diff], dim=-1)
        return self.mlp(h)



class LightweightDGCNN(nn.Module):
    def __init__(self, input_dim, output_dim=2, key_feature_idx=0):
        super().__init__()
        self.key_feature_idx = key_feature_idx if key_feature_idx is not None else input_dim - 1
        self.feature_dim = input_dim - 1  # Total features minus the one for special encoding
        
        # Encode the selected discriminative feature
        self.feature_encoder = nn.Linear(1, Config.hidden_dim // 4)  
        
        self.initial_mlp = nn.Sequential(
            nn.Linear(self.feature_dim + Config.hidden_dim // 4, Config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(Config.dropout)
        )
        
        self.conv1 = TempEdgeConv(Config.hidden_dim, Config.hidden_dim)
        self.conv2 = TempEdgeConv(Config.hidden_dim, Config.hidden_dim)
        self.final_mlp = nn.Sequential(
            nn.Linear(Config.hidden_dim * 2 * 2, Config.hidden_dim),  
            nn.ReLU(),
            nn.Linear(Config.hidden_dim, output_dim)
        )
    
    def forward(self, data):
        x, batch = data.x, data.batch
        
        # Extract the discriminative feature
        key_feature = x[:, self.key_feature_idx].unsqueeze(-1)
        
        if self.key_feature_idx == x.size(1) - 1:
            features = x[:, :-1]
        else:
            features = torch.cat([x[:, :self.key_feature_idx], 
                                  x[:, self.key_feature_idx+1:]], dim=1)
        
        # Encode key feature and fuse with other features
        key_embedding = self.feature_encoder(key_feature)  
        x = self.initial_mlp(torch.cat([features, key_embedding], dim=-1))
        
        x1 = F.relu(self.conv1(x, batch, key_feature))
        x2 = F.relu(self.conv2(x1, batch, key_feature))
        
        max_pool = global_max_pool(torch.cat([x1, x2], dim=1), batch)
        mean_pool = global_mean_pool(torch.cat([x1, x2], dim=1), batch)
        out = torch.cat([max_pool, mean_pool], dim=1)
        
        return F.log_softmax(self.final_mlp(out), dim=1)
