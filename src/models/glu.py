# Model 1 Gated Linear Unit (GLU) for edge features 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, knn_graph, global_max_pool, global_mean_pool
from ..configure import Config
from lightweightDGCNN import TempEdgeConv

class GatedLinearUnit(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.gate = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x) * torch.sigmoid(self.gate(x))

class GLUedgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels, k=5):
        super().__init__(aggr='max')
        self.k = k
        self.register_buffer('dummy_edge_index', torch.empty(2, 0, dtype=torch.long))
        
        self.gated_mlp = nn.Sequential(
            GatedLinearUnit(2 * in_channels + 1, out_channels),
            nn.Linear(out_channels, out_channels)
        )

    def message(self, x_i, x_j, feature_i, feature_j):
        feature_diff = feature_j - feature_i
        edge_features = torch.cat([x_i, x_j, feature_diff], dim=-1)
        return self.gated_mlp(edge_features)

    def forward(self, x, batch, feature):
        edge_index = knn_graph(x, k=self.k, batch=batch, loop=False, flow=self.flow, cosine=False)
        return self.propagate(edge_index, x=x, feature=feature) + x

class GlueEdgeDGCNN(nn.Module):
    def __init__(self, input_dim, output_dim=2, key_feature_idx=0):
        super().__init__()
        self.key_feature_idx = key_feature_idx if key_feature_idx is not None else input_dim - 1
        self.feature_dim = input_dim - 1

        self.time_encoder = nn.Linear(1, Config.hidden_dim // 4)

        self.initial_gated = nn.Sequential(
            GatedLinearUnit(self.feature_dim + Config.hidden_dim // 4, Config.hidden_dim),
            nn.Dropout(Config.dropout)
        )

        self.conv1 = TempEdgeConv(Config.hidden_dim, Config.hidden_dim)
        self.conv2 = TempEdgeConv(Config.hidden_dim, Config.hidden_dim)

        self.final_gated = nn.Sequential(
            GatedLinearUnit(Config.hidden_dim * 2 * 2, Config.hidden_dim),
            nn.Linear(Config.hidden_dim, output_dim)
        )

    def forward(self, data):
        x, batch = data.x, data.batch
        key_feature = x[:, self.key_feature_idx].unsqueeze(-1)

        if self.key_feature_idx == x.size(1) - 1:
            features = x[:, :-1]
        else:
            features = torch.cat([x[:, :self.key_feature_idx], x[:, self.key_feature_idx + 1:]], dim=1)

        key_feature_emb = self.time_encoder(key_feature)  
        x = self.initial_gated(torch.cat([features, key_feature_emb], dim=-1))

        x1 = F.relu(self.conv1(x, batch, key_feature))
        x2 = F.relu(self.conv2(x1, batch, key_feature))

        combined = torch.cat([x1, x2], dim=1)
        max_pool = global_max_pool(combined, batch)
        mean_pool = global_mean_pool(combined, batch)
        out = torch.cat([max_pool, mean_pool], dim=1)

        return F.log_softmax(self.final_gated(out), dim=1)