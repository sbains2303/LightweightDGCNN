# 4. CNN (Convolutional Neural Network) adapted for graph-like data
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool
from ..configure import Config

class CNNModel(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super().__init__()
        # We'll use 1D convolutions since we're dealing with feature vectors
        self.conv1 = nn.Conv1d(1, Config.hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(Config.hidden_dim, Config.hidden_dim, kernel_size=3, padding=1)
        self.fc = nn.Sequential(
            nn.Linear(Config.hidden_dim, Config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(Config.dropout),
            nn.Linear(Config.hidden_dim, output_dim)
        )
        
    def forward(self, data):
        x, batch = data.x, data.batch
        
        # Reshape x to (batch_size * num_nodes, 1, feature_dim)
        x = x.unsqueeze(1)
        
        # Apply 1D convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Max pooling over the feature dimension
        x = x.max(dim=2)[0]
        
        # Now we have (batch_size * num_nodes, hidden_dim)
        # We need to aggregate over nodes in each graph
        x = global_max_pool(x, batch)
        
        return F.log_softmax(self.fc(x), dim=1)