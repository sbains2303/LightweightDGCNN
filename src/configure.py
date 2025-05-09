import torch
from torch_cluster import knn_graph

class Config:
    seed = 42
    k = 5
    temporal_window = 5
    batch_size = 24
    epochs = 50
    lr = 0.001
    weight_decay = 1e-3
    dropout = 0.4
    hidden_dim = 46
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Future NVIDA GPU use

def create_spatiotemporal_edges(x, batch, k, temporal_window):
    num_nodes = x.size(0)
    batch_sizes = torch.bincount(batch)
    temporal_edges = []
    start_idx = 0
    
    for size in batch_sizes:
        indices = torch.arange(start_idx, start_idx + size)
        
        # Create all possible pairs within temporal window
        source = indices.unsqueeze(1).repeat(1, 2*temporal_window+1)
        target = torch.stack([torch.arange(i-temporal_window, i+temporal_window+1) 
                            for i in range(start_idx, start_idx + size)])
        
        # Mask invalid indices and self-loops
        mask = (target >= start_idx) & (target < start_idx + size) & (source != target)
        
        # Filter and add valid edges
        valid_edges = torch.stack([
            source[mask],
            target[mask]
        ]).t()
        
        temporal_edges.append(valid_edges)
        start_idx += size
    
    temporal_edges = torch.cat(temporal_edges, dim=0).t()
    knn_edge_index = knn_graph(x, k=k, batch=batch, loop=False)
    edge_index = torch.cat([knn_edge_index, temporal_edges], dim=1)
    
    return edge_index
