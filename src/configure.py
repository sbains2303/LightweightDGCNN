import torch

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