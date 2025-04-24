import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.utils import remove_self_loops
from sklearn.preprocessing import StandardScaler
from configure import Config


class UAVDataset(Dataset):
    def __init__(self, csv_path, scaler=None, train=True, test_data = None):
        super(UAVDataset, self).__init__()
        self.data = pd.read_csv(csv_path)
        self.train = train
        self.scaler = scaler
        self.data['label'] = self.data['label'].map({'genuine': 0, 'spoofed': 1})
        self.test_data = test_data
        
        self.data['route_id'] = self.data['route_number'].astype(str) + '_' + self.data['label'].astype(str)
        self.unique_routes = self.data['route_id'].unique()
        
        self.features = [
            'imu_magnetic_field_frd_right_gauss',
            'imu_magnetic_field_frd_forward_gauss',
            'imu_magnetic_field_frd_down_gauss',
            'ned_down_m_s',
            'imu_acceleration_frd_forward_m_s2',
            'imu_angular_velocity_frd_down_rad_s',
            'imu_acceleration_frd_right_m_s2'
        ] 
        
        missing_features = [f for f in self.features if f not in self.data.columns]
        if missing_features:
            raise ValueError(f"Missing features in CSV: {missing_features}")
        
        if train and scaler is None:
            self.scaler = StandardScaler()  
            self.scaler.fit(self.data[self.features].values, self.features)
    
    def len(self):
        return len(self.unique_routes)  
    
    def get(self, idx):
        try:
            route_id = self.unique_routes[idx]
            route_num, label = route_id.split('_')
            label = int(label)
            
            route_data = self.data[
                (self.data['route_number'] == int(route_num)) & 
                (self.data['label'] == label)
            ]
            
            features = route_data[self.features].values.astype(np.float32)
            
            x = torch.tensor(features, dtype=torch.float)
            num_nodes = len(route_data)
            
            temporal_edges = []
            for i in range(num_nodes):
                for j in range(max(0, i-Config.temporal_window), min(num_nodes, i+Config.temporal_window+1)):
                    if i != j:
                        temporal_edges.append([i, j])
            
            knn_edges = []
            if num_nodes > Config.k:
                edge_index = knn_graph(x, k=Config.k, loop=False)
                knn_edges = edge_index.t().tolist()
            
            all_edges = temporal_edges + knn_edges
            edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
            edge_index = remove_self_loops(edge_index)[0]
            
            return Data(
                x=x,
                edge_index=edge_index,
                y=torch.tensor([label], dtype=torch.long),  
                route_id=torch.tensor([int(route_num)], dtype=torch.long),  
                route_label=torch.tensor([label], dtype=torch.long)  
            )
            
        except Exception as e:
            print(f"Error processing route {route_id}: {str(e)}")
            return None