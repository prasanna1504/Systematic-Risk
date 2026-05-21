import os
import torch
from torch_geometric.data import Dataset, Data
import yaml

class FinancialGraphDataset(Dataset):
    def __init__(self, root, split='train', seq_len=30, transform=None, pre_transform=None):
        self.seq_len = seq_len
        self.split = split
        
        # Load config
        with open(os.path.join(root, "config.yaml"), "r") as f:
            self.config = yaml.safe_load(f)
            
        train_end = self.config['data']['train_end']
        val_start = self.config['data']['val_start']
        
        self.features = torch.load(os.path.join(root, 'data', 'processed', 'features.pt'), weights_only=False)
        self.labels = torch.load(os.path.join(root, 'data', 'labels', 'labels.pt'), weights_only=False)
        
        meta = torch.load(os.path.join(root, 'data', 'processed', 'meta.pt'), weights_only=False)
        self.dates = meta['dates']
        self.tickers = meta['valid_tickers']
        
        self.valid_steps = []
        
        # seq_len determines earliest we can start. Graph starts at day 60 anyway.
        start_t = max(60, self.seq_len)
        T = self.features.shape[1]
        
        for t in range(start_t, T - 5):
            date_str = self.dates[t].strftime('%Y-%m-%d')
            if self.split == 'train' and date_str <= train_end:
                self.valid_steps.append(t)
            elif self.split == 'val' and date_str >= val_start:
                self.valid_steps.append(t)
            elif self.split == 'all':
                self.valid_steps.append(t)
                
        super().__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def len(self):
        return len(self.valid_steps)

    def get(self, idx):
        t = self.valid_steps[idx]
        
        # Node features X: shape (N, seq_len, 5)
        # We cast to float32
        x = self.features[:, t-self.seq_len:t, :].float()
        
        date_str = self.dates[t].strftime('%Y-%m-%d')
        graph_dict = torch.load(os.path.join(self.root, 'data', 'graphs', f'adj_{date_str}.pt'), weights_only=False)
        edge_index = graph_dict['edge_index']
        edge_weight = graph_dict['edge_weight']
        
        # Labels for day t
        y = self.labels[:, t].float()
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y, t_idx=t)
