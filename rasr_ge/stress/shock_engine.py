import torch
import sys
import os
import yaml

from models.rasr_ge import RASR_GE
from training.dataset import FinancialGraphDataset

class ShockEngine:
    def __init__(self, root_dir, checkpoint_path, config, device='cpu'):
        self.device = device
        self.config = config
        self.root_dir = root_dir
        
        # Load Model architecture
        self.model = RASR_GE(
            seq_len=config['model']['seq_len'],
            input_dim=5,
            lstm_hidden=config['model']['lstm_hidden'],
            lstm_layers=config['model']['lstm_layers'],
            gat_heads=config['model']['gat_heads'],
            gat_out_dim=config['model']['gat_out_dim'],
            dropout=config['model']['dropout']
        ).to(device)
        
        # Load pre-trained weights safely
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Instantiate dataset to pull the "current day" graph
        self.dataset = FinancialGraphDataset(root_dir, split='val', seq_len=config['model']['seq_len'])
        
    def get_latest_graph(self):
        # We simulate the dashboard using the most recent graph state in the sequence
        return self.dataset[len(self.dataset) - 1]
        
    def get_baseline_risk(self, data):
        data = data.to(self.device)
        with torch.no_grad():
            logits, att_weights = self.model(data.x, data.edge_index, data.edge_attr,
                                             return_attention_weights=True)
            pd_baseline = torch.sigmoid(logits).cpu().numpy()
        return pd_baseline, att_weights
        
    def inject_shock(self, data, shocked_node_idx, shock_magnitude):
        """
        data: PyG Data object
        shocked_node_idx: Index of the firm to shock (e.g., Target Firm)
        shock_magnitude: e.g., -0.40 for a 40% drop
        """
        shocked_data = data.clone()
        
        # Applying empirical shock: log_return = ln(1 + shock)
        shock_log_ret = torch.log(torch.tensor(1.0 + shock_magnitude))
        
        # Apply the absolute drop entirely on the very last day of the historical lookback for that specific node
        shocked_data.x[shocked_node_idx, -1, 0] += shock_log_ret
        
        shocked_data = shocked_data.to(self.device)
        with torch.no_grad():
            logits, att_weights = self.model(shocked_data.x, shocked_data.edge_index,
                                             shocked_data.edge_attr,
                                             return_attention_weights=True)
            pd_shocked = torch.sigmoid(logits).cpu().numpy()
            
        return pd_shocked, att_weights
