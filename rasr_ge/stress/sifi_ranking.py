import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd

# Add to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rasr_ge import RASR_GE
from training.dataset import FinancialGraphDataset

def rank_sifi():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    with open(os.path.join(root_dir, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
        
    checkpoint_path = os.path.join(root_dir, "checkpoints", "best_model.pt")
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Load Model
    model = RASR_GE(
        seq_len=config['model']['seq_len'],
        input_dim=5,
        lstm_hidden=config['model']['lstm_hidden'],
        lstm_layers=config['model']['lstm_layers'],
        gat_heads=config['model']['gat_heads'],
        gat_out_dim=config['model']['gat_out_dim'],
        dropout=config['model']['dropout']
    ).to(device)
    
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Analyze the entire validation (out-of-sample) stress period
    print("Loading Validation Dataset for SIFI Analysis...")
    dataset = FinancialGraphDataset(root_dir, split='val', seq_len=config['model']['seq_len'])
    tickers = dataset.tickers
    
    num_nodes = len(tickers)
    aggregate_out_weights = np.zeros(num_nodes)
    
    print("Scoring Market Regimes across GAT Attention Graph...")
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i].to(device)
            logits, (edge_index, att_weights) = model(data.x, data.edge_index, data.edge_attr,
                                                      return_attention_weights=True)
            
            # GAT Attention outputs edge_index and attention_weights [E, heads]
            # Since gat2 has heads=1, it is [E, 1]
            e_idx = edge_index.cpu().numpy()
            att = att_weights.cpu().squeeze().numpy()
            
            # Sum up all outgoing attention for each node in this graph
            # source nodes are edge_index[0, :]
            np.add.at(aggregate_out_weights, e_idx[0], att)
            
    # Normalize
    aggregate_out_weights /= len(dataset)
    
    # Rank
    ranked_indices = np.argsort(aggregate_out_weights)[::-1]
    
    print("\n=======================================================")
    print("🏆 SYSTEMICALLY IMPORTANT FINANCIAL INSTITUTIONS (SIFIs)")
    print("Ranked by Aggregate Contagion Transmission Capability")
    print("=======================================================\n")
    
    sifi_data = []
    for rank, idx in enumerate(ranked_indices[:15]):
        ticker = tickers[idx]
        mass = aggregate_out_weights[idx]
        print(f"#{rank+1:<2} | {ticker:<15} | Spillover Score: {mass:.4f}")
        sifi_data.append({"Rank": rank+1, "Ticker": ticker, "Spillover Score": mass})
        
    df = pd.DataFrame(sifi_data)
    df.to_csv(os.path.join(root_dir, "data", "sifi_ranking.csv"), index=False)
    print("\nSaved full SIFI rankings to: data/sifi_ranking.csv")

if __name__ == "__main__":
    rank_sifi()
