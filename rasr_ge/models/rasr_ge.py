import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, x):
        # x: (sum_N, seq_len, input_dim)
        out, (hn, cn) = self.lstm(x)
        # hn: (num_layers, sum_N, hidden_dim) — take last layer
        return hn[-1]  # (sum_N, hidden_dim)


class RASR_GE(nn.Module):
    def __init__(self, seq_len=30, input_dim=5, lstm_hidden=64, lstm_layers=2,
                 gat_heads=4, gat_out_dim=32, dropout=0.3):
        super().__init__()

        self.temporal_encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout
        )

        # edge_dim=1: each edge carries one scalar feature (Pearson correlation weight)
        self.gat1 = GATConv(
            in_channels=lstm_hidden,
            out_channels=gat_out_dim,
            heads=gat_heads,
            concat=True,
            dropout=dropout,
            edge_dim=1
        )
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

        gat1_out_dim = gat_heads * gat_out_dim
        self.gat2 = GATConv(
            in_channels=gat1_out_dim,
            out_channels=gat_out_dim,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=1
        )

        self.predictor = nn.Linear(gat_out_dim, 1)

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=False):
        # Normalise edge_attr to shape (E, 1) so GATConv receives a proper 2D tensor.
        # edge_attr arrives as (E,) from the DataLoader; unsqueeze makes it (E, 1).
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        # 1. Temporal encoder — runs independently per node
        h = self.temporal_encoder(x)

        # 2. Spatial encoder — both GAT layers now use correlation weights as edge features
        z1 = self.gat1(h, edge_index, edge_attr)
        z1 = self.elu(z1)
        z1 = self.dropout(z1)

        if return_attention_weights:
            z2, att_weights = self.gat2(z1, edge_index, edge_attr,
                                        return_attention_weights=True)
            logits = self.predictor(z2).squeeze(-1)  # (N,)
            return logits, att_weights
        else:
            z2 = self.gat2(z1, edge_index, edge_attr)
            logits = self.predictor(z2).squeeze(-1)
            return logits
