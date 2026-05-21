"""
GNNExplainer — Prediction-Level Attribution
----------------------------------------------
For a specific firm i on a specific day t, identifies:
  - Which neighbor firms drove the distress prediction (edge mask)
  - Which input features drove the prediction (node feature mask)

This gives regulatory-grade explainability:
  "HDFC Bank's distress on March 20, 2020 was driven 62% by its link
   to ICICI and 38% by its own realized volatility."
"""

import os
import sys
import yaml
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.explain import Explainer, GNNExplainer
from models.rasr_ge import RASR_GE
from training.dataset import FinancialGraphDataset

FEATURE_NAMES = ['log_return', 'realized_vol', 'normalized_volume', 'high_low_range', 'close_to_sma20']


class RASRExplainer:
    """
    Wraps PyG's GNNExplainer to produce per-prediction attribution
    for the RASR-GE model.
    """

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()

        # PyG Explainer wraps the model and algorithm together
        self.explainer = Explainer(
            model=_ExplainerWrapper(model),
            algorithm=GNNExplainer(epochs=200, lr=0.01),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='binary_classification',
                task_level='node',
                return_type='raw',
            ),
        )

    def explain_node(self, data, node_idx, tickers):
        """
        Explain the prediction for a specific node in a specific graph snapshot.

        Parameters
        ----------
        data : torch_geometric.data.Data
            A single graph snapshot (from the dataset).
        node_idx : int
            The index of the node to explain.
        tickers : list[str]
            Ticker names for all nodes.

        Returns
        -------
        dict with:
          - 'target_ticker': str
          - 'distress_score': float
          - 'feature_importance': list[dict] with feature name + contribution
          - 'counterparty_importance': list[dict] with neighbor + contribution
        """
        data = data.to(self.device)

        # Flatten x for the wrapper: (N, seq_len, 5) -> (N, seq_len*5)
        x_flat = data.x.reshape(data.x.shape[0], -1)

        edge_attr = data.edge_attr
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        explanation = self.explainer(
            x=x_flat,
            edge_index=data.edge_index,
            edge_attr=edge_attr,
            index=node_idx,
        )

        # Get the actual distress score
        with torch.no_grad():
            logits = self.model(data.x, data.edge_index, data.edge_attr)
            score = torch.sigmoid(logits[node_idx]).item()

        # ── Feature importance ──
        # node_mask shape: (N, seq_len*5) — we care about node_idx's row
        node_mask = explanation.node_mask
        if node_mask is not None:
            # Reshape to (N, seq_len, 5) and average over time to get per-feature importance
            seq_len = data.x.shape[1]
            mask_reshaped = node_mask[node_idx].reshape(seq_len, 5)
            # Average across time steps to get per-feature importance
            feat_importance = mask_reshaped.mean(dim=0).cpu().numpy()
            feat_total = feat_importance.sum()
            if feat_total > 0:
                feat_pct = feat_importance / feat_total
            else:
                feat_pct = np.ones(5) / 5.0

            feature_results = []
            for f_idx in np.argsort(feat_pct)[::-1]:
                feature_results.append({
                    'feature': FEATURE_NAMES[f_idx],
                    'importance': float(feat_importance[f_idx]),
                    'contribution_pct': float(feat_pct[f_idx] * 100),
                })
        else:
            feature_results = []

        # ── Counterparty importance ──
        edge_mask = explanation.edge_mask
        edge_index = data.edge_index.cpu().numpy()
        n = len(tickers)

        counterparty_results = []
        if edge_mask is not None:
            edge_mask_np = edge_mask.cpu().numpy()
            # Find all edges pointing TO node_idx (incoming influence)
            for e in range(edge_index.shape[1]):
                src, dst = int(edge_index[0, e]), int(edge_index[1, e])
                if dst == node_idx and src != node_idx and src < n:
                    counterparty_results.append({
                        'counterparty': tickers[src],
                        'edge_importance': float(edge_mask_np[e]),
                    })

            # Sort and compute percentages
            counterparty_results.sort(key=lambda x: x['edge_importance'], reverse=True)
            total_edge = sum(c['edge_importance'] for c in counterparty_results)
            if total_edge > 0:
                for c in counterparty_results:
                    c['contribution_pct'] = float(c['edge_importance'] / total_edge * 100)
            else:
                for c in counterparty_results:
                    c['contribution_pct'] = 0.0

        return {
            'target_ticker': tickers[node_idx],
            'distress_score': score,
            'feature_importance': feature_results,
            'counterparty_importance': counterparty_results[:10],  # top 10
        }


class _ExplainerWrapper(torch.nn.Module):
    """
    Thin wrapper around RASR_GE that accepts pre-flattened x
    so that the Explainer can compute node attribute masks.
    The Explainer passes (x_flat, edge_index, edge_attr) and
    we reshape x back to (N, seq_len, 5) before calling the real model.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        # Infer seq_len from the LSTM
        self.seq_len = 30  # from config
        self.input_dim = 5

    def forward(self, x, edge_index, edge_attr=None):
        # x arrives as (N, seq_len*5) — reshape
        N = x.shape[0]
        x_3d = x.reshape(N, self.seq_len, self.input_dim)
        logits = self.model(x_3d, edge_index, edge_attr)
        return logits


# ── CLI for quick testing ──
if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with open(os.path.join(root_dir, "config.yaml")) as f:
        config = yaml.safe_load(f)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    checkpoint = os.path.join(root_dir, "checkpoints", "best_model.pt")

    model = RASR_GE(
        seq_len=config['model']['seq_len'], input_dim=5,
        lstm_hidden=config['model']['lstm_hidden'],
        lstm_layers=config['model']['lstm_layers'],
        gat_heads=config['model']['gat_heads'],
        gat_out_dim=config['model']['gat_out_dim'],
        dropout=config['model']['dropout']
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    dataset = FinancialGraphDataset(root_dir, split='val', seq_len=config['model']['seq_len'])
    tickers = dataset.tickers

    explainer = RASRExplainer(model, device)

    # Explain the top-risk firm on the latest day
    latest = dataset[len(dataset) - 1]
    with torch.no_grad():
        logits = model(latest.to(device).x, latest.to(device).edge_index, latest.to(device).edge_attr)
        scores = torch.sigmoid(logits).cpu().numpy()

    top_firm = int(np.argmax(scores))
    print(f"\nExplaining: {tickers[top_firm]} (DistressScore = {scores[top_firm]:.3f})")
    print("=" * 60)

    result = explainer.explain_node(latest, top_firm, tickers)

    print("\n📊 Feature Attribution:")
    for f in result['feature_importance']:
        bar = "█" * int(f['contribution_pct'] / 2)
        print(f"  {f['feature']:<22} {f['contribution_pct']:5.1f}%  {bar}")

    print("\n🔗 Counterparty Attribution:")
    for c in result['counterparty_importance'][:5]:
        bar = "█" * int(c['contribution_pct'] / 2)
        print(f"  {c['counterparty']:<15} {c['contribution_pct']:5.1f}%  {bar}")
