"""
Extended Evaluation Suite
----------------------------
AUROC, PRAUC, Brier Score, KS Statistic, and reliability diagram data
on the validation set.
"""

import os
import sys
import yaml
import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    f1_score, precision_score, recall_score
)
from torch_geometric.loader import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rasr_ge import RASR_GE
from training.dataset import FinancialGraphDataset


def ks_statistic(y_true, y_prob):
    """Kolmogorov–Smirnov statistic between positive and negative class score distributions."""
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    from scipy.stats import ks_2samp
    stat, _ = ks_2samp(pos, neg)
    return float(stat)


def reliability_diagram_data(y_true, y_prob, n_bins=10):
    """
    10-bin calibration data: for each bin, compute mean predicted score
    and actual positive rate.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    result = []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        mean_pred = float(y_prob[mask].mean())
        actual_rate = float(y_true[mask].mean())
        count = int(mask.sum())
        result.append({
            'bin_start': float(bins[i]),
            'bin_end': float(bins[i + 1]),
            'mean_predicted': mean_pred,
            'actual_positive_rate': actual_rate,
            'count': count
        })
    return result


def evaluate_full(root_dir=None, device=None):
    """Run full evaluation on the validation set and return all metrics."""
    if root_dir is None:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with open(os.path.join(root_dir, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    checkpoint_path = os.path.join(root_dir, "checkpoints", "best_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("No checkpoint found. Train the model first.")

    if device is None:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

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

    val_dataset = FinancialGraphDataset(root_dir, split='val', seq_len=config['model']['seq_len'])
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu())
            all_targets.append(batch.y.cpu())

    y_true = torch.cat(all_targets).numpy()
    y_prob = torch.cat(all_preds).numpy()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        'auroc': float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5,
        'prauc': float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        'brier_score': float(brier_score_loss(y_true, y_prob)),
        'ks_statistic': ks_statistic(y_true, y_prob),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
    }

    calibration = reliability_diagram_data(y_true, y_prob)

    return {
        'metrics': metrics,
        'calibration': calibration,
        'y_true': y_true,
        'y_prob': y_prob,
    }


if __name__ == "__main__":
    result = evaluate_full()
    m = result['metrics']
    print("=" * 50)
    print("RASR-GE Validation Metrics")
    print("=" * 50)
    print(f"  AUROC:        {m['auroc']:.4f}")
    print(f"  PRAUC:        {m['prauc']:.4f}")
    print(f"  Brier Score:  {m['brier_score']:.4f}")
    print(f"  KS Statistic: {m['ks_statistic']:.4f}")
    print(f"  F1:           {m['f1']:.4f}")
    print(f"  Precision:    {m['precision']:.4f}")
    print(f"  Recall:       {m['recall']:.4f}")
    print("\nReliability Diagram:")
    for b in result['calibration']:
        print(f"  [{b['bin_start']:.1f}–{b['bin_end']:.1f}] "
              f"pred={b['mean_predicted']:.3f}  actual={b['actual_positive_rate']:.3f}  "
              f"n={b['count']}")
