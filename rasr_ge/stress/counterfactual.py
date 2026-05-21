"""
Historical Counterfactual Stress Test
----------------------------------------
Applies the same shock to 4 different historical network snapshots.
The difference in ΔScore output is driven entirely by network topology
and node state — not shock magnitude.
"""

import os
import torch
import numpy as np
import yaml
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rasr_ge import RASR_GE
from training.dataset import FinancialGraphDataset
import networkx as nx


def load_snapshot(dataset, target_date_str):
    """
    Load the Data object closest to the given date from the dataset.

    Returns the Data object and its actual date string.
    """
    best_idx = None
    best_date = None

    for idx in range(len(dataset)):
        t = dataset.valid_steps[idx]
        d = dataset.dates[t].strftime('%Y-%m-%d')
        if d <= target_date_str:
            best_idx = idx
            best_date = d
        else:
            break  # dates are sorted

    if best_idx is None:
        # Fallback: use the first available
        best_idx = 0
        best_date = dataset.dates[dataset.valid_steps[0]].strftime('%Y-%m-%d')

    return dataset[best_idx], best_date


def compute_network_metrics(edge_index, n_nodes):
    """Graph density, edge count, clustering coefficient."""
    E = edge_index.shape[1]
    density = E / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0

    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    for i in range(E):
        G.add_edge(int(edge_index[0, i]), int(edge_index[1, i]))

    clustering = nx.average_clustering(G.to_undirected())
    return {
        'edge_count': E,
        'density': density,
        'clustering_coefficient': clustering
    }


def run_counterfactual(model, dataset, shock_ticker_idx, shock_magnitude,
                       snapshots, device='cpu'):
    """
    Parameters
    ----------
    model : RASR_GE
        Loaded model in eval mode.
    dataset : FinancialGraphDataset
        Full dataset (train or combined) to pull snapshots from.
    shock_ticker_idx : int
        Index of the target firm to shock.
    shock_magnitude : float
        e.g. -0.40 for a 40% drop.
    snapshots : list[dict]
        Each dict has 'label' and 'date' keys.
    device : str

    Returns
    -------
    list[dict] — one per snapshot with delta scores and metrics.
    """
    model.eval()
    results = []

    for snap in snapshots:
        target_date = snap['date']
        label = snap['label']

        if target_date is None:
            # "Current" — use last available
            data = dataset[len(dataset) - 1]
            actual_date = dataset.dates[dataset.valid_steps[-1]].strftime('%Y-%m-%d')
        else:
            data, actual_date = load_snapshot(dataset, target_date)

        n_nodes = data.x.shape[0]

        # Baseline inference
        data_dev = data.to(device)
        with torch.no_grad():
            logits_base, att_base = model(
                data_dev.x, data_dev.edge_index, data_dev.edge_attr,
                return_attention_weights=True)
            pd_base = torch.sigmoid(logits_base).cpu().numpy()

        # Shocked inference
        shocked = data.clone()
        shock_log_ret = torch.log(torch.tensor(1.0 + shock_magnitude))
        shocked.x[shock_ticker_idx, -1, 0] += shock_log_ret
        shocked_dev = shocked.to(device)

        with torch.no_grad():
            logits_shock, att_shock = model(
                shocked_dev.x, shocked_dev.edge_index, shocked_dev.edge_attr,
                return_attention_weights=True)
            pd_shocked = torch.sigmoid(logits_shock).cpu().numpy()

        delta = pd_shocked - pd_base
        top10_idx = np.argsort(delta)[::-1][:10]

        # Network metrics for this snapshot
        ei_np = data.edge_index.cpu().numpy()
        net_metrics = compute_network_metrics(ei_np, n_nodes)

        results.append({
            'label': label,
            'target_date': target_date,
            'actual_date': actual_date,
            'avg_delta_top10': float(delta[top10_idx].mean()),
            'firms_crossing_0.5': int((pd_shocked > 0.5).sum()),
            'max_delta': float(delta.max()),
            'pd_baseline': pd_base,
            'pd_shocked': pd_shocked,
            'delta': delta,
            'network_metrics': net_metrics,
        })

    return results


def compute_regime_contagion_multiplier(model, dataset, shock_magnitude,
                                        crisis_date, normal_date, device='cpu'):
    """
    Quantify how much more contagion an identical shock causes in a crisis
    network vs a calm network.

    Methodology: for every firm as shock target, compute mean ΔPD across all
    *other* firms (i.e., pure contagion, excluding the shocked node itself).
    Average across all N targets, then return crisis / normal ratio.

    Returns
    -------
    dict with keys:
        multiplier            – crisis_avg / normal_avg
        crisis_avg_contagion  – mean cross-firm ΔPD under crisis topology
        normal_avg_contagion  – mean cross-firm ΔPD under normal topology
        crisis_date_actual    – date string actually used for crisis snapshot
        normal_date_actual    – date string actually used for normal snapshot
    """
    model.eval()
    crisis_data, crisis_actual = load_snapshot(dataset, crisis_date)
    normal_data, normal_actual = load_snapshot(dataset, normal_date)

    n_nodes = crisis_data.x.shape[0]
    shock_log_ret = torch.log(torch.tensor(1.0 + shock_magnitude))

    crisis_contagions = []
    normal_contagions = []

    for target_idx in range(n_nodes):
        other = np.ones(n_nodes, dtype=bool)
        other[target_idx] = False

        # ── crisis snapshot ──────────────────────────────────────────────
        with torch.no_grad():
            cd = crisis_data.to(device)
            pd_base_c = torch.sigmoid(
                model(cd.x, cd.edge_index, cd.edge_attr,
                      return_attention_weights=True)[0]
            ).cpu().numpy()

            sc = crisis_data.clone()
            sc.x[target_idx, -1, 0] += shock_log_ret
            sc = sc.to(device)
            pd_shock_c = torch.sigmoid(
                model(sc.x, sc.edge_index, sc.edge_attr,
                      return_attention_weights=True)[0]
            ).cpu().numpy()

        crisis_contagions.append(float((pd_shock_c - pd_base_c)[other].mean()))

        # ── normal snapshot ──────────────────────────────────────────────
        with torch.no_grad():
            nd = normal_data.to(device)
            pd_base_n = torch.sigmoid(
                model(nd.x, nd.edge_index, nd.edge_attr,
                      return_attention_weights=True)[0]
            ).cpu().numpy()

            sn = normal_data.clone()
            sn.x[target_idx, -1, 0] += shock_log_ret
            sn = sn.to(device)
            pd_shock_n = torch.sigmoid(
                model(sn.x, sn.edge_index, sn.edge_attr,
                      return_attention_weights=True)[0]
            ).cpu().numpy()

        normal_contagions.append(float((pd_shock_n - pd_base_n)[other].mean()))

    crisis_avg = float(np.mean(crisis_contagions))
    normal_avg = float(np.mean(normal_contagions))
    multiplier = crisis_avg / max(normal_avg, 1e-8)

    return {
        'multiplier': multiplier,
        'crisis_avg_contagion': crisis_avg,
        'normal_avg_contagion': normal_avg,
        'crisis_date_actual': crisis_actual,
        'normal_date_actual': normal_actual,
    }


def compute_network_evolution(dataset, start_date, end_date):
    """
    Compute graph density, edge count, clustering coefficient over a date range.
    Returns lists suitable for time-series plotting.
    """
    dates_out = []
    densities = []
    edge_counts = []
    clustering_coeffs = []

    n_nodes = len(dataset.tickers)

    for idx in range(len(dataset)):
        t = dataset.valid_steps[idx]
        d = dataset.dates[t].strftime('%Y-%m-%d')
        if d < start_date or d > end_date:
            continue

        data = dataset[idx]
        ei = data.edge_index.cpu().numpy()
        metrics = compute_network_metrics(ei, n_nodes)

        dates_out.append(d)
        densities.append(metrics['density'])
        edge_counts.append(metrics['edge_count'])
        clustering_coeffs.append(metrics['clustering_coefficient'])

    return {
        'dates': dates_out,
        'density': densities,
        'edge_count': edge_counts,
        'clustering_coefficient': clustering_coeffs
    }
