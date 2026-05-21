"""
RASR-GE V2.0 — Full Data Point Extraction
=============================================
Runs all modules and exports every data point to a structured JSON file
for offline analysis if results directory: results/full_extraction.json
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
import pandas as pd
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from models.rasr_ge import RASR_GE
from training.dataset import FinancialGraphDataset
from training.evaluate import evaluate_full
from stress.shock_engine import ShockEngine
from stress.counterfactual import run_counterfactual
from risk.hmm_regime import RegimeDetector
from risk.var_engine import VaREngine
from risk.cva_engine import CVAEngine


def to_json_safe(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.strftime('%Y-%m-%d')
    elif isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    return obj


def main():
    print("=" * 60)
    print("RASR-GE V2.0 — Full Data Point Extraction")
    print("=" * 60)

    with open(os.path.join(ROOT_DIR, "config.yaml")) as f:
        config = yaml.safe_load(f)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    checkpoint = os.path.join(ROOT_DIR, "checkpoints", "best_model.pt")

    # ─────────────────────────────────────────────── Load Model
    print("\n[1/8] Loading model...")
    model = RASR_GE(
        seq_len=config['model']['seq_len'], input_dim=5,
        lstm_hidden=config['model']['lstm_hidden'],
        lstm_layers=config['model']['lstm_layers'],
        gat_heads=config['model']['gat_heads'],
        gat_out_dim=config['model']['gat_out_dim'],
        dropout=config['model']['dropout']
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    engine = ShockEngine(ROOT_DIR, checkpoint, config, device=device)
    tickers = engine.dataset.tickers
    N = len(tickers)
    print(f"  Loaded: {N} tickers, device={device}")

    output = {
        'metadata': {
            'extraction_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device': device,
            'n_tickers': N,
            'tickers': tickers,
            'config': config,
        }
    }

    # ─────────────────────────────────────────────── Baseline Risk
    print("\n[2/8] Computing baseline distress scores...")
    latest_graph = engine.get_latest_graph()
    pd_baseline, att_baseline = engine.get_baseline_risk(latest_graph)

    output['baseline_risk'] = {
        'per_firm': [{'ticker': tickers[i], 'distress_score': float(pd_baseline[i])}
                     for i in range(N)],
        'mean_distress': float(pd_baseline.mean()),
        'max_distress_firm': tickers[int(np.argmax(pd_baseline))],
        'max_distress_score': float(pd_baseline.max()),
        'firms_above_50pct': int((pd_baseline > 0.5).sum()),
    }

    # ─────────────────────────────────────────────── Shock Simulation (all firms)
    print("\n[3/8] Running shock simulations for all 49 firms...")
    shock_mag = config['stress']['shock_magnitude']
    shock_results = []
    sifi_scores = []

    for i in range(N):
        pd_shocked, att_shocked = engine.inject_shock(latest_graph, i, shock_mag)
        delta = pd_shocked - pd_baseline
        delta_copy = delta.copy()
        delta_copy[i] = 0  # exclude self
        sis = float(delta_copy.mean())
        sifi_scores.append(sis)

        # Top 5 victims for this shock
        victim_rank = np.argsort(delta_copy)[::-1][:5]
        victims = [{'ticker': tickers[v], 'delta_score': float(delta[v])} for v in victim_rank]

        shock_results.append({
            'shocked_firm': tickers[i],
            'sis_score': sis,
            'mean_delta': float(delta.mean()),
            'max_delta': float(delta.max()),
            'firms_crossing_50pct': int((pd_shocked > 0.5).sum()),
            'top_5_victims': victims,
            'pd_shocked_all': pd_shocked.tolist(),
        })

    # Sort by SIS
    sifi_ranking = sorted(range(N), key=lambda i: sifi_scores[i], reverse=True)
    output['sifi_ranking'] = [
        {'rank': r + 1, 'ticker': tickers[i], 'sis_score': sifi_scores[i]}
        for r, i in enumerate(sifi_ranking)
    ]
    output['shock_simulations'] = shock_results
    print(f"  Top 3 SIFIs: {[tickers[sifi_ranking[i]] for i in range(3)]}")

    # ─────────────────────────────────────────────── HMM Regime
    print("\n[4/8] Extracting HMM regime data...")
    hmm = RegimeDetector(config, ROOT_DIR)
    hmm.load()

    meta = torch.load(os.path.join(ROOT_DIR, 'data', 'processed', 'meta.pt'), weights_only=False)
    dates = meta['dates']

    # Per-day regime data (sampled every 5 days for file size)
    regime_timeline = []
    for t in range(0, len(dates), 5):
        d = dates[t].strftime('%Y-%m-%d')
        label, crisis_prob = hmm.get_regime(d)
        regime_timeline.append({
            'date': d,
            'regime': label,
            'crisis_posterior': float(crisis_prob),
            'correlation_density': float(hmm.density[t]) if hmm.density is not None else None,
        })

    output['hmm_regime'] = {
        'lambda': hmm.lambda_,
        'crisis_state': hmm.crisis_state,
        'normal_state': hmm.normal_state,
        'hmm_means': hmm.hmm_model.means_.flatten().tolist(),
        'hmm_variances': hmm.hmm_model.covars_.flatten().tolist(),
        'transition_matrix': hmm.hmm_model.transmat_.tolist(),
        'total_crisis_days': int((hmm.state_sequence == hmm.crisis_state).sum()),
        'total_days': len(hmm.state_sequence),
        'crisis_pct': float((hmm.state_sequence == hmm.crisis_state).mean() * 100),
        'key_dates': {
            '2020-03-23': to_json_safe(hmm.get_regime('2020-03-23')),
            '2024-12-31': to_json_safe(hmm.get_regime('2024-12-31')),
            '2017-06-01': to_json_safe(hmm.get_regime('2017-06-01')),
            '2018-09-15': to_json_safe(hmm.get_regime('2018-09-15')),
        },
        'timeline': regime_timeline,
    }

    # ─────────────────────────────────────────────── VaR
    print("\n[5/8] Computing VaR...")
    features = torch.load(os.path.join(ROOT_DIR, 'data', 'processed', 'features.pt'), weights_only=False)
    log_returns = features[:, :, 0].numpy()
    weights_eq = np.ones(N) / N
    portfolio_returns = (weights_eq[:, None] * log_returns).sum(axis=0)

    window = config['graph']['window']
    recent_returns = log_returns[:, -window:]
    corr_matrix = np.corrcoef(recent_returns)
    per_stock_vol = recent_returns.std(axis=1)

    var_engine = VaREngine(config)
    regime_label = "NORMAL"
    crisis_posteriors = None
    if hmm.posteriors is not None:
        regime_label, _ = hmm.get_regime(dates[-1].strftime('%Y-%m-%d'))
        crisis_posteriors = hmm.posteriors[-var_engine.lookback_days:, hmm.crisis_state]

    var_results = var_engine.compute_all(
        portfolio_returns, weights_eq, corr_matrix, per_stock_vol,
        crisis_posteriors, regime_label)

    var_output = {}
    for cl in config['var']['confidence_levels']:
        var_output[str(cl)] = {}
        for method in ['HS', 'Parametric', 'MC']:
            r = var_results[cl][method]
            var_output[str(cl)][method] = {
                'VaR': r['VaR'],
                'CVaR': r['CVaR'],
                'label': r['label'],
            }

    # Basel III
    bt = var_engine.basel_backtest(portfolio_returns, weights_eq, corr_matrix, per_stock_vol)

    output['var'] = {
        'results': var_output,
        'portfolio_stats': {
            'mean_daily_return': float(portfolio_returns.mean()),
            'std_daily_return': float(portfolio_returns.std()),
            'min_daily_return': float(portfolio_returns.min()),
            'max_daily_return': float(portfolio_returns.max()),
            'total_trading_days': len(portfolio_returns),
        },
        'per_stock_vol': {tickers[i]: float(per_stock_vol[i]) for i in range(N)},
        'basel_iii': {
            'exceptions': bt['exceptions'],
            'zone': bt['zone'],
            'total_days': bt['total_days'],
        }
    }

    # ─────────────────────────────────────────────── CVA (for Reliance shock)
    print("\n[6/8] Computing CVA for RELIANCE.NS shock...")
    target_ticker = 'RELIANCE.NS'
    target_idx = tickers.index(target_ticker) if target_ticker in tickers else 0

    pd_shocked, att_shocked = engine.inject_shock(latest_graph, target_idx, shock_mag)
    delta_pd = pd_shocked - pd_baseline

    ead = weights_eq * config['cva']['default_notional']
    lgd = config['cva']['lgd']
    lambda_ = hmm.lambda_

    cva_baseline = ead * lgd * pd_baseline * lambda_
    cva_updated = ead * lgd * pd_shocked * lambda_
    delta_cva = cva_updated - cva_baseline

    cva_per_firm = []
    for i in range(N):
        cva_per_firm.append({
            'ticker': tickers[i],
            'ead': float(ead[i]),
            'pd_baseline': float(pd_baseline[i]),
            'pd_updated': float(pd_shocked[i]),
            'delta_pd': float(delta_pd[i]),
            'cva_baseline': float(cva_baseline[i]),
            'cva_post_shock': float(cva_updated[i]),
            'delta_cva': float(delta_cva[i]),
        })

    output['cva'] = {
        'shocked_firm': target_ticker,
        'shock_magnitude': shock_mag,
        'lambda': lambda_,
        'lgd': lgd,
        'notional': config['cva']['default_notional'],
        'aggregate': {
            'cva_baseline_total': float(cva_baseline.sum()),
            'cva_post_shock_total': float(cva_updated.sum()),
            'delta_cva_total': float(delta_cva.sum()),
        },
        'per_firm': cva_per_firm,
    }

    # ─────────────────────────────────────────────── Counterfactual
    print("\n[7/8] Running counterfactual stress tests...")
    full_ds = FinancialGraphDataset(ROOT_DIR, split='all', seq_len=config['model']['seq_len'])
    snapshots = config['stress']['counterfactual_snapshots']

    cf_results = run_counterfactual(model, full_ds, target_idx, shock_mag, snapshots, device)

    cf_output = []
    for r in cf_results:
        cf_output.append({
            'label': r['label'],
            'target_date': r['target_date'],
            'actual_date': r['actual_date'],
            'avg_delta_top10': r['avg_delta_top10'],
            'firms_crossing_50pct': r['firms_crossing_0.5'],
            'max_delta': r['max_delta'],
            'network_metrics': r['network_metrics'],
        })
    output['counterfactual'] = cf_output

    # ─────────────────────────────────────────────── Model Evaluation
    print("\n[8/8] Running model evaluation on validation set...")
    eval_result = evaluate_full(ROOT_DIR, device=torch.device(device))

    output['model_evaluation'] = {
        'metrics': eval_result['metrics'],
        'calibration': eval_result['calibration'],
        'val_set_size': len(eval_result['y_true']),
        'positive_rate': float(eval_result['y_true'].mean()),
    }

    # ─────────────────────────────────────────────── Attention Weights (Reliance)
    print("\n  Extracting attention weights for RELIANCE.NS...")
    att_edge_index = att_baseline[0].cpu().numpy()
    att_weights_np = att_baseline[1].cpu().squeeze().numpy()

    valid_mask = (att_edge_index[0] != att_edge_index[1]) & \
                 (att_edge_index[0] < N) & (att_edge_index[1] < N)
    ei = att_edge_index[:, valid_mask]
    aw = att_weights_np[valid_mask]

    # All edges for Reliance
    reliance_idx = target_idx
    incoming = ei[1] == reliance_idx
    outgoing = ei[0] == reliance_idx

    output['attention_weights'] = {
        'target': target_ticker,
        'incoming': sorted([
            {'source': tickers[ei[0, i]], 'weight': float(aw[i])}
            for i in range(len(aw)) if incoming[i]
        ], key=lambda x: x['weight'], reverse=True),
        'outgoing': sorted([
            {'target': tickers[ei[1, i]], 'weight': float(aw[i])}
            for i in range(len(aw)) if outgoing[i]
        ], key=lambda x: x['weight'], reverse=True),
        'total_edges': int(ei.shape[1]),
        'mean_weight': float(aw.mean()),
    }

    # ─────────────────────────────────────────────── Save
    os.makedirs(os.path.join(ROOT_DIR, 'results'), exist_ok=True)
    out_path = os.path.join(ROOT_DIR, 'results', 'full_extraction.json')
    with open(out_path, 'w') as f:
        json.dump(to_json_safe(output), f, indent=2)

    # Also save per-firm summary as CSV
    csv_rows = []
    for i in range(N):
        row = {
            'ticker': tickers[i],
            'baseline_distress': pd_baseline[i],
            'sifi_rank': next(r['rank'] for r in output['sifi_ranking'] if r['ticker'] == tickers[i]),
            'sis_score': sifi_scores[i],
            'daily_vol_60d': per_stock_vol[i],
        }
        # Add CVA
        cva_row = cva_per_firm[i]
        row['cva_baseline'] = cva_row['cva_baseline']
        row['cva_post_shock'] = cva_row['cva_post_shock']
        row['delta_cva'] = cva_row['delta_cva']
        row['pd_post_reliance_shock'] = pd_shocked[i]
        row['delta_pd_reliance_shock'] = delta_pd[i]
        csv_rows.append(row)

    df = pd.DataFrame(csv_rows).sort_values('sifi_rank')
    csv_path = os.path.join(ROOT_DIR, 'results', 'firm_summary.csv')
    df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  JSON: {out_path} ({os.path.getsize(out_path) / 1024:.0f} KB)")
    print(f"  CSV:  {csv_path}")
    print(f"\nJSON sections:")
    for key in output:
        print(f"  • {key}")


if __name__ == '__main__':
    main()
