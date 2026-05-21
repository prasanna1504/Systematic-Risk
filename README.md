# RASR-GE: Regime-Aware Systemic Risk Engine

### *Can a graph neural network map contagion across Nifty 50 before a crash happens?*

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://prasanna1504-systematic-risk-rasr-gedashboardapp-eme0zz.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Domain](https://img.shields.io/badge/Domain-Systemic%20Risk%20%7C%20Basel%20IV-0a66c2?style=for-the-badge)
![Data](https://img.shields.io/badge/Universe-Nifty%2050%20%7C%202015--2025-2ea44f?style=for-the-badge)

---

## At a Glance

> "An identical −40% shock to one stock spreads **26× more contagion** during a crisis network than a calm one."

RASR-GE is an end-to-end systemic risk platform for Indian equities. It combines a custom **LSTM → GAT** deep learning model with production-grade regulatory engines (FRTB, CVA, VaR) and an interactive 6-tab dashboard — the kind of stack a quant risk team would build, not a textbook exercise.

| Capability | What it does |
|---|---|
| **Distress Prediction** | LSTM encodes 30-day price history per stock; dual-layer GAT propagates contagion across the correlation graph |
| **Regime Detection** | 2-state HMM (fitted 2015–2021, inferred OOS 2022+) detects NORMAL / CRISIS and adjusts VaR weighting live |
| **Stress Testing** | Inject a shock to any stock; watch delta-PD propagate to all 50 firms via attention weights |
| **Regulatory Capital** | FRTB Sensitivity-Based Method capital charges, pre- and post-shock |
| **CVA** | Contagion-adjusted Credit Valuation Adjustment per firm using GAT attention matrix as shock propagation |
| **Explainability** | GNNExplainer attributes each distress prediction to specific counterparties and input features |

---

## What Makes This Technically Interesting

- **Custom LSTM → GAT architecture** — Each Nifty 50 stock runs through a 2-layer LSTM (hidden=64) to compress its 30-day sequence into a node embedding. Two GAT layers (4-head, then 1-head) then propagate distress signals across a dynamic correlation graph where edge weights are Pearson correlations over a rolling 60-day window. The second GAT layer returns attention weights used downstream for SIFI ranking and CVA.

- **Dynamic correlation graph** — Rather than a fixed adjacency matrix, the graph is reconstructed weekly from rolling 60-day correlations with a threshold filter (|ρ| > 0.6), so the network topology itself is regime-dependent. Crisis periods produce denser, stronger graphs, which is exactly why shock propagation amplifies 26× under stress.

- **HMM-conditioned VaR** — The Historical Simulation VaR uses HMM crisis posterior probabilities as importance weights when the current regime is CRISIS. This overweights tail-loss days in the lookback window without discarding history, producing more conservative (and more accurate) estimates under regime stress.

- **FRTB SBM implemented from scratch** — The Sensitivity-Based Method aggregates delta and vega sensitivities from Black-Scholes Greeks into sector buckets (K_b), then cross-aggregates to total capital using the Basel IV correlation parameters (ρ=0.25 intra-bucket, γ=0.15 inter-bucket). Pre/post-shock capital is computed in under a second.

- **CVA propagated via attention matrix** — After a shock, the GAT attention matrix routes distress scores through the network using logit-space shifts rather than linear addition, keeping updated PDs in [0,1] and reflecting non-linear amplification at high-distress nodes.

- **SIFI scoring from attention aggregation** — Rather than a separate model, SIFI spillover scores are derived by summing outgoing GAT attention weights per node over the validation dataset — a natural measure of each firm's capacity to transmit distress. No additional training required.

- **GNNExplainer integration** — PyG's GNNExplainer runs a 200-epoch mask-optimization loop per firm to produce both node feature masks (which of the 5 input features drove the prediction) and edge masks (which counterparty contributed most), enabling regulatory-grade "why did the model flag this?" attribution.

---

## Model Architecture

```
Input: (N=50 stocks, T=30 days, F=5 features)
  ↓
LSTMEncoder (2-layer, hidden=64) — runs independently per node
  → node embeddings: (N, 64)
  ↓
GATConv Layer 1 (4 heads × out=32, concat) — edge_attr = correlation weight
  → (N, 128)  +  ELU + Dropout
  ↓
GATConv Layer 2 (1 head, out=32) — returns attention weights α
  → (N, 32)  +  attention map: (E,)
  ↓
Linear predictor → distress logit per node: (N,)
```

**Input features per node per day:** log return, realized volatility, normalized volume, high-low range, close-to-SMA20

---

## Key Numbers

| Item | Value |
|---|---|
| Training universe | Nifty 50, Jan 2015 – Dec 2023 |
| Validation (OOS) | Jan 2024 – Jan 2025 |
| Sequence length | 30 trading days |
| Correlation graph window | 60 days, threshold \|ρ\| > 0.6 |
| HMM fit window | 2015–2021 (OOS inference: 2022+) |
| VaR confidence levels | 95%, 99%, 99.5% |
| Monte Carlo simulations | 10,000 paths (dynamic covariance) |
| Basel III backtest window | 250 trading days |
| FRTB equity risk weight | 40% (SA simplified) |
| CVA LGD assumption | 60% (fixed) |
| Contagion amplification | **26× crisis vs. calm** (−40% shock, averaged across all 50 firms) |

---

## SIFI Rankings (Latest Snapshot)

Top firms by aggregate outgoing GAT attention — highest capacity to spread distress:

| Rank | Ticker | Spillover Score | Baseline Distress |
|---|---|---|---|
| 1 | TITAN.NS | 0.0197 | 7.0% |
| 2 | TATACONSUM.NS | 0.0185 | 7.2% |
| 3 | RELIANCE.NS | 0.0179 | 7.0% |
| 4 | TATASTEEL.NS | 0.0176 | 6.9% |
| 5 | ULTRACEMCO.NS | 0.0171 | 9.1% |

A −40% shock to RELIANCE produces a **+88 percentage point** delta-PD on itself and cascades to all connected financials, materials, and consumer names in one propagation step.

---

## Dashboard — 6 Tabs

The Streamlit dashboard surfaces every model output interactively. No code required.

| Tab | What you can do |
|---|---|
| 🕸️ **Network & Shock** | Select any stock, set shock magnitude (−5% to −80%), see contagion propagate across the live GAT network |
| 📉 **VaR** | HS / Parametric / MC VaR at three confidence levels, Basel III traffic-light backtesting, full HMM crisis posterior history |
| 🏛️ **FRTB** | Pre/post-shock SA capital charges by sector bucket, option Greeks (delta/vega) per position |
| ⚠️ **CVA** | Per-firm CVA and ΔCVA under the current shock, aggregated portfolio CVA, HMM λ multiplier |
| 🏆 **SIFI Ranking** | Full 50-firm spillover ranking with bar chart; color-coded by baseline distress probability |
| 🔍 **XAI Explainer** | On-demand GNNExplainer for any firm: feature attribution + counterparty attribution |

---

## Tech Stack

| Purpose | Tools |
|---|---|
| Deep learning | PyTorch, PyTorch Geometric |
| Temporal encoding | LSTM (custom LSTMEncoder) |
| Graph encoding | GATConv (2-layer, edge features) |
| Explainability | GNNExplainer (PyG) |
| Regime detection | hmmlearn (Gaussian HMM) |
| Risk engines | NumPy, SciPy (VaR, CVA, FRTB) |
| Visualization | Plotly, Streamlit-Agraph |
| Dashboard | Streamlit |
| Market data | yfinance |

---

## Project Structure

```
rasr_ge/
├── models/
│   └── rasr_ge.py              # LSTM → GAT model definition
│
├── training/
│   ├── dataset.py              # FinancialGraphDataset (PyG)
│   ├── train.py                # Training loop with early stopping
│   └── evaluate.py             # AUROC, PRAUC, Brier, KS, F1, calibration
│
├── risk/
│   ├── hmm_regime.py           # 2-state HMM, λ derivation, regime inference
│   ├── var_engine.py           # HS (regime-conditioned), Parametric, MC VaR
│   ├── cva_engine.py           # GAT-propagated CVA per firm
│   └── frtb_engine.py          # FRTB SBM capital charges
│
├── stress/
│   ├── shock_engine.py         # Inject shocks, return delta-PD + attention
│   ├── counterfactual.py       # Crisis vs. calm contagion multiplier
│   └── sifi_ranking.py         # Aggregate outgoing attention → SIFI scores
│
├── xai/
│   └── explainer.py            # GNNExplainer wrapper, feature + edge attribution
│
├── dashboard/
│   └── app.py                  # 6-tab Streamlit app
│
├── data_pipeline.py            # End-to-end: fetch → features → graphs → labels
├── extract_data.py             # yfinance downloader
├── config.yaml                 # All hyperparameters
└── requirements.txt
```

---

## Running Locally

```bash
git clone https://github.com/prasanna1504/Systematic-Risk.git
cd Systematic-Risk
pip install -r rasr_ge/requirements.txt
streamlit run rasr_ge/dashboard/app.py
```

The repository ships with preprocessed data (`rasr_ge/data/processed/`) and a trained checkpoint (`rasr_ge/checkpoints/best_model.pt`). The dashboard loads immediately without retraining.

To retrain from scratch:

```bash
python rasr_ge/data_pipeline.py          # fetch & preprocess (2015–2025)
python rasr_ge/training/train.py         # train LSTM-GAT, saves best checkpoint
python rasr_ge/training/evaluate.py      # AUROC, F1, calibration on 2024 OOS set
```
