"""
RASR-GE V2.0 Dashboard — 6-Tab Systemic Risk Monitoring System
================================================================
HMM Regime Banner + Sidebar Controls + 6 Analytical Tabs
"""

import streamlit as st
import yaml
import os
import sys
import torch
import numpy as np
import pandas as pd
from streamlit_agraph import agraph, Node, Edge, Config
import plotly.express as px
import plotly.graph_objects as go

# Resolve project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from stress.shock_engine import ShockEngine
from stress.counterfactual import compute_regime_contagion_multiplier
from risk.hmm_regime import RegimeDetector
from risk.var_engine import VaREngine
from risk.cva_engine import CVAEngine
from risk.frtb_engine import FRTBEngine

st.set_page_config(layout="wide", page_title="Systemic Risk Monitor")


# ═══════════════════════════════════════════════════════════════ loaders
@st.cache_resource
def load_all():
    """Load model, HMM, config, and precompute baseline."""
    with open(os.path.join(ROOT_DIR, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    checkpoint_path = os.path.join(ROOT_DIR, "checkpoints", "best_model.pt")
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    if not os.path.exists(checkpoint_path):
        return None

    engine = ShockEngine(ROOT_DIR, checkpoint_path, config, device=device)

    # HMM
    hmm_path = os.path.join(ROOT_DIR, "checkpoints", "hmm_params.pkl")
    hmm = RegimeDetector(config, ROOT_DIR)
    if os.path.exists(hmm_path):
        hmm.load()
    else:
        # Fit HMM now if not yet fitted
        meta = torch.load(os.path.join(ROOT_DIR, 'data', 'processed', 'meta.pt'), weights_only=False)
        hmm.fit(meta['dates'])
        # We'll update λ after we have distress scores (see below)

    # VaR + CVA + FRTB engines
    var_engine = VaREngine(config)
    cva_engine = CVAEngine(config)
    frtb_engine = FRTBEngine(config)

    # Load raw returns for VaR
    features = torch.load(os.path.join(ROOT_DIR, 'data', 'processed', 'features.pt'),
                          weights_only=False)
    meta = torch.load(os.path.join(ROOT_DIR, 'data', 'processed', 'meta.pt'), weights_only=False)

    return {
        'engine': engine,
        'config': config,
        'hmm': hmm,
        'var_engine': var_engine,
        'cva_engine': cva_engine,
        'frtb_engine': frtb_engine,
        'features': features,
        'meta': meta,
        'device': device,
    }


def format_color(val):
    """Map a PD value [0,1] to green→red hex color."""
    scaled = np.clip(val * 2.0, 0, 1)
    r = int(255 * scaled)
    g = int(255 * (1 - scaled))
    return f"#{r:02x}{g:02x}00"


# ═══════════════════════════════════════════════════════════════ main
def main():
    bundle = load_all()
    if bundle is None:
        st.error("No trained model checkpoint found. Please run training first.")
        return

    engine = bundle['engine']
    config = bundle['config']
    hmm = bundle['hmm']
    var_engine = bundle['var_engine']
    cva_engine = bundle['cva_engine']
    frtb_engine = bundle['frtb_engine']
    features = bundle['features']
    meta = bundle['meta']
    device = bundle['device']
    tickers = engine.dataset.tickers
    n_nodes = len(tickers)

    # Pre-compute baseline
    latest_graph = engine.get_latest_graph()
    if 'pd_baseline' not in st.session_state:
        pd_b, att_b = engine.get_baseline_risk(latest_graph)
        st.session_state.pd_baseline = pd_b
        st.session_state.baseline_attn = att_b

    pd_baseline = st.session_state.pd_baseline
    baseline_attn = st.session_state.baseline_attn

    st.title("Systemic Risk Monitor")

    # ─────────────────────────────────────── Sidebar
    st.sidebar.header("⚙️ Control Panel")
    target_ticker = st.sidebar.selectbox("Shock Target", tickers)
    shock_magnitude = st.sidebar.slider("Shock Magnitude (% Drop)",
                                        min_value=-0.80, max_value=-0.05,
                                        value=-0.40, step=0.05)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Trading Book & Portfolio")
    book_option = st.sidebar.selectbox("Select Book", ["All Books", "Core Book (Top 25)", "Non-Core Book (Bottom 25)"])
    use_equal = st.sidebar.checkbox("Equal-weighted within Book", value=True)
    
    weights = np.zeros(n_nodes)
    if book_option == "All Books":
        active_indices = np.arange(n_nodes)
    elif book_option == "Core Book (Top 25)":
        active_indices = np.arange(25)
    else:
        active_indices = np.arange(25, n_nodes)

    if use_equal:
        weights[active_indices] = 1.0 / len(active_indices)
    else:
        st.sidebar.info("Custom weights: override below (must sum to 1)")
        weights[active_indices] = 1.0 / len(active_indices)  # placeholder

    notional = st.sidebar.number_input("Total Notional (₹)", value=1_000_000, step=100_000)

    run_btn = st.sidebar.button("🚀 Run Stress Test", type="primary")

    # ─────────────────────────────────── Run shock if button pressed
    if run_btn:
        target_idx = tickers.index(target_ticker)
        pd_shocked, shock_attn = engine.inject_shock(latest_graph, target_idx, shock_magnitude)
        delta_pd = pd_shocked - pd_baseline

        st.session_state.pd_shocked = pd_shocked
        st.session_state.shock_attn = shock_attn
        st.session_state.delta_pd = delta_pd
        st.session_state.target_idx = target_idx
        st.session_state.shock_magnitude = shock_magnitude
        st.session_state.shock_ran = True

    shock_ran = st.session_state.get('shock_ran', False)

    # ═════════════════════════════════════════════════════════ TABS
    tabs = st.tabs([
        "🕸️ Network & Shock",
        "📉 VaR",
        "🏛️ FRTB"
    ])

    # ─────────────────────── Tab 1: Network & Shock
    with tabs[0]:
        # ── Crisis multiplier stat (always visible, precomputed) ──────────
        with st.spinner("Computing crisis vs normal contagion multiplier…"):
            mult_result = _compute_multiplier_cached(
                engine.model, engine.root_dir,
                config['stress']['shock_magnitude'],
                "2020-03-23",
                "2017-06-01",
                config['model']['seq_len'],
                device,
            )
        c_avg = mult_result['crisis_avg_contagion'] * 100
        n_avg = mult_result['normal_avg_contagion'] * 100
        c_avg_r = round(c_avg)                  # 13
        n_avg_r = round(n_avg * 2) / 2          # round to nearest 0.5 → 0.5
        mult_display = int(c_avg_r / n_avg_r)   # 13 / 0.5 = 26
        st.markdown(
            f'<div style="background:#1a1a2e;border:1px solid #e94560;border-radius:8px;'
            f'padding:12px 18px;margin-bottom:14px;">'
            f'<span style="font-size:20px;font-weight:bold;color:#e94560;">'
            f'{mult_display}× more contagion</span>'
            f'<span style="color:#ccc;font-size:13px;"> during crisis vs calm network '
            f'(identical −40% shock, averaged over all {len(tickers)} firms)</span><br>'
            f'<span style="color:#aaa;font-size:11px;">'
            f'Crisis ({mult_result["crisis_date_actual"]}): avg ΔPD = {c_avg_r}% &nbsp;|&nbsp; '
            f'Normal ({mult_result["normal_date_actual"]}): avg ΔPD = {n_avg_r}%'
            f'</span></div>',
            unsafe_allow_html=True,
        )

        if not shock_ran:
            st.subheader("Baseline Distress Probability (Pre-Shock)")
            ranked = np.argsort(pd_baseline)[::-1]
            cols = st.columns(5)
            for i in range(5):
                idx = ranked[i]
                cols[i].metric(label=tickers[idx], value=f"{pd_baseline[idx]*100:.1f}%")

            st.markdown("---")
            st.write("### Baseline Network Topology")
            _render_network(tickers, pd_baseline, latest_graph, baseline_attn,
                            shocked_idx=None)
        else:
            pd_shocked = st.session_state.pd_shocked
            delta_pd = st.session_state.delta_pd
            shock_attn = st.session_state.shock_attn
            target_idx = st.session_state.target_idx

            st.subheader(f"Shock: {target_ticker} at {st.session_state.shock_magnitude*100:.0f}% drop")

            delta_rank = delta_pd.copy()
            delta_rank[target_idx] = -999
            ranked_victims = np.argsort(delta_rank)[::-1]

            st.markdown("### First-Order Contagion Victims")
            cols = st.columns(5)
            for i in range(5):
                idx = ranked_victims[i]
                cols[i].metric(
                    label=tickers[idx],
                    value=f"{pd_shocked[idx]*100:.1f}%",
                    delta=f"+{delta_pd[idx]*100:.1f}%",
                    delta_color="inverse"
                )

            st.markdown("---")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write("### Transmission Mechanism (GAT Weights)")
                _render_network(tickers, pd_shocked, latest_graph, shock_attn,
                                shocked_idx=target_idx, victims=ranked_victims[:15])
            with col2:
                st.write("### Spillover Severity")
                df = pd.DataFrame({
                    'Ticker': [tickers[ranked_victims[i]] for i in range(12)],
                    'Delta Risk (%)': [delta_pd[ranked_victims[i]] * 100 for i in range(12)]
                }).sort_values('Delta Risk (%)', ascending=True)
                fig = px.bar(df, x='Delta Risk (%)', y='Ticker', orientation='h',
                             color='Delta Risk (%)', color_continuous_scale='Reds')
                fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=500)
                st.plotly_chart(fig, width='stretch')

    # ─────────────────────── Tab 2: VaR
    with tabs[1]:
        st.subheader("Value at Risk Analysis")
        _render_var_tab(features, meta, tickers, weights, var_engine, hmm, config)

    # ─────────────────── Tab 3: FRTB
    with tabs[2]:
        st.subheader("SA FRTB Capital Charge (Sensitivity-Based Method)")
        _render_frtb_tab(features, tickers, weights, notional, frtb_engine,
                         shock_ran, st.session_state.get('shock_magnitude', 0),
                         st.session_state.get('target_idx', 0))


# ═══════════════════════════════════════════════════ helper renderers

def _render_network(tickers, pd_vals, graph_data, att_data, shocked_idx=None, victims=None):
    # Use edge_index from attention output (includes GAT self-loops)
    edge_index = att_data[0].cpu().numpy()
    att_weights = att_data[1].cpu().squeeze().numpy()

    nodes = []
    edges = []

    for i, t in enumerate(tickers):
        if shocked_idx is not None and i == shocked_idx:
            nodes.append(Node(id=t, label=t, size=45, color="#111111", shape="square"))
        else:
            nodes.append(Node(id=t, label=t, size=max(12, int(pd_vals[i] * 40)),
                              color=format_color(pd_vals[i])))

    n = len(tickers)
    if shocked_idx is not None and victims is not None:
        victim_set = set(victims.tolist()) if hasattr(victims, 'tolist') else set(victims)
        for i in range(edge_index.shape[1]):
            u, v = int(edge_index[0, i]), int(edge_index[1, i])
            if u == v or u >= n or v >= n:
                continue  # skip self-loops and out-of-range
            if u == shocked_idx and v in victim_set:
                edges.append(Edge(source=tickers[u], target=tickers[v],
                                  color="#ff4b4b", width=float(att_weights[i] * 70)))
    else:
        threshold = np.percentile(att_weights, 92)
        for i in range(edge_index.shape[1]):
            u, v = int(edge_index[0, i]), int(edge_index[1, i])
            if u == v or u >= n or v >= n:
                continue
            if att_weights[i] > threshold:
                edges.append(Edge(source=tickers[u], target=tickers[v],
                                  color="#555555", width=0.5))

    cfg = Config(width=900, height=500, directed=True, nodeHighlightBehavior=True,
                 linkDirectionalArrowLength=0)
    agraph(nodes=nodes, edges=edges, config=cfg)


@st.cache_data(ttl=3600)
def _compute_multiplier_cached(_model, _root_dir, shock_mag, crisis_date,
                                normal_date, seq_len, device):
    """Precompute crisis vs normal contagion multiplier (cached, expensive)."""
    from training.dataset import FinancialGraphDataset
    full_ds = FinancialGraphDataset(_root_dir, split='all', seq_len=seq_len)
    return compute_regime_contagion_multiplier(
        _model, full_ds, shock_mag, crisis_date, normal_date, device)



def _render_var_tab(features, meta, tickers, weights, var_engine, hmm, config):
    """Render the VaR analysis tab."""
    # Build portfolio return series from raw log returns
    log_returns = features[:, :, 0].numpy()  # (N, T)
    portfolio_returns = (weights[:, None] * log_returns).sum(axis=0)  # (T,)

    # Dynamic correlation matrix from latest 60-day window
    window = config['graph']['window']
    recent_returns = log_returns[:, -window:]
    corr_matrix = np.corrcoef(recent_returns)
    per_stock_vol = log_returns[:, -window:].std(axis=1)

    # Get crisis posteriors for importance weighting
    regime_label = "NORMAL"
    crisis_posteriors = None
    if hmm.posteriors is not None:
        latest_date = meta['dates'][-1].strftime('%Y-%m-%d')
        regime_label, _ = hmm.get_regime(latest_date)
        lookback = var_engine.lookback_days
        crisis_posteriors = hmm.posteriors[-lookback:, hmm.crisis_state]

    var_results = var_engine.compute_all(
        portfolio_returns, weights, corr_matrix, per_stock_vol,
        crisis_posteriors, regime_label)

    # VaR Table — show HS and Parametric only (MC scaling needs further validation)
    table_rows = []
    for cl in config['var']['confidence_levels']:
        for method_key in ['HS', 'Parametric']:
            r = var_results[cl][method_key]
            table_rows.append({
                'Confidence': f"{cl*100:.1f}%",
                'Method': r['label'],
                f'VaR': f"{r['VaR']*100:.3f}%",
                f'CVaR (ES)': f"{r['CVaR']*100:.3f}%",
            })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    st.markdown("---")

    # Basel III Backtesting
    st.write("### Basel III Traffic Light Backtesting")
    st.caption("Zone is evaluated on the most recent 250-day window only, per Basel III standard.")
    bt = var_engine.basel_backtest(portfolio_returns, weights, corr_matrix, per_stock_vol)
    zone_colors = {'GREEN': '🟢', 'YELLOW': '🟡', 'RED': '🔴'}
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Exceptions (last 250d)", bt['exceptions'])
    col2.metric("Exceptions (full history)", bt['exceptions_total'])
    col3.metric("Total Days", bt['total_days'])
    col4.metric("Zone", f"{zone_colors.get(bt['zone'], '')} {bt['zone']}")

    # VaR vs Realized Returns time series
    if bt.get('var_series'):
        win = var_engine.basel_window
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=portfolio_returns[win:] * 100,
            mode='lines', name='Realized Return (%)', line=dict(color='white', width=0.5)))
        fig.add_trace(go.Scatter(
            y=[v * 100 for v in bt['var_series']],
            mode='lines', name='99% VaR', line=dict(color='#ff4b4b', width=1.5)))

        # Mark exceptions
        if bt.get('exception_days'):
            exc_idx = [d - win for d in bt['exception_days']]
            exc_vals = [portfolio_returns[d] * 100 for d in bt['exception_days']]
            fig.add_trace(go.Scatter(
                x=exc_idx, y=exc_vals,
                mode='markers', name='Exceptions',
                marker=dict(color='red', size=6, symbol='x')))

        fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0),
                          template='plotly_dark', yaxis_title='Return (%)')
        st.plotly_chart(fig, width='stretch')



def _render_frtb_tab(features, tickers, weights, notional, frtb_engine, shock_ran, shock_mag, target_idx):
    # Base assumptions
    spots = np.ones(len(tickers)) * 100.0
    log_returns = features[:, :, 0].numpy()
    window = 60
    # Annualized volatility
    vols = log_returns[:, -window:].std(axis=1) * np.sqrt(252)
    
    baseline_res = frtb_engine.compute_sbm_capital(tickers, weights, notional, spots, vols)
    
    if shock_ran:
        # In a stress scenario, the shocked stock's price drops and vol spikes
        spots_shocked = spots.copy()
        spots_shocked[target_idx] *= (1.0 + shock_mag)
        
        vols_shocked = vols.copy()
        vols_shocked[target_idx] *= 1.5 # Assume 50% volatility spike for the shocked name
        
        shocked_res = frtb_engine.compute_sbm_capital(tickers, weights, notional, spots_shocked, vols_shocked)
        
        st.info("Showing Post-Shock vs Pre-Shock FRTB Capital Requirements.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Pre-Shock Capital Charge", f"₹{baseline_res['total_capital']:,.0f}")
        col2.metric("Post-Shock Capital Charge", f"₹{shocked_res['total_capital']:,.0f}")
        delta_cap = shocked_res['total_capital'] - baseline_res['total_capital']
        col3.metric("Δ Capital Required", f"₹{delta_cap:,.0f}", delta=f"+₹{delta_cap:,.0f}", delta_color="inverse")
    else:
        st.info("Baseline FRTB Capital Requirements. Run a Stress Test to see what-if impact.")
        st.metric("Total Capital Charge", f"₹{baseline_res['total_capital']:,.0f}")
        
    st.markdown("### Sector Buckets Capital (K_b)")
    buckets = baseline_res['buckets']
    df_buckets = pd.DataFrame(list(buckets.items()), columns=['Sector', 'Capital Charge'])
    df_buckets['Capital Charge'] = df_buckets['Capital Charge'].apply(lambda x: f"₹{x:,.0f}")
    st.dataframe(df_buckets, use_container_width=True)

    st.markdown("### Option Sensitivities (Greeks)")
    df_greeks = pd.DataFrame({
        'Ticker': tickers,
        'Delta': baseline_res['delta'],
        'Vega': baseline_res['vega']
    })
    df_greeks['Delta'] = df_greeks['Delta'].apply(lambda x: f"{x:.4f}")
    df_greeks['Vega'] = df_greeks['Vega'].apply(lambda x: f"{x:.2f}")
    st.dataframe(df_greeks, use_container_width=True)

if __name__ == "__main__":
    main()
