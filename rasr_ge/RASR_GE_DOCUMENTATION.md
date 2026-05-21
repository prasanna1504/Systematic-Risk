# RASR-GE: Regime-Aware Systemic Risk Graph Engine
## Complete Technical Reference — Aligned to Project Deliverables

---

## Table of Contents

1. [The Problem: Why Traditional Risk Models Fall Short](#1-the-problem)
2. [Pipeline Overview](#2-pipeline-overview)
3. [Data Pipeline: Features, Denoising, and Graph Construction](#3-data-pipeline)
4. [Model Architecture: LSTM + GAT](#4-model-architecture)
5. [Training: Loss, Metrics, Threshold Calibration](#5-training)
6. [Risk Analytics Framework: VaR, CVA, Basel III](#6-risk-analytics-framework)
7. [Systemic Shock Propagation and Contagion Hubs](#7-systemic-shock-propagation-and-contagion-hubs)
8. [Regime-Aware Stress Testing: HMM, λ, and the Crisis Multiplier](#8-regime-aware-stress-testing)
9. [Dashboard](#9-dashboard)
10. [Key Innovations vs Traditional Models](#10-key-innovations)
11. [Regulatory Alignment: Basel III](#11-regulatory-alignment)

---

## 1. The Problem

### 1.1 The Microprudential Trap

Traditional risk management monitors each firm in isolation — balance sheets, earnings, volatility. This approach is incomplete in a dangerous way. A firm can appear solvent by every standard metric and still be a catastrophic systemic liability if it is densely interconnected with others through credit agreements, shared investor bases, or correlated asset holdings.

The **2008 Global Financial Crisis** made this unforgettable. Lehman Brothers' balance sheet was not uniquely poor in the months before collapse. What regulators missed was the **network**: counterparty obligations, mortgage-backed securities held simultaneously across dozens of institutions, and the behavioural correlation that emerges when all participants react to the same shock. When Lehman fell, it cascaded — margin calls, forced asset sales, frozen interbank lending. None of this was in any microprudential model.

### 1.2 Systemic Risk and Contagion

**Systemic risk** is the probability that distress at one entity triggers broader collapse across the financial system. It is a **network property**, not a property of any single node.

**Contagion** travels through several channels:
- **Direct exposure**: Firm A holds bonds issued by Firm B — B's distress causes direct losses at A.
- **Liquidity spirals**: Forced asset sales depress prices, forcing other holders to mark down.
- **Behavioural correlation**: Investors pull capital from firms perceived as similar, even without any direct financial link.
- **Index heavyweight synchrony**: Large-cap index constituents face correlated institutional selling when any major constituent is stressed.

Traditional models — VaR, GARCH, single-firm neural networks — are blind to these dynamics because they treat each stock as an independent time series.

### 1.3 The Macroprudential Mandate

Post-2008 Basel III introduced **macroprudential supervision**: monitoring the stability of the system as a whole, not just individual firms. This requires tools that:

1. Represent connectivity between firms explicitly.
2. Model how a shock at one node propagates through the network.
3. Produce actionable early-warning signals before contagion materialises.

RASR-GE is built to address exactly this gap.

---

## 2. Pipeline Overview

RASR-GE processes raw OHLCV data for all 50 Nifty 50 constituent stocks and produces, for each stock on each trading day, a **Probability of Distress (PD)** score — the model's estimate that the stock will drop more than 5% within the next five trading days.

Beyond individual scores, the system provides a full quantitative risk management suite:

| Stage | Component | Resume Bullet |
|-------|-----------|---------------|
| 1 | Wavelet DWT Denoising | Data quality for all downstream |
| 2 | Dynamic Correlation Graph | Foundation for shock propagation |
| 3 | LSTM + GAT Model | Shock propagation via GAT |
| 4 | VaR Engine (HS + Parametric) | Bullet 1: VaR framework |
| 5 | CVA Engine (contagion-adjusted) | Bullet 1: CVA across 49 firms |
| 6 | Basel III Backtesting | Bullet 1: traffic-light validation |
| 7 | HMM Regime Detector | Bullet 3: regime-aware stress testing |
| 8 | SIFI Ranking | Bullet 2: major contagion hubs |
| 9 | Crisis Multiplier Computation | Bullet 3: 23x more contagion finding |

---

## 3. Data Pipeline

### 3.1 Five Features Per Stock Per Day

For each of the 50 Nifty stocks, five features are computed daily from raw OHLCV data. Each captures a distinct dimension of distress risk.

**Feature 1 — log_return**
```
log_return_t = log(Close_t / Close_{t-1})
```
Log returns are additive over time and more symmetrically distributed than percentage returns — better behaved for neural network training. This is the primary distress signal: large negative log returns directly precede the distress label.

**Feature 2 — realized_vol (20-day)**

Rolling 20-day standard deviation of daily log returns. Volatility is a leading indicator of regime change: a firm whose volatility has quietly been rising is under accumulating stress even if price has not decisively moved. Realized volatility captures this pre-distress signature that pure return series miss.

**Feature 3 — normalized_volume (20-day)**
```
norm_vol_t = Volume_t / mean(Volume_{t-20:t})
```
Abnormally high volume accompanies institutional liquidation. When a major holder reduces a position under duress, volume spikes before price fully reflects the change. Normalising by rolling mean makes this comparable across stocks of vastly different trading scales.

**Feature 4 — high_low_range**
```
hl_range_t = (High_t - Low_t) / Close_t
```
Intraday range as a fraction of closing price. A wide range without a directional close is a signature of uncertainty and buyer-seller disagreement — common during the first hours of a contagion event, before daily close prices fully move.

**Feature 5 — close_to_sma20**
```
c_sma20_t = (Close_t / SMA_20_t) - 1
```
How far price sits from its 20-day moving average. A stock trading significantly below its 20-day average is under sustained downward pressure; this contextualises the daily return within the recent trend.

Together, these five features form a compact fingerprint of each firm's current risk state. Every day, the LSTM receives a 30-day window of these features per stock — a tensor of shape **(30, 5)** per node.

### 3.2 Wavelet DWT Denoising

Raw financial time series contain high-frequency noise — random day-to-day fluctuations from order flow, bid-ask bounce, and microstructure effects — that obscures the stress dynamics the model needs to learn.

RASR-GE uses **Discrete Wavelet Transform (DWT) denoising** with the Daubechies-4 (`db4`) wavelet at decomposition level 3. The signal is split into:
- One **approximation coefficient array** (cA3): the low-frequency structural trend.
- Three **detail coefficient arrays** (cD1, cD2, cD3): high-frequency noise components.

All three detail arrays are zeroed out. The signal is reconstructed from approximation coefficients alone — smooth, retaining genuine trend and volatility dynamics while discarding daily microstructure noise. The advantage over moving averages is that DWT preserves **time-locality**: a volatility spike in March 2020 is not blurred together with a spike in September 2023.

### 3.3 Dynamic Correlation Graph

Rather than fixed supply-chain or ownership linkages (infrequent, incomplete, only first-order), RASR-GE constructs the graph from **rolling 60-day Pearson correlations** between log returns.

**Adjacency Rule:**
```
A_t[i,j] = ρ_{ij}    if |ρ_{ij}| > 0.60
A_t[i,j] = 0          otherwise
```

This creates a **sparse, weighted, directed** graph that updates every single day. The edge weight is the actual correlation value, not a binary flag — it directly informs how strongly distress flows along each connection.

**Crisis densification:** In normal regimes, correlations are moderate and heterogeneous; the graph is relatively sparse. During a market crisis, a well-documented phenomenon occurs: **correlation convergence**. Systematic selling pressure causes most large-cap stocks to move together regardless of sector — correlations that were 0.35–0.45 jump above 0.60 and stay there. The graph becomes nearly fully connected.

This is precisely what makes systemic risk so dangerous: a manageable network in calm times becomes almost fully connected under stress, turning a localised shock into a market-wide event. RASR-GE's dynamic adjacency captures this regime shift **automatically from data**, with no hard-coded crisis flag.

### 3.4 Label Generation

```
distress = 1   if  (Close_{t+5} - Close_t) / Close_t  <  -0.05
distress = 0   otherwise
```

A firm is distressed if its price drops more than 5% over the next five trading days. Positive rate: **~7.4%** — approximately 1 in 14 observations is a genuine distress event.

---

## 4. Model Architecture

### 4.1 LSTM Temporal Encoder

Before any cross-firm communication, each stock needs a representation capturing its recent history.

For each node `i`, the LSTM receives the 30-day denoised feature sequence — shape **(30, 5)**. It processes this day by day, maintaining a hidden state summarising what has happened.

**Configuration:** input_dim=5, hidden=64, layers=2, dropout=0.3

The final hidden state **h_i** (64-dimensional vector) is the LSTM's compressed summary of everything financially relevant about firm `i` over the past 30 days. It encodes whether returns have been trending down, whether volatility has been rising, whether volume has been surging.

Why LSTM rather than feedforward? Because distress does not happen instantaneously — the *sequence* matters: first a volatility uptick, then elevated volume, then a price break below SMA. The LSTM's gating mechanism learns which past signals to retain and which to discard. Temporal memory that a feedforward network would lose.

### 4.2 Graph Attention Network (GAT)

Once each firm has its temporal hidden state, the GAT determines how much distress each firm's state should influence its neighbours' risk assessment.

**Attention weight computation:**

*Step 1 — Raw attention score:*
```
e_{ij} = LeakyReLU( a^T [ W·h_i  ||  W·h_j  ||  W_e·ρ_{ij} ] )
```
Node states h_i and h_j are projected and concatenated with the edge feature (correlation weight ρ_{ij}). This scalar answers: "given what firm i and j look like, and how strongly they are correlated, how relevant is j's state to i's risk?"

*Step 2 — Softmax normalisation:*
```
α_{ij} = exp(e_{ij}) / Σ_{k ∈ N(i)} exp(e_{ik})
```
α_{ij} is interpretable as: "what fraction of node i's attention is directed at neighbour j?"

*Step 3 — Message aggregation:*
```
h_i' = σ( Σ_{j ∈ N(i)} α_{ij} · W · h_j )
```

**Why edge features (correlation weights) matter:** A firm that is 0.85 correlated with another should receive a much stronger distress signal than a firm that is only 0.61 correlated. Without edge features, the GAT might assign similar attention to both. With edge features, attention amplifies propagation along high-correlation edges and dampens it along weaker ones. Graph edges are not just structural — they are dynamically informative.

**Architecture:** Two-layer GAT. Layer 1: 4-head attention (outputs concatenated). Layer 2: 1-head attention (consolidates into final 32-dim node embedding).

**Asymmetric attention:** α_{ij} ≠ α_{ji}. The model can learn that j's distress strongly predicts i's risk without the reverse being equally true. This asymmetry is economically realistic — a major bank's distress propagates more broadly than a mid-cap stock's distress would.

### 4.3 Prediction Head

```
logit_i  = W_out · z_i + b_out
PD_i     = σ(logit_i) = 1 / (1 + exp(-logit_i))
```

**PD_i** is the probability that firm i will experience a drop greater than 5% within the next five trading days, conditioned on the current network state.

---

## 5. Training

### 5.1 Class Imbalance: BCEWithLogitsLoss with pos_weight=5.0

With only 7.4% positive labels, a naive model that always predicts "no distress" achieves 92.6% accuracy while being completely useless. RASR-GE uses:

```
L = -(1/N) Σ_i [ 5.0 · y_i · log(σ(logit_i))  +  (1-y_i) · log(1 - σ(logit_i)) ]
```

`pos_weight=5.0` means each genuine distress event contributes five times as much to the loss as a non-distress event. From a systemic risk perspective: missing a real contagion event is far more costly than a false alarm.

### 5.2 AUROC as Primary Metric

AUROC measures the model's ability to *rank* firms by risk: the probability that a randomly chosen distressed firm is assigned a higher PD score than a randomly chosen non-distressed firm. An AUROC of 0.82 means in 82% of such pairings, the model correctly ranks the genuinely distressed firm as higher risk.

This is the right metric because the task is fundamentally a **ranking task** — a risk manager wants to know which firms are most at risk right now, not a binary verdict.

### 5.3 Threshold Calibration: τ* = 0.21

Default threshold of 0.50 is appropriate only when classes are balanced. Here they are not.

```
τ* = argmax_τ  F1( (y_prob ≥ τ), y_true )
```

Optimal threshold: τ* ≈ **0.21**. A PD of 0.21 already represents nearly 3× the base rate of 7.4%.

| Threshold | F1 Score |
|-----------|----------|
| 0.50 (default) | 0.037 |
| 0.21 (calibrated) | 0.337 |

A 30-point improvement in practical detection quality with no change to model weights.

### 5.4 Training Setup

- **Optimiser**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau on val AUROC (factor=0.5, patience=5)
- **Gradient clipping**: norm=1.0 — prevents LSTM gradient explosion
- **Early stopping**: patience=10 epochs on val AUROC
- **Train/Val split**: Train 2015-2023; Val 2024-2025 (strict temporal split, no data leakage)

---

## 6. Risk Analytics Framework: VaR, CVA, Basel III

*This section maps directly to Resume Bullet 1: "computing Value-at-Risk (Historical Simulation, Parametric) and contagion-adjusted CVA across 49 firms with Basel III traffic-light backtesting."*

### 6.1 Value at Risk — Historical Simulation

Historical Simulation (HS) is the most assumption-light VaR method. It uses the **empirical distribution** of past portfolio returns directly:

```
VaR_HS(α) = percentile(portfolio_returns[-504:], (1-α) × 100)
CVaR_HS   = mean(returns | returns ≤ VaR_HS)
```

RASR-GE uses a 504-day lookback (2 trading years) at three confidence levels: **95%, 99%, 99.5%**.

**Regime-conditioned HS (in crisis mode):** When the HMM detects a CRISIS regime, plain HS equally weights all 504 historical days — including many calm-period days that are no longer representative. RASR-GE conditions the simulation on crisis posterior probabilities:

```
w_t = P(crisis | observations_1:t)  from HMM
w_t = w_t / Σ w_t   (normalise to sum to 1)
```

Days with a high crisis posterior receive proportionally higher sampling probability, so the loss distribution reflects the current stressed environment more accurately. This is a form of **importance sampling** — the same 504 observations but reweighted by their regime relevance.

### 6.2 Value at Risk — Parametric (Gaussian)

Assumes portfolio returns are normally distributed:

```
VaR_param(α)  = μ + z_α × σ
CVaR_param(α) = μ - σ × φ(z_α) / (1 - α)
```

Where μ and σ are the sample mean and standard deviation of the lookback window, z_α is the α-quantile of the standard normal, and φ is the standard normal PDF.

**Inputs:** Dynamic correlation matrix (60-day rolling Pearson across all 50 stocks) and per-stock volatility from the same window. This means the parametric VaR updates daily as the correlation structure evolves — during a crisis, higher pairwise correlations translate directly into higher portfolio variance and therefore higher VaR.

**HS vs Parametric:** HS makes no distributional assumptions and captures fat tails empirically; it relies on having relevant historical precedents in the lookback window. Parametric is analytically tractable and extrapolates smoothly but understates tail risk when returns are non-normal (e.g. during crashes). Presenting both provides a range with upper and lower bounds on risk.

### 6.3 Basel III Traffic Light Backtesting

**The regulatory standard:** Basel III requires banks to validate their internal VaR models by comparing 99% VaR predictions against actual daily P&L over a rolling **250-trading-day** window (approximately one calendar year). Each day the actual loss exceeds the predicted VaR is called an **exception**.

**Implementation:**
```
For t in [window, T]:
    VaR_99 = percentile(returns[t-250:t], 1.0)   # 1% left tail
    if returns[t] < VaR_99:
        exception_count += 1
```

**Traffic light zones:**

| Exceptions in 250 days | Zone | Interpretation |
|------------------------|------|----------------|
| 0 – 4 | 🟢 GREEN | Model is accurate; at 99% confidence, ≤4 exceptions expected |
| 5 – 9 | 🟡 YELLOW | Model may be underestimating tail risk; regulatory scrutiny |
| 10+ | 🔴 RED | Model is failing; regulator may require capital add-on |

**Why this matters for this project:** The backtesting module validates that the HS and Parametric VaR estimates are not systematically too optimistic. If the model is correctly calibrated at 99%, it should generate exceptions in the GREEN zone. The dashboard displays the rolling exception count, zone, and plots each exception as a marked point on the VaR vs. realised return time series — exactly the output a risk manager would show a regulator.

### 6.4 Contagion-Adjusted CVA

**What CVA is:** Credit Valuation Adjustment is the market value of counterparty credit risk — the expected loss on a portfolio from counterparty defaults. Standard CVA uses static credit PDs. RASR-GE computes **contagion-adjusted CVA**: PD estimates that incorporate shock propagation through the GAT network.

**The formula:**
```
CVA[i] = EAD[i]  ×  LGD  ×  PD_updated[i]  ×  λ
```

Where:
- **EAD[i]** = Exposure at Default = weight[i] × total notional. The hypothetical exposure to firm i.
- **LGD** = Loss Given Default = 0.60 (fixed; implies 40% recovery rate, a standard Basel assumption).
- **PD_updated[i]** = distress score after GAT shock propagation, not the pre-shock baseline.
- **λ** = HMM-derived regime stress multiplier (see Section 8.2).

**What makes this "contagion-adjusted":** The PD used is not a static credit rating or a single-firm model output. It is the GAT model's distress score *after* a shock has been propagated through the network. So CVA[i] captures not just firm i's own risk but its **network-inherited risk** — the additional distress probability it carries because of its connections to shocked counterparties. A firm with no direct connection to the shocked firm but two hops away via highly-correlated intermediaries will see its PD_updated elevated above its baseline, and therefore its CVA rises.

**ΔCVA** is the key regulatory output:
```
ΔCVA[i] = CVA_updated[i] - CVA_baseline[i]
```

This directly quantifies how much additional credit reserve a portfolio holder would need to set aside due to a specific stress scenario — per firm, aggregated across the portfolio.

---

## 7. Systemic Shock Propagation and Contagion Hubs

*This section maps to Resume Bullet 2: "Modeled financial shock propagation using Graph Attention Networks to quantify how credit risk transmits across interconnected firms, revealing sector-wise differences in systemic risk transmission and major contagion hubs."*

### 7.1 Shock Injection Mechanics

When a firm is shocked at magnitude s (e.g. −40%):

**Step 1 — Convert to log return:**
```
shock_log_ret = log(1 + s) = log(0.60) ≈ −0.5108
```

**Step 2 — Perturb last-day feature only:**
```
X_shocked[target_firm, day=29, feature=0] += shock_log_ret
```
Only the log_return on the last of the 30 lookback days is modified. All other 29 days, all other features, and all other firms' sequences are unchanged.

**Step 3 — Frozen forward pass:**
No retraining occurs. This is a pure forward pass through the already-trained network with one modified input. The propagation patterns reflect what the model learned from 2015–2023 historical market dynamics.

**Step 4 — LSTM re-encodes the shocked firm's hidden state:**
```
h_shocked = LSTM(X_shocked)
```
This hidden state now encodes a firm that has just experienced a severe return shock. All other firms' hidden states remain identical to baseline.

**Step 5 — GAT propagates through the network:**
The GAT layers receive the full set of hidden states, with only one changed. During message passing, every firm connected to the shocked firm receives a different message. The distress signal diffuses across **two hops**: first to direct neighbours (Layer 1), then to their neighbours (Layer 2), creating a two-hop contagion radius.

**Step 6 — Compute ΔPD:**
```
ΔPD[i] = PD_shocked[i] - PD_baseline[i]
```

Firms with the largest positive ΔPD are the contagion victims — those whose distress probability increased most through network propagation, with no change to their own input data.

### 7.2 Sector-Wise Differences in Systemic Risk Transmission

The GAT model — trained solely on market correlation data — **empirically recovers sector-level transmission patterns** without any pre-programmed sector classifications. This emerges from the structure of rolling correlations:

**Banking sector cluster:** HDFCBANK, ICICIBANK, AXISBANK, KOTAKBANK, SBIN consistently appear as top contagion victims when a large index constituent is shocked, regardless of which firm is the source. This reflects their shared exposure mechanisms:
- Major corporate borrowers (like Reliance) carry credit facilities running into hundreds of billions of rupees with these banks. An equity shock signals potential credit stress.
- These banks hold index-tracking portfolios; a large-cap shock triggers risk-management responses (stop-losses, reduced risk appetite) across all of them simultaneously.
- Their mutual correlations during the 2020 COVID crash and 2022 global selloff were consistently above the 0.60 threshold, creating a densely connected banking sub-graph.

**IT sector behaviour:** Large IT firms (INFY, TCS, WIPRO) show a different transmission pattern — they appear as victims of large-cap shocks primarily through **index heavyweight synchrony** (both IT stocks and the shocked firm are top-10 Nifty 50 constituents, creating correlated institutional selling) rather than direct credit linkages. Their ΔPD values tend to be larger in magnitude but of shorter duration than banking contagion.

**Defensive sectors (FMCG, Pharma):** Stocks like HINDUNILVR, NESTLEIND, SUNPHARMA consistently show lower ΔPD across shock scenarios. Their rolling correlations with financials and industrials typically fall near or below the 0.60 edge-creation threshold, resulting in fewer or weaker edges to shock sources. The model correctly identifies them as lower contagion receivers — consistent with their role as traditional safe havens during market stress.

These sector-wise patterns are **emergent results of training on historical data**, not pre-programmed rules.

### 7.3 Major Contagion Hubs: SIFI Ranking

Systemically Important Firms (SIFIs) are firms whose distress would have outsized systemic impact. RASR-GE quantifies this from the **aggregate outgoing GAT attention weight** across the entire validation period:

```
Spillover_Score[i] = (1/T) Σ_{t=1}^{T}  Σ_{j ∈ N(i)} α_{ij,t}
```

For every trading day in the validation set, for every firm, sum all outgoing attention weights — how strongly this firm's current state is being routed to its neighbours. Average across all T validation days.

A high Spillover Score means: across all market conditions in the out-of-sample period, this firm's state was consistently routed outward to its neighbours with high attention weight. It is a **data-driven, time-averaged measure of systemic importance** — not based on size, sector, or regulatory designation alone.

This provides a quantitative basis for the regulatory G-SIB/D-SIB (Global/Domestic Systemically Important Bank) identification process under Basel III.

---

## 8. Regime-Aware Stress Testing

*This section maps to Resume Bullet 3: "regime-aware stress testing, demonstrating that identical shocks cause ~23x more contagion during crisis network conditions, with per-firm risk attribution reports."*

### 8.1 HMM Regime Detection

A **2-state Gaussian Hidden Markov Model** is fitted to the log-transformed rolling average correlation density of the Nifty 50 network:

```
feature_t = log( mean(|ρ_{ij,t}| for all edges in A_t) + ε )
```

This scalar summarises the "connectedness" of the network on day t. It rises sharply during crises (correlation convergence) and is low during calm periods.

The HMM is fitted on 2015–2021 data (in-sample) and inferred on the full timeline including 2022–2025 (out-of-sample). The crisis state is identified as the state with the **higher mean log-density** — corresponding to the densely-connected, high-correlation regime.

**Outputs per day:**
- `state_sequence[t]` — hard assignment: CRISIS (1) or NORMAL (0)
- `posteriors[t, crisis_state]` — soft probability of being in the crisis regime

The posteriors are used to importance-weight the Historical Simulation VaR, ensuring that when the system is in or near a crisis regime, recent return observations receive proportionally higher sampling weight.

### 8.2 The λ Stress Multiplier

λ is the ratio of mean distress scores during crisis days vs normal days, computed on the in-sample period:

```
λ = mean(PD_baseline[t] for crisis days) / mean(PD_baseline[t] for normal days)
```

**Economic meaning:** λ captures how much more elevated average firm distress is during a crisis regime compared to normal conditions — a direct measure of the regime's severity. λ is used to scale CVA:

```
CVA[i] = EAD[i] × LGD × PD_updated[i] × λ
```

In crisis mode, CVA is scaled up by λ, reflecting that identical credit exposures carry more systemic risk when the overall network is in a crisis state. This is the quantitative mechanism by which regime information enters the CVA computation.

### 8.3 The Crisis Multiplier: ~23x More Contagion

**The key empirical finding:** An identical −40% shock applied to the COVID-peak crisis network (2020-03-23) causes **~23.3× more contagion** than the same shock applied to a calm 2017 network (2017-06-01).

**Methodology:**
For every firm as shock target (all 50 firms), compute mean ΔPD across all *other* 49 firms after shock propagation. Average across all 50 possible shock targets:

```
contagion(snapshot) = (1/N) Σ_{i=1}^{N} [ (1/(N-1)) Σ_{j≠i} ΔPD_j(shocked_at_i) ]
```

| Snapshot | Date | Avg ΔPD per firm |
|----------|------|-----------------|
| COVID Crisis | 2020-03-23 | 13.179% |
| Calm Baseline | 2017-06-01 | 0.565% |
| **Ratio** | | **~23.3×** |

**Why the multiplier is so large:** During the 2020 crisis, the correlation graph was near-fully-connected (crisis-induced densification). Every shock source had edges to almost every other firm with high weights. The GAT's two-hop propagation therefore reaches virtually every node, each receiving a strong message. In the 2017 calm network, the sparse graph means most firms receive no message at all from a given shock source, and those that do receive weak-weight messages. The architecture of the dynamic graph — automatically reflecting the market regime — is entirely responsible for this amplification. No crisis flag was hard-coded: the model uses the same weights on both snapshots.

**Interview-ready explanation:** *"I applied the same −40% shock to the same model but two different network snapshots — COVID crisis vs 2017 calm. The only difference is the graph topology, which the rolling correlation automatically reflects. In crisis mode, the graph was nearly fully connected, so the GAT routed the distress signal to almost every firm. In calm mode, the sparse graph meant most firms were unreachable. Averaging over all 50 possible shock targets, the crisis network produced 23× more cross-firm contagion than the calm network."*

### 8.4 Per-Firm Risk Attribution

For every stress test run, RASR-GE produces a full per-firm attribution table:

| Field | Meaning |
|-------|---------|
| **Ticker** | Firm identifier |
| **EAD** | Exposure at Default (weight × notional) |
| **PD_baseline** | Pre-shock distress probability |
| **PD_updated** | Post-shock distress probability (network-propagated) |
| **CVA_baseline** | Credit valuation adjustment before shock |
| **CVA_post_shock** | Credit valuation adjustment after shock |
| **ΔCVA** | Incremental CVA from the shock event |

Sorted by ΔCVA descending — the firms requiring the most additional credit reserves due to the shock are listed first. This is the **direct stakeholder output**: a risk manager can present this table to show exactly which counterparty exposures have increased in value after a stress scenario, by exactly how much, and why (network-propagated PD increase).

The **Attention Weights tab** supplements this with incoming and outgoing attention edges per firm — showing which specific counterparty connections are carrying the distress signal. This provides the "why" behind each ΔCVA, which is the interpretability requirement under Basel III Pillar 2.

---

## 9. Dashboard

The RASR-GE Streamlit dashboard has **5 tabs**, each corresponding to a distinct analytical output:

### Tab 1 — Network & Shock

**Always visible:** The crisis multiplier banner showing the ~23x finding (computed at load time across all 50 shock targets). This is the headline systemic risk finding.

**Baseline mode:** Top 5 firms by current distress probability. Network graph with node size proportional to PD, colour on green→red gradient, top 8% of edges by GAT attention weight rendered.

**Shock mode (after pressing Run Stress Test):** Top 5 contagion victims by ΔPD with delta metrics. Network graph with red directed edges from shocked firm to top victims, edge thickness proportional to attention weight × 70. Spillover Severity horizontal bar chart ranked by ΔPD.

**Node colouring:**
```
scaled = clip(PD_i / max_PD_today × 1.5, 0, 1)
colour = RGB(255×scaled, 255×(1-scaled), 0)
```
The scale adapts to today's risk distribution — the reddest node is always the riskiest *relative to today*, not a fixed scale.

### Tab 2 — VaR

VaR table at 95%, 99%, 99.5% for HS and Parametric methods. Basel III traffic light zone, exception count, total days. Rolling 99% VaR vs realised return time series with exceptions marked.

### Tab 3 — CVA

Per-firm CVA table (EAD, PD_baseline, PD_updated, CVA_baseline, CVA_post_shock, ΔCVA). Aggregate metrics (total baseline CVA, total post-shock CVA, total ΔCVA). Regime and λ displayed.

### Tab 4 — Model Calibration

AUROC, PRAUC, Brier Score, KS Statistic, F1, Precision, Recall. 10-bin reliability diagram comparing predicted scores to actual positive rates.

### Tab 5 — Attention Weights

For any selected firm: top-10 incoming attention edges (which firms are most influencing this firm's risk) and top-10 outgoing attention edges (which firms this firm is most influencing). The quantitative basis for the "per-firm risk attribution" claim in the resume.

---

## 10. Key Innovations vs Traditional Models

| Dimension | GARCH / VaR | Factor Models | Static Network | **RASR-GE** |
|-----------|------------|---------------|----------------|------------|
| Unit of analysis | Single firm | Firm vs. factors | Network | **Dynamic network** |
| Temporal dynamics | Volatility clustering | Static betas | None | **LSTM (30-day sequential)** |
| Cross-firm contagion | Not modelled | Factor co-exposure | Fixed adjacency | **Dynamic GAT over rolling correlation graph** |
| Graph structure | N/A | N/A | Fixed | **Rolling 60-day Pearson, updates daily** |
| Attention | N/A | N/A | Equal-weight | **Learned, edge-weighted α_{ij}** |
| Edge features | N/A | N/A | Binary | **Correlation weight ρ_{ij} conditions attention** |
| Shock simulation | Parametric | Factor shocks | Cascading threshold | **Counterfactual forward pass, frozen model** |
| Regime adaptation | Volatility switching | Fixed loadings | None | **Graph auto-densifies in crises; HMM λ multiplier** |
| VaR method | Standard HS | N/A | N/A | **Regime-conditioned importance-weighted HS + Parametric** |
| Credit risk | Not modelled | N/A | N/A | **Contagion-adjusted CVA with network-propagated PD** |
| Interpretability | High (parametric) | Medium | Medium | **Attention weights = explicit economic channels** |

---

## 11. Regulatory Alignment: Basel III

### 11.1 Stress Testing (Pillar 2)

Basel III requires stress tests analysing how adverse scenarios would affect capital adequacy. RASR-GE's shock simulation directly generates the **network contagion component** of such stress tests — not just "how does a firm shock affect that firm?" but the second-order: "how does that shock propagate through correlation linkages to affect the rest of the portfolio?" This is precisely the analysis regulatory bodies have called for since the 2008 post-mortems.

### 11.2 VaR Model Validation (Market Risk, Pillar 1)

The Basel III internal models approach requires banks to backtest their VaR models using the traffic light framework. The VaR engine implements this exactly: 250-day rolling window, 99% confidence level, GREEN/YELLOW/RED zones. A model producing GREEN zone results has demonstrated statistical validity of its tail risk estimates.

### 11.3 SIFI Identification (G-SIB / D-SIB Frameworks)

The SIFI spillover ranking provides a quantitative basis for the G-SIB/D-SIB identification process. A firm with a high spillover score has demonstrated, across an out-of-sample period, that its state is consistently routed outward through the network with high attention weights — a data-driven systemic importance measure independent of size or regulatory designation.

### 11.4 Countercyclical Capital Buffers

Basel III's countercyclical buffer requires additional capital during credit booms. The dynamic correlation graph provides a direct early-warning signal: as graph density rises (correlations converging above 0.60 market-wide), the system is entering a high-systemic-risk regime. Monitoring graph density over time provides a macroprudential early-warning indicator that activates before distress scores themselves rise.

### 11.5 Model Interpretability (Pillar 2 Model Risk)

Basel III requires quantitative risk models to be validated, documented, and interpretable. The GAT attention weights provide this: regulators can ask "why is firm X flagged as a contagion victim?" and receive a quantifiable answer — attention weights show exactly which channels carry the distress signal, and the correlation graph shows the edge weights enabling those channels. This is a meaningful improvement over black-box models producing a single score with no factor decomposition.

---

## Summary

RASR-GE shifts risk assessment from firm-centric to network-centric by combining:

- **Wavelet denoising** — signal quality upstream of all analytics
- **Dynamic correlation graphs** — regime-sensitive connectivity that auto-densifies in crises
- **LSTM temporal encoding** — sequential pattern recognition per firm
- **GAT with edge-weighted propagation** — learned, interpretable contagion simulation
- **Regime-conditioned VaR** — HS importance-weighted by HMM crisis posteriors
- **Contagion-adjusted CVA** — per-firm credit reserves incorporating network-propagated distress
- **Basel III backtesting** — regulatory validation of VaR model quality
- **SIFI ranking** — data-driven systemic importance from aggregate attention spillover
- **Crisis multiplier (~23x)** — empirical quantification of network amplification during crises

The system produces risk assessments that are accurate (Val AUROC 0.82+), regulatory-aligned (Basel III stress testing, VaR backtesting, SIFI identification), and economically meaningful — with every output traceable to a specific computation, not a manually tuned rule.

---

*Document version: 2.0 | RASR-GE | Updated: 2026-04-15*
