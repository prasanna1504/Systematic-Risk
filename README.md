# Systematic Risk & Market Quantification Engine (RASR-GE)

🟢 **Live Demo:** [View Dashboard on Streamlit Cloud](https://prasanna1504-systematic-risk-rasr-gedashboardapp-eme0zz.streamlit.app/)

## 📖 Overview

In the rapidly evolving landscape of quantitative finance and regulatory compliance, risk managers and quant traders need tools that go beyond basic historical simulations. They require systems that can handle the nuanced demands of Basel IV / FRTB (Fundamental Review of the Trading Book), model complex market regimes dynamically, and provide actionable, explainable intelligence during severe stress events.

**RASR-GE (Regime-Aware Systemic Risk and General Equilibrium)** was built to bridge the gap between advanced econometric modeling and practical regulatory capital requirements. 

We developed a comprehensive, production-grade **Market Risk Quantification System**. It acts as a unified engine designed to evaluate, explain, and mitigate risks across complex trading portfolios (specifically focusing on Indian Equities like the Nifty 50, but extensible to any asset class). The goal is to provide a unified platform where a risk manager can not only see their daily Value-at-Risk (VaR) but also instantly understand the regulatory capital impact of a hypothetical market crash.

## 🎯 Key Capabilities & Architecture

### 1. FRTB Standardised Approach (SA) Capital Calculations
At the core of modern market risk is the FRTB framework. This engine implements the **Sensitivities-Based Method (SBM)** to compute capital charges. It intelligently splits the portfolio into **Core** and **Non-Core** trading books, allowing for targeted risk analysis and accurate regulatory reporting.

### 2. Regime-Aware Risk Modeling (HMM)
Markets don't behave the same way in a crisis as they do in a bull run. Our system utilizes **Hidden Markov Models (HMM)** to continuously analyze market data and detect underlying macroeconomic regimes (e.g., low-volatility bull, high-volatility bear). Risk metrics like VaR and Expected Shortfall are dynamically adjusted based on the current detected regime.

### 3. Advanced "What-If" Stress Testing Pipeline
Risk management is about preparing for the worst. The system features a robust **Shock Engine** and **Counterfactual Analysis** module. Users can simulate severe market shocks (e.g., a sudden 20% drop in financials or an interest rate spike) and instantly visualize the resulting impact on the portfolio's PnL and its subsequent FRTB capital charge requirements.

### 4. SIFI Ranking & Interconnectedness
Understanding systemic risk means knowing which assets or institutions pose the greatest threat to the overall portfolio. The system calculates **Systemically Important Financial Institution (SIFI)** rankings to highlight vulnerabilities and interconnectedness within the holdings using Graph Neural Networks (GNNs).

### 5. Explainable AI (XAI)
When a model flags a massive spike in risk, stakeholders need to know *why*. The integrated **XAI Explainer** unboxes complex risk attributions using GNNExplainer, providing clear, human-readable explanations of which specific factors, assets, or regime shifts are driving up capital charges.

### 6. Interactive Streamlit Dashboard
All of these complex backend calculations are surfaced through a dynamic, interactive web application built with Streamlit. This allows risk managers to manipulate portfolios, run stress tests, and view capital impacts in real-time without writing a single line of code.

## 🛠️ Technical Stack
- **Language:** Python
- **Core Libraries:** PyTorch, PyTorch Geometric, Pandas, NumPy, Scikit-learn
- **Risk Models:** Graph Attention Networks (GAT), HMM (Regime Detection), Historical/Parametric VaR, CVA
- **Frontend/Deployment:** Streamlit, Plotly, Streamlit-Agraph

## 🚀 Getting Started (Local Development)

1. **Clone the repository**
   ```bash
   git clone https://github.com/prasanna1504/Systematic-Risk.git
   cd Systematic-Risk
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit Dashboard**
   ```bash
   streamlit run rasr_ge/dashboard/app.py
   ```

*(Note: The repository includes preprocessed data in `rasr_ge/data/processed/`. You can immediately run the dashboard to explore the system.)*
