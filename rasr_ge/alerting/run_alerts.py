import os
import sys
import yaml
import torch
import numpy as np
from datetime import datetime

# Resolve project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from risk.hmm_regime import RegimeDetector
from risk.var_engine import VaREngine

def generate_alert_message(date, regime, var_99, actual_return, top_losers):
    msg = f"""
===================================================
🚨 SYSTEMIC RISK ALERT: MAJOR MOVE DETECTED 🚨
===================================================
Date: {date}
Active Market Regime: {regime}

⚠️ LIMIT BREACH DETAILS:
Portfolio Return: {actual_return:.2f}%
99% VaR Limit:    {var_99:.2f}%
Status:           BREACHED

📉 TOP CONTRIBUTORS TO LOSS:
"""
    for ticker, ret in top_losers:
        msg += f"- {ticker}: {ret:.2f}%\n"
        
    msg += """
ACTION REQUIRED:
Please review the FRTB Capital Charge impacts and CVA adjustments 
on the RASR-GE Dashboard immediately.
===================================================
"""
    return msg

def main():
    print("Running daily exception alerting script...")
    
    with open(os.path.join(ROOT_DIR, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    # Load Data
    features = torch.load(os.path.join(ROOT_DIR, 'data', 'processed', 'features.pt'), weights_only=False)
    meta = torch.load(os.path.join(ROOT_DIR, 'data', 'processed', 'meta.pt'), weights_only=False)
    tickers = meta['valid_tickers']
    dates = meta['dates']
    
    # We only look at the most recent day
    log_returns = features[:, :, 0].numpy() # (N, T)
    latest_returns = log_returns[:, -1] * 100 # In percentage
    
    # Compute Equal Weighted Portfolio Return
    weights = np.ones(len(tickers)) / len(tickers)
    portfolio_return = np.sum(latest_returns * weights)
    
    # Run Regime Detection
    hmm = RegimeDetector(config, ROOT_DIR)
    hmm.load()
    latest_date_str = dates[-1].strftime('%Y-%m-%d')
    regime_label, _ = hmm.get_regime(latest_date_str)
    
    # Compute VaR Limit
    var_engine = VaREngine(config)
    window = config['graph']['window']
    recent_history = log_returns[:, -window:]
    corr_matrix = np.corrcoef(recent_history)
    per_stock_vol = recent_history.std(axis=1)
    
    historical_port_returns = (weights[:, None] * log_returns).sum(axis=0) * 100
    
    # Get the 99% VaR for the day
    lookback = var_engine.lookback_days
    var_99 = float(np.percentile(historical_port_returns[-lookback-1:-1], 1.0))
    
    is_breach = portfolio_return < var_99
    is_crisis = regime_label == "CRISIS"
    
    if is_breach or is_crisis:
        print("\nAlert condition met! Generating report...\n")
        
        # Find top 3 losers
        loser_indices = np.argsort(latest_returns)[:3]
        top_losers = [(tickers[i], latest_returns[i]) for i in loser_indices]
        
        alert_msg = generate_alert_message(
            date=latest_date_str,
            regime=regime_label,
            var_99=var_99,
            actual_return=portfolio_return,
            top_losers=top_losers
        )
        print(alert_msg)
        
        # Here you could add smtplib to actually email it, 
        # or post to a Slack webhook via requests.post()
    else:
        print(f"No alerts today. (Return: {portfolio_return:.2f}% | VaR Limit: {var_99:.2f}% | Regime: {regime_label})")

if __name__ == "__main__":
    main()
