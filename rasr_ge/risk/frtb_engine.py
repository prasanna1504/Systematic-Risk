import numpy as np
import scipy.stats as stats

# Mock sector map for NIFTY 50 constituents for FRTB Bucketing
SECTOR_MAP = {
    "RELIANCE.NS": "Energy", "TCS.NS": "IT", "HDFCBANK.NS": "Financials", "ICICIBANK.NS": "Financials", "INFY.NS": "IT",
    "ITC.NS": "Consumer", "SBIN.NS": "Financials", "BHARTIARTL.NS": "Telecom", "BAJFINANCE.NS": "Financials", "LT.NS": "Industrials",
    "KOTAKBANK.NS": "Financials", "HCLTECH.NS": "IT", "ASIANPAINT.NS": "Consumer", "AXISBANK.NS": "Financials", "MARUTI.NS": "Consumer",
    "SUNPHARMA.NS": "Healthcare", "TITAN.NS": "Consumer", "ULTRACEMCO.NS": "Materials", "TATAMOTORS.NS": "Consumer", "BAJAJFINSV.NS": "Financials",
    "WIPRO.NS": "IT", "NESTLEIND.NS": "Consumer", "M&M.NS": "Consumer", "POWERGRID.NS": "Utilities", "JSWSTEEL.NS": "Materials",
    "TATASTEEL.NS": "Materials", "NTPC.NS": "Utilities", "ADANIENT.NS": "Industrials", "ONGC.NS": "Energy", "GRASIM.NS": "Materials",
    "HINDUNILVR.NS": "Consumer", "TECHM.NS": "IT", "HINDALCO.NS": "Materials", "DIVISLAB.NS": "Healthcare", "ADANIPORTS.NS": "Industrials",
    "CIPLA.NS": "Healthcare", "DRREDDY.NS": "Healthcare", "BRITANNIA.NS": "Consumer", "APOLLOHOSP.NS": "Healthcare", "EICHERMOT.NS": "Consumer",
    "COALINDIA.NS": "Materials", "TATACONSUM.NS": "Consumer", "HEROMOTOCO.NS": "Consumer", "BAJAJ-AUTO.NS": "Consumer",
    "UPL.NS": "Materials", "BPCL.NS": "Energy", "INDUSINDBK.NS": "Financials", "SHREECEM.NS": "Materials", "HDFCLIFE.NS": "Financials",
    "SBILIFE.NS": "Financials"
}

class FRTBEngine:
    """
    Computes Standardised Approach (SA) FRTB Capital Charge using the 
    Sensitivity-Based Method (SBM) for Equity Risk.
    """
    def __init__(self, config=None):
        # Basel III Standardised Approach parameters (Simplified)
        self.risk_weight = 0.40 # Base equity risk weight (40%)
        self.rho = 0.25 # Intra-bucket correlation
        self.gamma = 0.15 # Inter-bucket correlation

    def compute_greeks(self, spots, vols, r=0.05, t=1.0):
        """
        Calculates Delta and Vega for a synthetic At-The-Money (ATM) call option
        portfolio using Black-Scholes.
        """
        # Avoid division by zero
        vols = np.clip(vols, 1e-6, None)
        
        # d1 = (ln(S/K) + (r + sigma^2/2)t) / (sigma * sqrt(t))
        # Since it's ATM, S=K, so ln(S/K) = 0
        d1 = (r + 0.5 * vols**2) * t / (vols * np.sqrt(t))
        
        # Delta of a Call option = N(d1)
        delta = stats.norm.cdf(d1)
        
        # Vega = S * sqrt(t) * N'(d1)
        vega = spots * np.sqrt(t) * stats.norm.pdf(d1)
        
        return delta, vega

    def compute_sbm_capital(self, tickers, weights, notional, spots, vols, r=0.05, t=1.0):
        """
        Aggregates sensitivities into buckets and computes the final FRTB Capital Charge.
        """
        delta, vega = self.compute_greeks(spots, vols, r, t)
        
        # Calculate Delta Sensitivities (s_k)
        # Simplified: s_k = Delta * Exposure * Risk_Weight
        exposures = weights * notional
        s_k = delta * exposures * self.risk_weight
        
        # 1. Bucket Aggregation
        buckets = {}
        for i, ticker in enumerate(tickers):
            sector = SECTOR_MAP.get(ticker, "Others")
            if sector not in buckets:
                buckets[sector] = []
            buckets[sector].append(s_k[i])
            
        Kb_dict = {}
        Sb_dict = {}
        
        for sector, sens in buckets.items():
            s = np.array(sens)
            Sb = np.sum(s)
            
            # K_b^2 = sum(s_k^2) + sum_k(sum_l(rho_kl * s_k * s_l))
            sum_sq = np.sum(s**2)
            cross_sum = 0
            for i in range(len(s)):
                for j in range(len(s)):
                    if i != j:
                        cross_sum += self.rho * s[i] * s[j]
                        
            Kb_sq = sum_sq + cross_sum
            Kb = np.sqrt(max(0, Kb_sq)) # Floor at 0
            
            Kb_dict[sector] = Kb
            Sb_dict[sector] = Sb
            
        # 2. Inter-bucket Aggregation
        sectors = list(Kb_dict.keys())
        total_sq = 0
        for i, b in enumerate(sectors):
            total_sq += Kb_dict[b]**2
            for j, c in enumerate(sectors):
                if i != j:
                    total_sq += self.gamma * Sb_dict[b] * Sb_dict[c]
                    
        total_capital = np.sqrt(max(0, total_sq))
        
        return {
            'total_capital': total_capital,
            'delta': delta,
            'vega': vega,
            'sensitivities': s_k,
            'buckets': Kb_dict
        }
