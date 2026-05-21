"""
VaR Engine — Historical Simulation, Parametric, and Monte Carlo VaR
with Basel III Traffic Light backtesting and regime-conditioned HS.
"""

import numpy as np
from scipy import stats


class VaREngine:
    """Computes portfolio VaR using three methods at multiple confidence levels."""

    def __init__(self, config):
        self.confidence_levels = config['var']['confidence_levels']
        self.mc_simulations = config['var']['mc_simulations']
        self.lookback_days = config['var']['lookback_days']
        self.basel_window = config['var']['basel_window']

    # ================================================================ core
    def compute_all(self, portfolio_returns, weights, corr_matrix, per_stock_vol,
                    crisis_posteriors=None, current_regime='NORMAL'):
        """
        Parameters
        ----------
        portfolio_returns : np.ndarray, shape (T,)
            Historical portfolio return series (weighted sum of stock returns).
        weights : np.ndarray, shape (N,)
            Portfolio weights (sum to 1).
        corr_matrix : np.ndarray, shape (N, N)
            Dynamic correlation matrix for the selected window.
        per_stock_vol : np.ndarray, shape (N,)
            Per-stock annualised volatility for the same window.
        crisis_posteriors : np.ndarray or None, shape (lookback,)
            HMM crisis posterior probabilities for importance weighting.
        current_regime : str
            'NORMAL' or 'CRISIS' — drives whether HS uses importance weighting.
        """
        lookback = min(self.lookback_days, len(portfolio_returns))
        recent = portfolio_returns[-lookback:]

        results = {}
        for cl in self.confidence_levels:
            hs  = self._historical_sim(recent, cl, crisis_posteriors, current_regime)
            par = self._parametric(recent, cl)
            mc  = self._monte_carlo(weights, corr_matrix, per_stock_vol, cl)
            results[cl] = {'HS': hs, 'Parametric': par, 'MC': mc}
        return results

    # ----------------------------------------------------- Historical Sim
    def _historical_sim(self, returns, confidence, crisis_posteriors, regime):
        alpha = 1.0 - confidence
        if regime == 'CRISIS' and crisis_posteriors is not None:
            # Importance-weighted draw
            w = crisis_posteriors[-len(returns):]
            if len(w) < len(returns):
                pad = np.ones(len(returns) - len(w)) * w.mean()
                w = np.concatenate([pad, w])
            w = w / w.sum()
            sampled = np.random.choice(returns, size=10_000, p=w, replace=True)
            var = float(np.percentile(sampled, alpha * 100))
            cvar = float(sampled[sampled <= var].mean()) if (sampled <= var).any() else var
            label = 'HS (Crisis-weighted)'
        else:
            var = float(np.percentile(returns, alpha * 100))
            cvar = float(returns[returns <= var].mean()) if (returns <= var).any() else var
            label = 'HS'
        return {'VaR': var, 'CVaR': cvar, 'label': label}

    # --------------------------------------------------------- Parametric
    def _parametric(self, returns, confidence):
        mu = returns.mean()
        sigma = returns.std()
        alpha = 1.0 - confidence
        z = stats.norm.ppf(alpha)
        var = float(mu + z * sigma)
        # CVaR for normal: mu - sigma * phi(z)/alpha
        cvar = float(mu - sigma * stats.norm.pdf(z) / alpha)
        return {'VaR': var, 'CVaR': cvar, 'label': 'Parametric'}

    # -------------------------------------------------------- Monte Carlo
    def _monte_carlo(self, weights, corr_matrix, per_stock_vol, confidence):
        n = len(weights)
        alpha = 1.0 - confidence

        # Build covariance from correlation + volatility
        D = np.diag(per_stock_vol)
        cov = D @ corr_matrix @ D

        # Ensure positive semi-definite
        eigvals = np.linalg.eigvalsh(cov)
        if eigvals.min() < 0:
            cov += np.eye(n) * (abs(eigvals.min()) + 1e-8)

        # Cholesky + simulate
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # Fallback: eigenvalue repair
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-8)
            cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
            L = np.linalg.cholesky(cov)

        z = np.random.randn(self.mc_simulations, n)
        sim_returns = z @ L.T  # (M, N)
        portfolio_sim = sim_returns @ weights

        var = float(np.percentile(portfolio_sim, alpha * 100))
        cvar = float(portfolio_sim[portfolio_sim <= var].mean()) if (portfolio_sim <= var).any() else var
        return {'VaR': var, 'CVaR': cvar, 'label': 'MC (dynamic corr)'}

    # ================================================ Basel III Backtesting
    def basel_backtest(self, portfolio_returns, weights, corr_matrix, per_stock_vol):
        """
        Count VaR exceptions over a rolling 250-day window.
        Returns exception count and traffic light zone.
        """
        T = len(portfolio_returns)
        window = min(self.basel_window, T)
        if T < window + 10:
            return {'exceptions': 0, 'zone': 'GREEN', 'total_days': 0}

        exception_days = []
        var_series = []
        for t in range(window, T):
            lookback = portfolio_returns[t - window:t]
            var_99 = float(np.percentile(lookback, 1.0))  # 99% VaR
            var_series.append(var_99)
            if portfolio_returns[t] < var_99:
                exception_days.append(t)

        # Total exceptions across full history (for display)
        exceptions_total = len(exception_days)

        # Basel III zone: count exceptions only in the most recent 250-day window
        recent_start = T - window
        exceptions_recent = sum(1 for d in exception_days if d >= recent_start)

        if exceptions_recent <= 4:
            zone = 'GREEN'
        elif exceptions_recent <= 9:
            zone = 'YELLOW'
        else:
            zone = 'RED'

        return {
            'exceptions': exceptions_recent,
            'exceptions_total': exceptions_total,
            'zone': zone,
            'total_days': T - window,
            'exception_days': exception_days,
            'var_series': var_series
        }
