"""
HMM Regime Detection Module
------------------------------
Fits a 2-state Gaussian HMM on log-transformed correlation density to detect
Normal vs Crisis market regimes.  Derives the stress multiplier λ used downstream
by the CVA engine.
"""

import os
import pickle
import yaml
import torch
import numpy as np
from hmmlearn.hmm import GaussianHMM


def _compute_correlation_density(graphs_dir, dates):
    """Average absolute pairwise correlation per day from stored adjacency matrices."""
    density = np.full(len(dates), np.nan)
    for t, dt in enumerate(dates):
        date_str = dt.strftime('%Y-%m-%d')
        path = os.path.join(graphs_dir, f'adj_{date_str}.pt')
        if not os.path.exists(path):
            continue
        g = torch.load(path, weights_only=False)
        ew = g['edge_weight']
        if ew.numel() == 0:
            density[t] = 0.0
        else:
            density[t] = float(ew.mean())
    return density


class RegimeDetector:
    """Wraps HMM fitting, inference, λ derivation, and serialisation."""

    def __init__(self, config, root_dir):
        self.config = config
        self.root_dir = root_dir
        self.graphs_dir = os.path.join(root_dir, 'data', 'graphs')
        self.hmm_cfg = config['hmm']
        self.n_states = self.hmm_cfg['n_states']

        # Will be populated after fit()
        self.hmm_model = None
        self.crisis_state = None
        self.normal_state = None
        self.lambda_ = 1.0
        self.density = None
        self.density_log = None
        self.state_sequence = None
        self.posteriors = None
        self.dates = None

    # ------------------------------------------------------------------ fit
    def fit(self, dates, distress_scores_baseline=None):
        """
        Parameters
        ----------
        dates : pandas DatetimeIndex or list of datetime
            Full date axis from the data pipeline.
        distress_scores_baseline : np.ndarray, optional, shape (N, T)
            Baseline DistressScore per node per day, used to derive λ.
            If None, λ stays at 1.0.
        """
        self.dates = dates
        density = _compute_correlation_density(self.graphs_dir, dates)

        # Forward-fill NaNs (days before graph window starts)
        mask_valid = ~np.isnan(density)
        if mask_valid.sum() == 0:
            raise ValueError("No valid correlation density values found.")
        first_valid = np.argmax(mask_valid)
        density[:first_valid] = density[first_valid]
        density = np.nan_to_num(density, nan=0.0)
        self.density = density

        # Log-transform (Gaussian emission assumption)
        self.density_log = np.log(density + 1e-6)

        # Determine fit window indices
        fit_start = self.hmm_cfg['fit_start']
        fit_end = self.hmm_cfg['fit_end']
        date_strs = [d.strftime('%Y-%m-%d') for d in dates]
        fit_mask = np.array([(fit_start <= d <= fit_end) for d in date_strs])
        fit_data = self.density_log[fit_mask].reshape(-1, 1)

        # Fit HMM
        self.hmm_model = GaussianHMM(
            n_components=self.n_states,
            covariance_type='full',
            n_iter=100,
            random_state=42
        )
        self.hmm_model.fit(fit_data)

        # Crisis state = state with higher mean log-density
        self.crisis_state = int(np.argmax(self.hmm_model.means_))
        self.normal_state = 1 - self.crisis_state

        # Full-history inference (including out-of-sample 2022+)
        full_data = self.density_log.reshape(-1, 1)
        self.state_sequence = self.hmm_model.predict(full_data)
        self.posteriors = self.hmm_model.predict_proba(full_data)

        # λ derivation from fit window only
        if distress_scores_baseline is not None:
            fit_states = self.state_sequence[fit_mask]
            # distress_scores_baseline should have shape (T,) — mean across nodes
            if distress_scores_baseline.ndim == 2:
                ds = distress_scores_baseline.mean(axis=0)
            else:
                ds = distress_scores_baseline
            ds_fit = ds[fit_mask]
            crisis_mask = fit_states == self.crisis_state
            normal_mask = fit_states == self.normal_state
            mean_crisis = ds_fit[crisis_mask].mean() if crisis_mask.any() else 1.0
            mean_normal = ds_fit[normal_mask].mean() if normal_mask.any() else 1.0
            self.lambda_ = float(mean_crisis / max(mean_normal, 1e-8))
        else:
            self.lambda_ = 1.5  # sensible default before distress scores available

        # Persist
        self.save()
        return self

    # --------------------------------------------------------------- query
    def get_regime(self, date_str):
        """Return regime label and crisis posterior for a single date."""
        date_strs = [d.strftime('%Y-%m-%d') for d in self.dates]
        if date_str in date_strs:
            idx = date_strs.index(date_str)
        else:
            # Find nearest date before
            idx = max(i for i, d in enumerate(date_strs) if d <= date_str)
        state = self.state_sequence[idx]
        label = "CRISIS" if state == self.crisis_state else "NORMAL"
        crisis_prob = float(self.posteriors[idx, self.crisis_state])
        return label, crisis_prob

    def get_crisis_posteriors(self, start_idx, end_idx):
        """Return crisis posterior array for a date window (used by VaR HS weighting)."""
        return self.posteriors[start_idx:end_idx, self.crisis_state]

    # --------------------------------------------------------- persistence
    def save(self):
        path = os.path.join(self.root_dir, 'checkpoints', 'hmm_params.pkl')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            'hmm_model': self.hmm_model,
            'crisis_state': self.crisis_state,
            'normal_state': self.normal_state,
            'lambda': self.lambda_,
            'density': self.density,
            'density_log': self.density_log,
            'state_sequence': self.state_sequence,
            'posteriors': self.posteriors,
            'dates': self.dates,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self):
        path = os.path.join(self.root_dir, 'checkpoints', 'hmm_params.pkl')
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.hmm_model = data['hmm_model']
        self.crisis_state = data['crisis_state']
        self.normal_state = data['normal_state']
        self.lambda_ = data['lambda']
        self.density = data['density']
        self.density_log = data['density_log']
        self.state_sequence = data['state_sequence']
        self.posteriors = data['posteriors']
        self.dates = data['dates']
        return self
