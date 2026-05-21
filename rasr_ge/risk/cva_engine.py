"""
Contagion-Adjusted CVA Engine
-------------------------------
Computes instantaneous point-in-time CVA per firm using GAT-transmitted
distress scores and the HMM-derived regime stress multiplier λ.
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


class CVAEngine:
    """
    CVA[i] = EAD[i] × LGD × DistressScore_updated[i] × λ

    Assumptions (stated prominently in UI):
    - Instantaneous point-in-time. No discount curve, no term structure.
    - Entity: hypothetical equal-notional portfolio holder.
    - LGD: fixed 0.6 (40% recovery). Not user-editable.
    - PD: ordinal distress score, not a Basel credit PD.
    """

    def __init__(self, config):
        self.lgd = config['cva']['lgd']
        self.default_notional = config['cva']['default_notional']

    def compute(self, tickers, weights, baseline_scores, attention_weights_matrix,
                shock_vector, lambda_=1.0, notional=None):
        """
        Parameters
        ----------
        tickers : list[str]
            Firm names.
        weights : np.ndarray, shape (N,)
            Portfolio weights summing to 1.
        baseline_scores : np.ndarray, shape (N,)
            DistressScore_baseline[i] ∈ [0, 1].
        attention_weights_matrix : np.ndarray, shape (N, N)
            Attention weight from firm j→i.  Row i = weights incoming to i.
            attention_weights_matrix[i, j] = α_ij.
        shock_vector : np.ndarray, shape (N,)
            Shock magnitude per firm (0 for unshocked, e.g. −0.40 for shocked firm).
        lambda_ : float
            HMM-derived regime stress multiplier.
        notional : float or None
            Total notional (defaults to config).

        Returns
        -------
        dict with keys: 'per_firm' (DataFrame-ready list of dicts), 'aggregate'.
        """
        if notional is None:
            notional = self.default_notional

        N = len(tickers)
        ead = weights * notional  # per-firm EAD

        # PD update via logit-space shift
        baseline_logits = logit(baseline_scores)
        # Weighted attention-shock contribution per firm
        shock_contribution = attention_weights_matrix @ shock_vector  # (N,)
        updated_logits = baseline_logits + shock_contribution
        updated_scores = sigmoid(updated_logits)

        # CVA computation
        cva_baseline = ead * self.lgd * baseline_scores * lambda_
        cva_updated = ead * self.lgd * updated_scores * lambda_
        delta_cva = cva_updated - cva_baseline

        per_firm = []
        for i in range(N):
            per_firm.append({
                'Ticker': tickers[i],
                'EAD': float(ead[i]),
                'PD_baseline': float(baseline_scores[i]),
                'PD_updated': float(updated_scores[i]),
                'CVA_baseline': float(cva_baseline[i]),
                'CVA_post_shock': float(cva_updated[i]),
                'ΔCVA': float(delta_cva[i]),
            })

        return {
            'per_firm': per_firm,
            'aggregate': {
                'CVA_baseline_total': float(cva_baseline.sum()),
                'CVA_post_shock_total': float(cva_updated.sum()),
                'ΔCVA_total': float(delta_cva.sum()),
                'lambda': lambda_,
            }
        }

    @staticmethod
    def build_attention_matrix(edge_index, att_weights, n_nodes):
        """
        Convert GAT edge_index + attention weights into a dense (N, N) matrix.

        Parameters
        ----------
        edge_index : np.ndarray, shape (2, E)
        att_weights : np.ndarray, shape (E,)
        n_nodes : int

        Returns
        -------
        np.ndarray, shape (N, N) where A[i, j] = α from j→i.
        """
        A = np.zeros((n_nodes, n_nodes))
        for e in range(edge_index.shape[1]):
            src = edge_index[0, e]
            dst = edge_index[1, e]
            A[dst, src] = att_weights[e]  # attention from src → dst stored at [dst, src]
        return A
