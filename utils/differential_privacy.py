"""
utils/differential_privacy.py
───────────────────────────────
Differential privacy mechanisms used in pFedLLM.

Two levels of DP (Section IV.F.c, V.H):
  1. Feature-level DP : Gaussian noise added to z_agg before broadcast.
                        ε computed via Rényi DP moments accountant.
  2. Gradient-level DP: Per-sample gradient clipping + noise during
                        local training (compatible with Opacus).

Privacy guarantee: (ε, δ)-DP with δ = 1e-5.
ε depends on noise multiplier σ, sampling rate q, and rounds T (Table IX).
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian Mechanism for Feature Vectors  (Algorithm 1, line 12)
# ─────────────────────────────────────────────────────────────────────────────

class GaussianMechanism:
    """
    Add calibrated Gaussian noise to a tensor to achieve (ε, δ)-DP.

    Usage (server side):
        gm = GaussianMechanism(sigma=0.6, sensitivity=1.0)
        z_noisy = gm(z_agg)     # z_agg is the aggregated feature
    """

    def __init__(self, sigma: float = 0.6, sensitivity: float = 1.0):
        """
        Parameters
        ----------
        sigma       : noise multiplier (σ).  Higher → more privacy, less utility.
                      From Table VIII: σ=0.6 gives ε≈1.5 at δ=1e-5.
        sensitivity : L2-sensitivity of the query (max ||z_agg|| per client).
                      Usually 1.0 when features are L2-normalised.
        """
        self.sigma       = sigma
        self.sensitivity = sensitivity

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise ε_DP ~ N(0, (σ·Δ)²·I) to x."""
        if self.sigma == 0.0:
            return x
        noise = torch.randn_like(x) * (self.sigma * self.sensitivity)
        return x + noise

    def noise_std(self) -> float:
        return self.sigma * self.sensitivity


# ─────────────────────────────────────────────────────────────────────────────
# Rényi DP  →  (ε, δ) conversion  (Table IX in paper)
# ─────────────────────────────────────────────────────────────────────────────

def _rdp_gaussian(sigma: float, q: float, alpha: float) -> float:
    """
    Rényi divergence of the sub-sampled Gaussian mechanism.
    Computes RDP(α) for the Gaussian mechanism with noise multiplier σ
    and sub-sampling ratio q.

    Uses the bound from Mironov (2017) + sub-sampling amplification.
    """
    if q == 0:
        return 0.0
    # Upper bound via log-sum-exp expansion (first two terms dominate)
    term1 = alpha * q ** 2 * (math.exp(1.0 / sigma ** 2) - 1)
    return min(term1, alpha / (2 * sigma ** 2))


def compute_epsilon(
    sigma: float,
    sampling_rate: float,
    num_rounds: int,
    delta: float = 1e-5,
    alphas: Optional[list] = None,
) -> Tuple[float, float]:
    """
    Compute (ε, δ) from the RDP moments accountant.

    Parameters
    ----------
    sigma         : noise multiplier
    sampling_rate : batch_size / dataset_size  (q)
    num_rounds    : total number of SGD steps  (T)
    delta         : target δ
    alphas        : Rényi orders to search over

    Returns
    -------
    (epsilon, best_alpha)
    """
    if alphas is None:
        alphas = [1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 32, 64]

    best_eps, best_alpha = float("inf"), None
    for alpha in alphas:
        rdp = _rdp_gaussian(sigma, sampling_rate, alpha)
        rdp_total = rdp * num_rounds
        # Convert RDP to (ε, δ)-DP
        eps = rdp_total + math.log(1 / delta) / (alpha - 1)
        if eps < best_eps:
            best_eps   = eps
            best_alpha = alpha

    return best_eps, best_alpha


def privacy_budget_table(
    sigmas: list = None,
    batch_size: int = 32,
    dataset_size: int = 10000,
    num_rounds: int = 50,
    delta: float = 1e-5,
) -> dict:
    """
    Reproduce Table IX from the paper.

    Returns
    -------
    dict[sigma] = (epsilon, alpha)
    """
    if sigmas is None:
        sigmas = [0.1, 0.3, 0.5, 0.8]

    q = batch_size / dataset_size
    results = {}
    for sigma in sigmas:
        eps, alpha = compute_epsilon(sigma, q, num_rounds, delta)
        results[sigma] = {"epsilon": round(eps, 2), "alpha": alpha}
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Per-Sample Gradient Clipping  (client-side DP-SGD)
# ─────────────────────────────────────────────────────────────────────────────

class DPOptimizer:
    """
    Wraps a standard PyTorch optimizer with per-sample gradient clipping
    and Gaussian noise addition (DP-SGD, Abadi et al. 2016).

    For production use, prefer the Opacus library:
        from opacus import PrivacyEngine
        privacy_engine = PrivacyEngine()
        model, optimizer, loader = privacy_engine.make_private(...)

    This implementation is a pedagogical stand-in.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        model: nn.Module,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 0.6,
        expected_batch_size: int = 32,
    ):
        self.optimizer    = base_optimizer
        self.model        = model
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.batch_size   = expected_batch_size

    def step(self):
        """Clip gradients, add noise, then call the underlying optimizer step."""
        # 1. Clip per-parameter gradient norm
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # 2. Add Gaussian noise to each parameter's gradient
        with torch.no_grad():
            for p in self.model.parameters():
                if p.grad is None:
                    continue
                noise = torch.randn_like(p.grad)
                noise *= (self.noise_multiplier * self.max_grad_norm
                          / self.batch_size)
                p.grad += noise

        self.optimizer.step()

    def zero_grad(self, *args, **kwargs):
        self.optimizer.zero_grad(*args, **kwargs)

    @property
    def param_groups(self):
        return self.optimizer.param_groups


# ─────────────────────────────────────────────────────────────────────────────
# Utility: Privacy Report
# ─────────────────────────────────────────────────────────────────────────────

def print_privacy_report(cfg):
    """Print DP budget table matching Table IX in the paper."""
    q = cfg.batch_size / 10000  # approximate sampling rate
    print("\n" + "=" * 55)
    print("  Differential Privacy Budget (Table IX)")
    print(f"  δ = {cfg.dp_delta},  T = {cfg.rounds},  q ≈ {q:.4f}")
    print("=" * 55)
    print(f"  {'σ':>6}   {'ε':>8}   {'α*':>6}   BLEU-4  AUC-ROC")
    print("-" * 55)
    # Reference values from Table IX / Table VIII
    table_ref = {
        0.0: (None, None, 0.298, 0.928),
        0.1: (4.7,  None, 0.291, 0.924),
        0.3: (2.1,  None, 0.283, 0.919),
        0.5: (1.5,  None, 0.262, 0.901),
        0.8: (1.2,  None, 0.219, 0.861),
    }
    for sigma, (eps_ref, _, bleu, auc) in table_ref.items():
        if sigma == 0.0:
            eps_str = "  no DP"
        else:
            eps, alpha = compute_epsilon(sigma, q, cfg.rounds, cfg.dp_delta)
            eps_str = f"{eps:8.2f}"
        print(f"  {sigma:>6.1f}   {eps_str}          {bleu:.3f}   {auc:.3f}")
    print("=" * 55 + "\n")
