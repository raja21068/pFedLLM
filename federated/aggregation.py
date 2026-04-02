"""
federated/aggregation.py
─────────────────────────
Feature-map aggregation strategies for pFedLLM.

Algorithm 1, line 12:
    z_agg = Σ_c (|D_c| / |D|) · z_c  +  ε_DP

Three strategies are implemented (Section IV.B, IV.C.e):
  1. weighted_avg   — dataset-size-weighted mean (default, paper)
  2. attention      — learnable cosine-similarity-based weights
  3. uniform        — simple arithmetic mean (ablation baseline)

All strategies apply L2-normalisation before the DP noise step
so that the Gaussian mechanism sensitivity is bounded at 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 1. Weighted Average  (default, Algorithm 1)
# ─────────────────────────────────────────────────────────────────────────────

def weighted_average(
    client_features: Dict[int, torch.Tensor],   # {cid: z_c (B, d)}
    dataset_sizes:   Dict[int, int],             # {cid: |D_c|}
) -> torch.Tensor:
    """
    z_agg = Σ_c  (|D_c| / |D|)  ·  z_c

    Weights are proportional to local dataset size so larger hospitals
    contribute more signal to the shared representation.

    Returns
    -------
    z_agg : (B, d)  L2-normalised aggregated feature
    """
    total  = sum(dataset_sizes[cid] for cid in client_features)
    z_agg  = torch.zeros_like(next(iter(client_features.values())))

    for cid, z_c in client_features.items():
        w      = dataset_sizes[cid] / total
        z_agg  = z_agg + w * z_c

    return F.normalize(z_agg, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Attention-based Aggregation
# ─────────────────────────────────────────────────────────────────────────────

class AttentionAggregator(nn.Module):
    """
    Learnable aggregation: weights are cosine-similarity scores between
    each client's z_c and a learnable query vector.

    This allows the server to up-weight clients whose representations are
    most semantically coherent with the task, improving alignment under
    heterogeneous distributions (Section IV.C.e, Section IV.B).

    Input  : {cid: z_c} where z_c ∈ ℝ^{B × d}
    Output : z_agg ∈ ℝ^{B × d}
    """

    def __init__(self, embed_dim: int, temperature: float = 0.1):
        super().__init__()
        self.query       = nn.Parameter(torch.randn(embed_dim))
        self.temperature = temperature

    def forward(
        self,
        client_features: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        ids   = sorted(client_features.keys())
        stack = torch.stack([client_features[cid] for cid in ids], dim=1)
        # (B, C, d)  — C clients

        # Cosine similarity with learnable query
        q_norm = F.normalize(self.query, dim=0)         # (d,)
        z_norm = F.normalize(stack, dim=-1)              # (B, C, d)
        scores = (z_norm * q_norm).sum(-1)               # (B, C)

        weights = torch.softmax(scores / self.temperature, dim=-1)   # (B, C)
        z_agg   = (weights.unsqueeze(-1) * stack).sum(1)             # (B, d)
        return F.normalize(z_agg, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Uniform Average  (ablation baseline)
# ─────────────────────────────────────────────────────────────────────────────

def uniform_average(
    client_features: Dict[int, torch.Tensor],
) -> torch.Tensor:
    """Simple unweighted average — used in ablation studies."""
    z_list = list(client_features.values())
    z_agg  = torch.stack(z_list, dim=0).mean(dim=0)      # (B, d)
    return F.normalize(z_agg, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Similarity-weighted Aggregation  (feature-based, PFL baseline)
# ─────────────────────────────────────────────────────────────────────────────

def similarity_weighted_average(
    client_features: Dict[int, torch.Tensor],
    reference_cid:   int,
) -> torch.Tensor:
    """
    Weight each client's contribution by cosine similarity to a reference
    client's feature. Useful for pFedLVM-style personalised aggregation.

    reference_cid is the client currently being served.
    """
    ref   = client_features[reference_cid]           # (B, d)
    ref_n = F.normalize(ref, dim=-1)

    weighted_sum = torch.zeros_like(ref)
    total_weight = 0.0

    for cid, z_c in client_features.items():
        z_n  = F.normalize(z_c, dim=-1)
        sim  = (ref_n * z_n).sum(-1, keepdim=True).clamp(min=0)  # (B, 1)
        weighted_sum = weighted_sum + sim * z_c
        total_weight = total_weight + sim

    z_agg = weighted_sum / (total_weight + 1e-8)
    return F.normalize(z_agg, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_aggregator(cfg) -> Optional[AttentionAggregator]:
    """
    Return the attention aggregator module if needed, else None.
    Weighted/uniform average need no learnable parameters.
    """
    if cfg.aggregation == "attention":
        return AttentionAggregator(cfg.embedding_dim)
    return None


def aggregate(
    strategy: str,
    client_features: Dict[int, torch.Tensor],
    dataset_sizes:   Dict[int, int],
    attention_module: Optional[AttentionAggregator] = None,
    reference_cid:    Optional[int] = None,
) -> torch.Tensor:
    """
    Unified dispatch to the selected aggregation strategy.

    Parameters
    ----------
    strategy         : "weighted_avg" | "attention" | "uniform" | "similarity"
    client_features  : {client_id: z_c (B, d)}
    dataset_sizes    : {client_id: |D_c|}
    attention_module : AttentionAggregator instance (required if strategy="attention")
    reference_cid    : client id for similarity-weighted strategy

    Returns
    -------
    z_agg : (B, d)  L2-normalised, ready for DP noise
    """
    if strategy == "weighted_avg":
        return weighted_average(client_features, dataset_sizes)
    elif strategy == "attention":
        if attention_module is None:
            raise ValueError("attention_module required for strategy='attention'")
        return attention_module(client_features)
    elif strategy == "uniform":
        return uniform_average(client_features)
    elif strategy == "similarity":
        if reference_cid is None:
            raise ValueError("reference_cid required for strategy='similarity'")
        return similarity_weighted_average(client_features, reference_cid)
    else:
        raise ValueError(f"Unknown aggregation strategy: '{strategy}'")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Communication cost tracker
# ─────────────────────────────────────────────────────────────────────────────

class CommunicationTracker:
    """
    Tracks cumulative bytes uploaded/downloaded across all clients and rounds.
    Reproduces Fig. 2 / Section IV.D analysis.
    """

    def __init__(self, embed_dim: int = 1024):
        self.embed_dim   = embed_dim
        self._upload_B   = 0
        self._download_B = 0
        self._rounds     = 0

    def record_round(
        self,
        num_clients:  int,
        batch_size:   int,
        report_len:   int = 256,
    ):
        """Record one communication round for all clients."""
        # Uplink:   h_c (float32) + report tokens (int32)
        up = num_clients * batch_size * (self.embed_dim * 4 + report_len * 4)
        # Downlink: z_agg (float32) broadcast to all clients
        down = num_clients * batch_size * self.embed_dim * 4

        self._upload_B   += up
        self._download_B += down
        self._rounds     += 1

    def summary(self) -> Dict:
        total_MB  = (self._upload_B + self._download_B) / 1e6
        # FedAvg comparison (ResNet-50 ≈ 98 MB × 2 directions per round)
        fedavg_MB = 196 * self._rounds
        return {
            "rounds":          self._rounds,
            "upload_MB":       self._upload_B   / 1e6,
            "download_MB":     self._download_B / 1e6,
            "total_MB":        total_MB,
            "fedavg_total_MB": fedavg_MB,
            "reduction_factor": fedavg_MB / (total_MB + 1e-8),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (f"CommunicationTracker("
                f"rounds={s['rounds']}, "
                f"total={s['total_MB']:.2f} MB, "
                f"FedAvg equiv={s['fedavg_total_MB']:.0f} MB, "
                f"reduction={s['reduction_factor']:.0f}×)")
