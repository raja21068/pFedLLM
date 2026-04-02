"""
federated/server.py
────────────────────
Federated server  —  central cloud node.

Responsibilities (Section IV.B, Algorithm 1, lines 11–13):
  1. Receive compressed features h_c and de-identified reports R from each client.
  2. Run the fixed multimodal LLM  F_θ(h_c, R) → z_c  for each client.
  3. Aggregate client representations:
         z_agg = Σ_c (|D_c| / |D|) · z_c  +  ε_DP
  4. Broadcast z_agg to all clients (same representation for all clients).
  5. Optionally support attention-based aggregation weighting.

The server NEVER:
  • sends F_θ parameters to clients (>10 GB)
  • stores raw images or PHI
  • updates F_θ during federated rounds (fixed / pre-trained)
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from models.server_llm           import ServerLLM, build_server_llm
from utils.differential_privacy  import GaussianMechanism, compute_epsilon


class FederatedServer:
    """
    Central server that hosts the multimodal LLM and performs aggregation.

    Parameters
    ----------
    cfg    : FederatedConfig
    device : torch.device (server-side GPU/CPU)
    """

    def __init__(self, cfg, device: torch.device):
        self.cfg    = cfg
        self.device = device

        # ── Server-side LLM (fixed; never distributed) ────────────────
        self.llm = build_server_llm(cfg).to(device)
        self.llm.eval()
        for p in self.llm.parameters():
            p.requires_grad_(False)

        # ── Differential privacy: Gaussian noise on z_agg ─────────────
        self.dp_mechanism = GaussianMechanism(
            sigma=cfg.dp_noise_multiplier if cfg.use_dp else 0.0,
            sensitivity=1.0,
        )

        # ── Aggregation strategy ──────────────────────────────────────
        self.aggregation = cfg.aggregation   # "weighted_avg" | "attention"
        if self.aggregation == "attention":
            # Learnable attention weights for semantic similarity-based agg.
            self.agg_attn = nn.Linear(cfg.embedding_dim, 1, bias=False).to(device)

        # ── Running statistics ────────────────────────────────────────
        self.round_count = 0
        self.total_clients_served = 0

    # ─────────────────────────────────────────────────────────────────
    # Single-client LLM call  (used inside the training loop)
    # ─────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def process_client_features(
        self,
        h_c: torch.Tensor,              # (B, d) client image feature
        report_ids: torch.Tensor,        # (B, T) tokenised report
        attn_mask: Optional[torch.Tensor] = None,  # (B, T)
    ) -> torch.Tensor:                   # (B, d) z_c (no DP yet)
        """
        F_θ(h_c, R) → z_c

        Called once per mini-batch per client.
        """
        h_c       = h_c.to(self.device)
        report_ids = report_ids.to(self.device)
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)
        return self.llm(h_c, report_ids, attn_mask)

    # ─────────────────────────────────────────────────────────────────
    # Aggregation  (line 12, Algorithm 1)
    # ─────────────────────────────────────────────────────────────────

    def aggregate(
        self,
        client_features: Dict[int, torch.Tensor],   # {client_id: z_c}
        dataset_sizes:   Dict[int, int],             # {client_id: |D_c|}
    ) -> torch.Tensor:
        """
        Weighted average aggregation with optional DP noise.

        z_agg = Σ_c (|D_c| / |D|) · z_c  +  ε_DP

        Parameters
        ----------
        client_features : z_c per client for this mini-batch
        dataset_sizes   : number of training samples per client

        Returns
        -------
        z_agg : (B, d) — same shared representation broadcast to all clients
        """
        if self.aggregation == "attention":
            return self._attention_aggregate(client_features, dataset_sizes)

        # ── Weighted average (default) ────────────────────────────────
        total = sum(dataset_sizes[cid] for cid in client_features)
        z_agg = torch.zeros_like(next(iter(client_features.values())))

        for cid, z_c in client_features.items():
            weight = dataset_sizes[cid] / total
            z_agg  = z_agg + weight * z_c

        # L2-normalise before adding DP noise (sensitivity = 1)
        z_agg = F.normalize(z_agg, dim=-1)

        # Add ε_DP Gaussian noise
        z_agg = self.dp_mechanism(z_agg)

        return z_agg

    def _attention_aggregate(
        self,
        client_features: Dict[int, torch.Tensor],
        dataset_sizes:   Dict[int, int],
    ) -> torch.Tensor:
        """
        Attention-based aggregation: weights proportional to semantic
        similarity scores (alternative to plain weighted average).
        """
        ids  = list(client_features.keys())
        stack = torch.stack([client_features[cid] for cid in ids], dim=1)  # (B, C, d)
        # Compute attention score for each client
        scores = self.agg_attn(stack).squeeze(-1)  # (B, C)
        weights = F.softmax(scores, dim=-1)         # (B, C)
        z_agg   = (weights.unsqueeze(-1) * stack).sum(dim=1)  # (B, d)
        z_agg   = F.normalize(z_agg, dim=-1)
        return self.dp_mechanism(z_agg)

    # ─────────────────────────────────────────────────────────────────
    # Full-round aggregation (one call per communication round)
    # Processes ALL clients; used when training is co-located
    # ─────────────────────────────────────────────────────────────────

    def build_server_fn(self, dataset_sizes: Dict[int, int]):
        """
        Returns a callable  server_fn(h_c, report_ids, attn_mask) → z_agg
        that is passed to each client's local_train() call.

        In this single-machine simulation the server_fn encapsulates:
          1. LLM encoding of the single client's mini-batch
          2. Broadcast of the (same batch) z_agg with DP noise

        NOTE: In a real distributed system, the server would:
          - Buffer all clients' h_c tensors
          - Run LLM on all simultaneously
          - Aggregate once per round
          - Broadcast z_agg to all clients

        Here we approximate by running per-client and adding DP per call.
        """
        def server_fn(
            h_c: torch.Tensor,
            report_ids: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            z_c = self.process_client_features(h_c, report_ids, attn_mask)
            z_c = F.normalize(z_c, dim=-1)
            # In single-client call: DP applied immediately to z_c
            return self.dp_mechanism(z_c).to(h_c.device)

        return server_fn

    # ─────────────────────────────────────────────────────────────────
    # Multi-stage fine-tuning (Section IV.C, pre-training stage)
    # ─────────────────────────────────────────────────────────────────

    def pretrain(
        self,
        train_loader,
        num_epochs: int = 2,
        lr: float = 1e-4,
    ):
        """
        Stage 1 — fine-tune F_θ on the available server data (e.g. MIMIC-CXR)
        before federated rounds begin. After this, F_θ is frozen.
        """
        print("[Server] Starting multi-stage LLM pre-training...")
        self.llm.unfreeze_parameters()
        optimizer = torch.optim.Adam(self.llm.parameters(), lr=lr)

        self.llm.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            steps = 0
            for batch in train_loader:
                images    = batch["image"].to(self.device)
                rep_ids   = batch["report_ids"].to(self.device)
                attn_mask = batch["attn_mask"].to(self.device)

                # Proxy self-supervised objective: predict [CLS] representation
                # In production: masked language modelling + contrastive loss
                with torch.enable_grad():
                    # Dummy loss for the proxy encoder
                    z = self.llm.text_encoder(rep_ids, attn_mask)
                    loss = -z[:, 0].mean()   # placeholder
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.llm.parameters(), 1.0)
                    optimizer.step()
                total_loss += loss.item()
                steps += 1

            print(f"  [Pretrain Epoch {epoch+1}/{num_epochs}] "
                  f"loss = {total_loss/max(steps,1):.4f}")

        # Freeze F_θ permanently
        self.llm.freeze_parameters()
        self.llm.eval()
        print("[Server] LLM frozen. Federated rounds beginning.\n")

    # ─────────────────────────────────────────────────────────────────
    # Privacy budget reporting
    # ─────────────────────────────────────────────────────────────────

    def privacy_budget(self, num_samples: int) -> Tuple[float, float]:
        """
        Compute (ε, δ) for the current run configuration.
        Based on Table IX / Section V.H.
        """
        if not self.cfg.use_dp:
            return float("inf"), 0.0
        q = self.cfg.batch_size / max(num_samples, 1)
        eps, _ = compute_epsilon(
            sigma=self.cfg.dp_noise_multiplier,
            sampling_rate=q,
            num_rounds=self.cfg.rounds,
            delta=self.cfg.dp_delta,
        )
        return eps, self.cfg.dp_delta

    # ─────────────────────────────────────────────────────────────────
    # Communication cost estimation
    # ─────────────────────────────────────────────────────────────────

    def communication_cost_per_round(self, batch_size: int = 32) -> Dict:
        """
        Estimate per-round communication cost (Section IV.D).

        pFedLLM:   ~5-6 KB uplink  +  4 KB downlink  =  9-10 KB per sample
        FedAvg:    model parameters, typically > 100 MB
        """
        d = self.cfg.embedding_dim
        T = self.cfg.max_report_len

        h_c_bytes = batch_size * d * 4        # uplink: feature vector (float32)
        rep_bytes = batch_size * T * 4        # uplink: tokenised report
        z_agg_bytes = batch_size * d * 4      # downlink: shared representation

        uplink   = h_c_bytes + rep_bytes
        downlink = z_agg_bytes

        return {
            "uplink_KB":       uplink / 1024,
            "downlink_KB":     downlink / 1024,
            "total_KB":        (uplink + downlink) / 1024,
            "vs_fedavg_ratio": f">{100_000 / ((uplink + downlink) / 1024):.0f}x less",
        }
