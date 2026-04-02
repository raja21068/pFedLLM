"""
federated/client.py
────────────────────
Federated client  —  hospital / institution node.

Responsibilities (Section IV.B, Algorithm 1):
  1. Run local feature extractor C_φc on each image I → h_c
  2. Send (h_c, R) to the server  [latent + de-identified report]
  3. Receive z_agg from the server  [shared semantic + DP noise]
  4. Update personalized head H_ψc and compressor C_φc using local loss ℓ_c
  5. Never share raw images, labels, or model parameters

Each client holds:
  • C_φc  — lightweight feature compressor (CNN / tiny ViT)
  • H_ψc  — personalized task-specific head  (MLP / decoder)
  • Local dataset D_c  (stays entirely on the client)
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple

from models.feature_compressor  import FeatureCompressor, build_compressor
from models.personalized_head   import PersonalizedHead,   build_personalized_head
from utils.differential_privacy import DPOptimizer, GaussianMechanism
from utils.metrics               import (classification_metrics, vqa_accuracy,
                                         grounding_metrics, MetricTracker)
from data.synthetic_dataset      import GenerativeAugmentor, RARE_CLASSES, DISEASE_LABELS


class FederatedClient:
    """
    Single federated client representing one hospital site.

    Parameters
    ----------
    client_id   : unique integer identifier
    cfg         : FederatedConfig
    train_loader: DataLoader for this client's private training data
    device      : torch device
    """

    def __init__(
        self,
        client_id: int,
        cfg,
        train_loader: DataLoader,
        device: torch.device,
    ):
        self.client_id    = client_id
        self.cfg          = cfg
        self.loader       = train_loader
        self.device       = device
        self.dataset_size = len(train_loader.dataset)

        # ── Local model components ────────────────────────────────────
        self.compressor = build_compressor(cfg).to(device)
        self.head       = build_personalized_head(cfg).to(device)

        # ── Optimizer for local parameters (ϕ_c, ψ_c) ────────────────
        local_params = (list(self.compressor.parameters()) +
                        list(self.head.parameters()))
        base_opt = torch.optim.Adam(local_params, lr=cfg.learning_rate,
                                    weight_decay=cfg.weight_decay)

        if cfg.use_dp:
            self.optimizer = DPOptimizer(
                base_opt, model=nn.ModuleList([self.compressor, self.head]),
                max_grad_norm=cfg.dp_max_grad_norm,
                noise_multiplier=cfg.dp_noise_multiplier,
                expected_batch_size=cfg.batch_size,
            )
        else:
            self.optimizer = base_opt   # standard Adam

        # ── Generative augmentor (rare classes) ───────────────────────
        self.augmentor = GenerativeAugmentor(embed_dim=cfg.embedding_dim)
        self.rare_ids  = [DISEASE_LABELS.index(c)
                          for c in RARE_CLASSES
                          if c in DISEASE_LABELS]

        # ── Metrics tracker ───────────────────────────────────────────
        self.tracker = MetricTracker()

        # ── Communication statistics ──────────────────────────────────
        self.bytes_uploaded   = 0
        self.bytes_downloaded = 0

    # ─────────────────────────────────────────────────────────────────
    # Training for one communication round
    # ─────────────────────────────────────────────────────────────────

    def local_train(
        self,
        server_fn,           # callable: (h_c_batch, report_ids, attn_mask) → z_agg
        num_epochs: int = 1,
    ) -> Dict:
        """
        Execute E local epochs, communicating with the server each mini-batch.

        Parameters
        ----------
        server_fn : function that takes (h_c, report_ids, attn_mask)
                    and returns z_agg ∈ ℝ^{B × d}  (simulates server call)
        num_epochs: E in the paper (default 1)

        Returns
        -------
        dict with average loss and metrics for this round.
        """
        self.compressor.train()
        self.head.train()
        self.tracker.reset()

        total_loss  = 0.0
        total_steps = 0

        for epoch in range(num_epochs):
            for batch in self.loader:
                loss, metrics = self._train_step(batch, server_fn)
                total_loss  += loss
                total_steps += 1
                for k, v in metrics.items():
                    self.tracker.update(k, v)

        avg_loss = total_loss / max(total_steps, 1)
        avg_metrics = self.tracker.compute()
        avg_metrics["loss"] = avg_loss
        return avg_metrics

    def _train_step(self, batch: Dict, server_fn) -> Tuple[float, Dict]:
        """Single mini-batch forward/backward pass."""
        self.optimizer.zero_grad()

        # Move batch to device
        images    = batch["image"].to(self.device)       # (B, 1, H, W)
        rep_ids   = batch["report_ids"].to(self.device)  # (B, T)
        attn_mask = batch["attn_mask"].to(self.device)   # (B, T)
        labels    = batch["labels"].to(self.device)      # (B, C)

        # ── Step 1: Extract compressed features ───────────────────────
        h_c = self.compressor(images)                    # (B, d)

        # ── Step 2: Send to server, receive shared representation ─────
        # In real deployment: RPC / HTTP call to cloud server.
        # Here: direct function call (simulates communication).
        self._record_upload(h_c, rep_ids)
        with torch.no_grad():
            z_agg = server_fn(h_c.detach(), rep_ids, attn_mask)
        self._record_download(z_agg)

        # ── Generative augmentation for rare classes ──────────────────
        if self.cfg.use_gen_aug:
            h_c_aug, labels_aug = self.augmentor.augment_batch(
                h_c.detach(), labels, self.rare_ids, ratio=self.cfg.aug_ratio
            )
            # Re-expand z_agg to match augmented batch size
            extra = h_c_aug.shape[0] - h_c.shape[0]
            if extra > 0:
                z_agg_aug = torch.cat([z_agg, z_agg[:extra]], dim=0)
            else:
                z_agg_aug = z_agg
            h_c_fwd, labels_fwd, z_fwd = h_c_aug.to(self.device), labels_aug.to(self.device), z_agg_aug.to(self.device)
        else:
            h_c_fwd, labels_fwd, z_fwd = h_c, labels, z_agg

        # ── Step 3: Forward through personalized head ─────────────────
        task = self.cfg.primary_task
        if task == "report_generation":
            tgt = rep_ids[:, :-1]
            logits = self.head(h_c_fwd, z_fwd, tgt_ids=tgt)
            target = rep_ids[:, 1:]
        elif task == "vqa":
            logits = self.head(h_c_fwd, z_fwd)
            target = batch["vqa_answer"].to(self.device)
        elif task == "visual_grounding":
            logits = self.head(h_c_fwd, z_fwd)
            target = batch["box"].to(self.device)
        else:  # disease_classification
            logits = self.head(h_c_fwd, z_fwd)
            target = labels_fwd

        # ── Step 4: Compute loss and backprop ─────────────────────────
        loss = self.head.loss(logits, target)
        loss.backward()

        # Gradient clipping (non-DP path)
        if not self.cfg.use_dp:
            nn.utils.clip_grad_norm_(
                list(self.compressor.parameters()) + list(self.head.parameters()),
                self.cfg.grad_clip,
            )

        self.optimizer.step()

        # ── Step 5: Compute metrics ───────────────────────────────────
        metrics = self._compute_batch_metrics(logits.detach(), labels.detach())
        return loss.item(), metrics

    def _compute_batch_metrics(self, logits: torch.Tensor,
                                labels: torch.Tensor) -> Dict:
        task = self.cfg.primary_task
        if task == "disease_classification":
            return classification_metrics(
                logits.cpu().numpy(), labels.cpu().numpy()
            )
        elif task == "vqa":
            return {"accuracy": vqa_accuracy(
                logits.cpu().numpy(),
                labels.cpu().long().numpy()
            )}
        else:
            return {}

    # ─────────────────────────────────────────────────────────────────
    # Evaluation (test-time inference)
    # ─────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader, server_fn) -> Dict:
        """Run inference on held-out data and return aggregated metrics."""
        self.compressor.eval()
        self.head.eval()
        tracker = MetricTracker()

        for batch in test_loader:
            images    = batch["image"].to(self.device)
            rep_ids   = batch["report_ids"].to(self.device)
            attn_mask = batch["attn_mask"].to(self.device)
            labels    = batch["labels"].to(self.device)

            h_c   = self.compressor(images)
            z_agg = server_fn(h_c, rep_ids, attn_mask)
            logits = self.head(h_c, z_agg)
            metrics = self._compute_batch_metrics(logits, labels)
            for k, v in metrics.items():
                tracker.update(k, v, count=len(images))

        return tracker.compute()

    # ─────────────────────────────────────────────────────────────────
    # Communication accounting
    # ─────────────────────────────────────────────────────────────────

    def _record_upload(self, h_c: torch.Tensor, rep_ids: torch.Tensor):
        """Track bytes uploaded per sample (feature + report tokens)."""
        # h_c: B × d × 4 bytes; rep_ids: B × T × 4 bytes
        B = h_c.shape[0]
        self.bytes_uploaded += (h_c.numel() * 4 + rep_ids.numel() * 4)

    def _record_download(self, z_agg: torch.Tensor):
        """Track bytes downloaded per sample (z_agg)."""
        self.bytes_downloaded += z_agg.numel() * 4

    def communication_summary(self) -> Dict:
        return {
            "bytes_uploaded":   self.bytes_uploaded,
            "bytes_downloaded": self.bytes_downloaded,
            "total_MB":         (self.bytes_uploaded + self.bytes_downloaded) / 1e6,
        }

    # ─────────────────────────────────────────────────────────────────
    # State serialisation (for checkpointing)
    # ─────────────────────────────────────────────────────────────────

    def state_dict(self) -> Dict:
        return {
            "client_id":   self.client_id,
            "compressor":  self.compressor.state_dict(),
            "head":        self.head.state_dict(),
        }

    def load_state_dict(self, state: Dict):
        self.compressor.load_state_dict(state["compressor"])
        self.head.load_state_dict(state["head"])
