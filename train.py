"""
train.py
─────────
Main entry point for pFedLLM federated training.

Implements Algorithm 1 from the paper:
────────────────────────────────────────────────────────────────────────
    Require: R rounds, E local epochs, client set {1,…,|C|}
    Initialise server LLM F_θ (pre-trained)
    for each client c: initialise C_φc and H_ψc
    for round r = 1 to R:
        for each client c in parallel:
            for epoch e = 1 to E:
                for each mini-batch (I, R, y) ~ D_c:
                    h_c ← C_φc(I)
                    Send (h_c, R) to server
                    Server: z_c ← F_θ(h_c, R)
                    z_agg ← Σ(|D_c|/|D|) · z_c + ε_DP
                    Send z_agg to each client
                    ŷ ← H_ψc([h_c, z_agg])
                    Update (φ_c, ψ_c) by minimising ℓ_c(ŷ, y)
────────────────────────────────────────────────────────────────────────

Usage:
    python train.py                          # synthetic data, default config
    python train.py --use_synthetic --rounds 5 --num_clients 3  # fast test
    python train.py --config non_iid --rounds 50 --num_clients 10
"""

import os
import sys
import json
import time
import argparse
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List

# ── Project imports ────────────────────────────────────────────────────────
from configs.default     import FederatedConfig, get_config
from federated.server    import FederatedServer
from federated.client    import FederatedClient
from data.synthetic_dataset import make_synthetic_loaders
from utils.differential_privacy import print_privacy_report


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Federated Training Loop
# ─────────────────────────────────────────────────────────────────────────────

class pFedLLMTrainer:
    """Orchestrates the full pFedLLM federated training procedure."""

    def __init__(self, cfg: FederatedConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n{'='*60}")
        print(f"  pFedLLM — Personalized Federated Learning")
        print(f"{'='*60}")
        print(f"  Device       : {self.device}")
        print(f"  Clients      : {cfg.num_clients}")
        print(f"  Rounds       : {cfg.rounds}")
        print(f"  Local epochs : {cfg.local_epochs}")
        print(f"  Task         : {cfg.primary_task}")
        print(f"  Partition    : {cfg.partition}")
        print(f"  DP (σ)       : {cfg.dp_noise_multiplier if cfg.use_dp else 'disabled'}")
        print(f"  Embedding dim: {cfg.embedding_dim}")
        print(f"{'='*60}\n")

        set_seed(cfg.seed)
        os.makedirs(cfg.output_dir, exist_ok=True)

    def setup(self):
        """Initialise server, clients, and data loaders."""
        print("[Setup] Building data loaders...")
        if self.cfg.use_synthetic:
            self.train_loaders, self.test_loader = make_synthetic_loaders(self.cfg)
        else:
            raise NotImplementedError(
                "Real dataset loading not included. "
                "Run with --use_synthetic, or implement "
                "data/mimic_cxr_dataset.py for your local MIMIC-CXR path."
            )

        print("[Setup] Initialising server LLM...")
        self.server = FederatedServer(self.cfg, self.device)

        print("[Setup] Initialising federated clients...")
        self.clients: List[FederatedClient] = []
        dataset_sizes = {}
        for c in range(self.cfg.num_clients):
            client = FederatedClient(
                client_id=c,
                cfg=self.cfg,
                train_loader=self.train_loaders[c],
                device=self.device,
            )
            self.clients.append(client)
            dataset_sizes[c] = client.dataset_size

        self.dataset_sizes = dataset_sizes

        # Model parameter count
        total_client_params = (
            self.clients[0].compressor.num_parameters +
            self.clients[0].head.num_parameters
        )
        server_params = self.server.llm.num_parameters
        print(f"\n  Client params (C_φc + H_ψc): {total_client_params:,}")
        print(f"  Server LLM params (F_θ)    : {server_params:,}")

        # Communication cost
        cost = self.server.communication_cost_per_round(self.cfg.batch_size)
        print(f"\n  Communication per round:")
        print(f"    Uplink  : {cost['uplink_KB']:.1f} KB")
        print(f"    Downlink: {cost['downlink_KB']:.1f} KB")
        print(f"    Total   : {cost['total_KB']:.1f} KB  ({cost['vs_fedavg_ratio']})")

        # Privacy budget
        if self.cfg.use_dp:
            avg_ds = sum(dataset_sizes.values()) // max(len(dataset_sizes), 1)
            eps, delta = self.server.privacy_budget(avg_ds)
            print(f"\n  Privacy budget: ε = {eps:.2f}, δ = {delta}")

        print()

    def pretrain(self):
        """Multi-stage LLM pre-training (stage 1, Section IV.C)."""
        if self.cfg.pretrain_rounds > 0:
            # Use client 0's loader as proxy server data
            self.server.pretrain(
                self.train_loaders[0],
                num_epochs=min(2, self.cfg.pretrain_rounds),
            )

    def train(self) -> Dict:
        """
        Main federated training loop (Algorithm 1).

        Returns
        -------
        history dict with per-round metrics for all clients.
        """
        history = {"rounds": [], "train_metrics": [], "test_metrics": []}
        best_metric = 0.0

        print(f"[Train] Starting {self.cfg.rounds} federated rounds...\n")
        t0 = time.time()

        for rnd in range(1, self.cfg.rounds + 1):
            round_start = time.time()

            # Build server function (simulates server being live)
            server_fn = self.server.build_server_fn(self.dataset_sizes)

            # ── Train all clients in parallel (simulated sequentially) ──
            round_metrics_all = []
            for client in self.clients:
                metrics = client.local_train(server_fn, self.cfg.local_epochs)
                round_metrics_all.append(metrics)

            # ── Aggregate round metrics ───────────────────────────────────
            avg_metrics = self._average_metrics(round_metrics_all)
            elapsed = time.time() - round_start

            history["rounds"].append(rnd)
            history["train_metrics"].append(avg_metrics)

            # ── Logging ───────────────────────────────────────────────────
            if rnd % self.cfg.log_every == 0 or rnd == 1:
                metric_str = "  ".join(
                    f"{k}: {v:.4f}" for k, v in avg_metrics.items()
                )
                print(f"  [Round {rnd:>3}/{self.cfg.rounds}] {metric_str}  "
                      f"({elapsed:.1f}s)")

            # ── Evaluation ────────────────────────────────────────────────
            if rnd % self.cfg.eval_every == 0 or rnd == self.cfg.rounds:
                test_metrics = self._evaluate(server_fn)
                history["test_metrics"].append(
                    {"round": rnd, **test_metrics}
                )
                auc = test_metrics.get("auc_roc", test_metrics.get("accuracy", 0.0))
                print(f"\n  [Eval  Round {rnd:>3}] " +
                      "  ".join(f"{k}: {v:.4f}" for k, v in test_metrics.items()))

                # Save best model
                if auc > best_metric:
                    best_metric = auc
                    self._save_checkpoint(rnd, test_metrics)
                    print(f"  ↑ New best checkpoint saved (metric={auc:.4f})\n")

        total_time = time.time() - t0
        print(f"\n[Train] Done. Total time: {total_time:.1f}s")
        print(f"[Train] Best test metric: {best_metric:.4f}")
        self._save_history(history)
        return history

    def _evaluate(self, server_fn) -> Dict:
        """Evaluate all clients on the shared test set and average."""
        all_metrics = []
        for client in self.clients:
            m = client.evaluate(self.test_loader, server_fn)
            all_metrics.append(m)
        return self._average_metrics(all_metrics)

    @staticmethod
    def _average_metrics(metrics_list: List[Dict]) -> Dict:
        """Average a list of metric dicts across clients."""
        if not metrics_list:
            return {}
        keys = metrics_list[0].keys()
        return {k: float(np.mean([m[k] for m in metrics_list if k in m]))
                for k in keys}

    def _save_checkpoint(self, rnd: int, metrics: Dict):
        path = Path(self.cfg.output_dir) / "best_model.pt"
        torch.save({
            "round":    rnd,
            "metrics":  metrics,
            "server":   self.server.llm.state_dict(),
            "clients": [c.state_dict() for c in self.clients],
            "cfg":      self.cfg.__dict__,
        }, path)

    def _save_history(self, history: Dict):
        path = Path(self.cfg.output_dir) / "training_history.json"
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"[Train] History saved to {path}")

    def print_final_summary(self, history: Dict):
        """Print a summary matching paper tables."""
        print("\n" + "=" * 60)
        print("  pFedLLM — Final Results Summary")
        print("=" * 60)
        if history["test_metrics"]:
            last = history["test_metrics"][-1]
            for k, v in last.items():
                if k != "round":
                    print(f"  {k:20s}: {v:.4f}")
        print("=" * 60)

        # Communication summary
        total_up   = sum(c.bytes_uploaded   for c in self.clients)
        total_down = sum(c.bytes_downloaded for c in self.clients)
        print(f"\n  Total uplink  : {total_up/1e6:.2f} MB")
        print(f"  Total downlink: {total_down/1e6:.2f} MB")
        print(f"  (FedAvg equivalent: >1 GB per round × {self.cfg.rounds} rounds)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="pFedLLM Training")

    # Config shortcut
    p.add_argument("--config", type=str, default="fast",
                   choices=["default", "fast", "iid", "non_iid", "no_dp", "glm45v"],
                   help="Named config preset (default=fast for quick testing)")

    # Override individual options
    p.add_argument("--num_clients", type=int,   default=None)
    p.add_argument("--rounds",      type=int,   default=None)
    p.add_argument("--local_epochs",type=int,   default=None)
    p.add_argument("--batch_size",  type=int,   default=None)
    p.add_argument("--embed_dim",   type=int,   default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--dp_sigma",    type=float, default=None)
    p.add_argument("--no_dp",       action="store_true")
    p.add_argument("--no_aug",      action="store_true")
    p.add_argument("--use_synthetic", action="store_true", default=True)
    p.add_argument("--partition",   type=str,   default=None,
                   choices=["iid", "non_iid_temporal", "non_iid_clinical"])
    p.add_argument("--task",        type=str,   default=None,
                   choices=["disease_classification", "report_generation",
                            "vqa", "visual_grounding"])
    p.add_argument("--output_dir",  type=str,   default="./outputs")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--privacy_report", action="store_true",
                   help="Print DP budget table and exit")

    return p.parse_args()


def main():
    args = parse_args()

    # Load base config
    cfg = get_config(args.config)

    # Apply CLI overrides
    if args.num_clients is not None: cfg.num_clients = args.num_clients
    if args.rounds      is not None: cfg.rounds      = args.rounds
    if args.local_epochs is not None: cfg.local_epochs = args.local_epochs
    if args.batch_size  is not None: cfg.batch_size  = args.batch_size
    if args.embed_dim   is not None: cfg.embedding_dim = args.embed_dim
    if args.lr          is not None: cfg.learning_rate = args.lr
    if args.dp_sigma    is not None: cfg.dp_noise_multiplier = args.dp_sigma
    if args.no_dp:                   cfg.use_dp = False
    if args.no_aug:                  cfg.use_gen_aug = False
    if args.use_synthetic:           cfg.use_synthetic = True
    if args.partition   is not None: cfg.partition = args.partition
    if args.task        is not None: cfg.primary_task = args.task
    cfg.output_dir = args.output_dir
    cfg.seed       = args.seed

    # Just print privacy report if requested
    if args.privacy_report:
        print_privacy_report(cfg)
        return

    # Run training
    trainer = pFedLLMTrainer(cfg)
    trainer.setup()
    trainer.pretrain()
    history = trainer.train()
    trainer.print_final_summary(history)


if __name__ == "__main__":
    main()
