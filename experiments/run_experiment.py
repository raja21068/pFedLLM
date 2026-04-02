"""
experiments/run_experiment.py
──────────────────────────────
Reproduces ablation studies and comparison experiments from Section V.I.

Ablations implemented:
  • full             — complete pFedLLM (baseline for ablations)
  • no_dp            — remove differential privacy
  • no_aug           — remove generative augmentation
  • shared_head      — share H_ψ across all clients (no personalisation)
  • vision_only      — replace LLM with vision-only encoder (no text)
  • dim_256/512/2048 — feature dimensionality sensitivity (Fig. 6)

Usage:
    python experiments/run_experiment.py --ablation full
    python experiments/run_experiment.py --ablation no_dp
    python experiments/run_experiment.py --ablation all    # run all ablations
    python experiments/run_experiment.py --compare_baselines
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from configs.default          import FederatedConfig, get_config
from train                    import pFedLLMTrainer


# ─────────────────────────────────────────────────────────────────────────────
# Ablation configurations  (Section V.I)
# ─────────────────────────────────────────────────────────────────────────────

def get_ablation_config(ablation: str) -> FederatedConfig:
    base = get_config("fast")   # fast = synthetic data, 5 rounds, 3 clients

    if ablation == "full":
        return base

    elif ablation == "no_dp":
        # "Removing DP slightly increases metrics but weakens privacy"
        base.use_dp = False
        base.dp_noise_multiplier = 0.0

    elif ablation == "no_aug":
        # "Without generative augmentation, rare-class metrics drop by 11%"
        base.use_gen_aug = False

    elif ablation == "shared_head":
        # "Sharing the personalized head across clients decreases performance by 9%"
        # Simulated by using the same initialisation (no per-client adaptation)
        base._shared_head = True   # flag checked in modified trainer

    elif ablation == "vision_only":
        # "Replacing the multimodal LLM with a vision-only backbone reduces BLEU-4 by 14%"
        base._vision_only = True   # server LLM ignores text

    elif ablation == "dim_256":
        base.embedding_dim = 256

    elif ablation == "dim_512":
        base.embedding_dim = 512

    elif ablation == "dim_1024":
        base.embedding_dim = 1024   # default

    elif ablation == "dim_2048":
        base.embedding_dim = 2048

    elif ablation == "dp_sigma_0":
        base.dp_noise_multiplier = 0.0

    elif ablation == "dp_sigma_01":
        base.dp_noise_multiplier = 0.1

    elif ablation == "dp_sigma_03":
        base.dp_noise_multiplier = 0.3

    elif ablation == "dp_sigma_05":
        base.dp_noise_multiplier = 0.5

    elif ablation == "dp_sigma_08":
        base.dp_noise_multiplier = 0.8

    else:
        raise ValueError(f"Unknown ablation: '{ablation}'")

    return base


# ─────────────────────────────────────────────────────────────────────────────
# Baseline comparison configs
# ─────────────────────────────────────────────────────────────────────────────

BASELINE_NAMES = [
    "Local (no FL)",
    "FedAvg",
    "Per-FedAvg",
    "pFedLVM",
    "AdaptiveDualBranchNet",
    "pFedLLM (ours)",
]

# Reference values from Table V (Non-IID partition, disease classification)
PAPER_RESULTS = {
    "Local (no FL)":          {"accuracy": 0.784, "auc_roc": 0.845, "f1": 0.763},
    "FedAvg":                 {"accuracy": 0.811, "auc_roc": 0.868, "f1": 0.786},
    "Per-FedAvg":             {"accuracy": 0.823, "auc_roc": 0.879, "f1": 0.797},
    "pFedLVM":                {"accuracy": 0.829, "auc_roc": 0.882, "f1": 0.803},
    "AdaptiveDualBranchNet":  {"accuracy": 0.834, "auc_roc": 0.887, "f1": 0.808},
    "GLM-4.5V":               {"accuracy": 0.840, "auc_roc": 0.900, "f1": 0.820},
    "Med-R1":                 {"accuracy": 0.832, "auc_roc": 0.890, "f1": 0.808},
    "pFedLLM (ours)":         {"accuracy": 0.849, "auc_roc": 0.939, "f1": 0.825},
}


# ─────────────────────────────────────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_single_ablation(ablation: str, output_dir: str = "./outputs/ablations") -> Dict:
    """Run one ablation and return the final test metrics."""
    print(f"\n{'─'*50}")
    print(f"  ABLATION: {ablation}")
    print(f"{'─'*50}")

    cfg = get_ablation_config(ablation)
    cfg.output_dir = os.path.join(output_dir, ablation)

    trainer = pFedLLMTrainer(cfg)
    trainer.setup()
    history = trainer.train()

    final_metrics = {}
    if history["test_metrics"]:
        final_metrics = {k: v for k, v in history["test_metrics"][-1].items()
                         if k != "round"}

    return {"ablation": ablation, **final_metrics}


def run_all_ablations(output_dir: str = "./outputs/ablations") -> List[Dict]:
    """Run all ablations from Section V.I."""
    ablations = [
        "full", "no_dp", "no_aug", "dim_256", "dim_512", "dim_2048",
        "dp_sigma_0", "dp_sigma_01", "dp_sigma_03", "dp_sigma_05", "dp_sigma_08",
    ]
    results = []
    for abl in ablations:
        try:
            r = run_single_ablation(abl, output_dir)
            results.append(r)
        except Exception as e:
            print(f"  [WARN] Ablation '{abl}' failed: {e}")
            results.append({"ablation": abl, "error": str(e)})

    return results


def print_ablation_table(results: List[Dict]):
    """Pretty-print ablation results."""
    print("\n" + "=" * 70)
    print(f"  {'Ablation':<25} {'Accuracy':>10} {'AUC-ROC':>10} {'F1':>10}")
    print("=" * 70)
    for r in results:
        name = r.get("ablation", "?")
        acc  = r.get("accuracy", float("nan"))
        auc  = r.get("auc_roc",  float("nan"))
        f1   = r.get("f1",       float("nan"))
        print(f"  {name:<25} {acc:>10.4f} {auc:>10.4f} {f1:>10.4f}")
    print("=" * 70)


def print_paper_comparison_table():
    """Print Table V from the paper for reference."""
    print("\n" + "=" * 65)
    print("  TABLE V (Paper) — Disease Classification, Non-IID")
    print("=" * 65)
    print(f"  {'Method':<25} {'Accuracy':>10} {'AUC-ROC':>10} {'F1':>8}")
    print("-" * 65)
    for name, vals in PAPER_RESULTS.items():
        marker = " ◄" if "ours" in name else ""
        print(f"  {name:<25} {vals['accuracy']:>10.3f} "
              f"{vals['auc_roc']:>10.3f} {vals['f1']:>8.3f}{marker}")
    print("=" * 65)


def run_dp_noise_sweep(output_dir: str = "./outputs/dp_sweep") -> List[Dict]:
    """
    Reproduce Table VIII / Table IX — DP noise vs utility.
    Sweeps σ ∈ {0.0, 0.1, 0.3, 0.5, 0.8}.
    """
    sigmas = [0.0, 0.1, 0.3, 0.5, 0.8]
    results = []
    for sigma in sigmas:
        cfg = get_config("fast")
        cfg.dp_noise_multiplier = sigma
        cfg.use_dp = (sigma > 0.0)
        cfg.output_dir = os.path.join(output_dir, f"sigma_{sigma}")

        t = pFedLLMTrainer(cfg)
        t.setup()
        hist = t.train()

        final = {}
        if hist["test_metrics"]:
            final = {k: v for k, v in hist["test_metrics"][-1].items()
                     if k != "round"}
        results.append({"sigma": sigma, **final})

    print("\n" + "=" * 60)
    print("  TABLE VIII — DP Noise vs Performance (MIMIC-CXR)")
    print("=" * 60)
    print(f"  {'σ':>6}  {'AUC-ROC':>10}  {'Accuracy':>10}")
    print("-" * 60)
    for r in results:
        print(f"  {r['sigma']:>6.1f}  {r.get('auc_roc',0):>10.4f}  {r.get('accuracy',0):>10.4f}")
    print("=" * 60)

    return results


def run_dimensionality_sweep(output_dir: str = "./outputs/dim_sweep") -> List[Dict]:
    """
    Reproduce Fig. 6 — feature dimensionality vs utility.
    Sweeps d ∈ {256, 512, 1024, 2048}.
    """
    dims = [256, 512, 1024, 2048]
    results = []
    for d in dims:
        cfg = get_config("fast")
        cfg.embedding_dim = d
        cfg.output_dir = os.path.join(output_dir, f"dim_{d}")

        t = pFedLLMTrainer(cfg)
        t.setup()
        hist = t.train()

        final = {}
        if hist["test_metrics"]:
            final = {k: v for k, v in hist["test_metrics"][-1].items()
                     if k != "round"}
        results.append({"dim": d, **final})

    print("\n" + "=" * 50)
    print("  FIG. 6 — Dimensionality vs Utility")
    print("=" * 50)
    for r in results:
        print(f"  d={r['dim']:>5}  acc={r.get('accuracy',0):.4f}  "
              f"auc={r.get('auc_roc',0):.4f}")
    print("=" * 50)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="pFedLLM Experiment Runner")
    p.add_argument("--ablation", type=str, default="full",
                   help="Ablation name or 'all'")
    p.add_argument("--compare_baselines", action="store_true",
                   help="Print Table V (paper reference values)")
    p.add_argument("--dp_sweep",  action="store_true",
                   help="Run DP noise sweep (Table VIII)")
    p.add_argument("--dim_sweep", action="store_true",
                   help="Run dimensionality sweep (Fig. 6)")
    p.add_argument("--output_dir", type=str, default="./outputs/ablations")
    args = p.parse_args()

    if args.compare_baselines:
        print_paper_comparison_table()
        return

    if args.dp_sweep:
        run_dp_noise_sweep(args.output_dir)
        return

    if args.dim_sweep:
        run_dimensionality_sweep(args.output_dir)
        return

    if args.ablation == "all":
        results = run_all_ablations(args.output_dir)
        print_ablation_table(results)
        # Save results
        out = Path(args.output_dir) / "ablation_results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out}")
    else:
        r = run_single_ablation(args.ablation, args.output_dir)
        print(f"\nResult: {r}")


if __name__ == "__main__":
    main()
