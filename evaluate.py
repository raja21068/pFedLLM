"""
evaluate.py
────────────
Standalone evaluation and result visualization for pFedLLM.
Produces all figures and tables from Sections V.D–V.K of the paper.

Usage:
    # Quick demo (NumPy, no PyTorch needed):
    python evaluate.py --demo

    # Full evaluation after training:
    python evaluate.py --checkpoint ./outputs/best_model.pt

    # Show all paper tables and figures:
    python evaluate.py --paper_tables
"""

import os, sys, json, math, argparse
import numpy as np

# ── Paper reference values (verbatim from Tables IV–XII) ─────────────────────

TABLE_IV = {   # Radiology Report Generation, MIMIC-CXR
    "method":     ["Local","FedAvg","Per-FedAvg","pFedLVM","AdaptiveDualBranch",
                   "GLM-4.5V","Med-R1","pFedLLM (ours)"],
    "IID BLEU-4": [0.196, 0.225, 0.241, 0.248, 0.262, 0.275, 0.268, 0.298],
    "IID ROUGE-L":[0.255, 0.272, 0.288, 0.297, 0.311, 0.321, 0.315, 0.346],
    "NonIID BLEU-4": [0.181,0.207,0.222,0.229,0.238,0.243,0.236,0.264],
    "NonIID ROUGE-L":[0.239,0.256,0.271,0.279,0.289,0.297,0.290,0.319],
}

TABLE_V = {    # Disease Classification, MIMIC-CXR
    "method":   ["Local","FedAvg","Per-FedAvg","pFedLVM","AdaptiveDualBranch",
                 "GLM-4.5V","Med-R1","pFedLLM (ours)"],
    "IID Acc":  [0.819,0.838,0.846,0.851,0.857,0.859,0.854,0.872],
    "IID AUC":  [0.872,0.891,0.903,0.907,0.912,0.915,0.910,0.928],
    "IID F1":   [0.801,0.823,0.832,0.839,0.844,0.842,0.838,0.859],
    "NonIID Acc":[0.784,0.811,0.823,0.829,0.834,0.840,0.832,0.849],
    "NonIID AUC":[0.845,0.868,0.879,0.882,0.887,0.900,0.890,0.939],
    "NonIID F1": [0.763,0.786,0.797,0.803,0.808,0.820,0.808,0.825],
}

TABLE_VI = {   # VQA Accuracy, Med-MAT
    "method": ["Local","FedAvg","Per-FedAvg","pFedLVM","AdaptiveDualBranch",
               "GLM-4.5V","Med-R1","pFedLLM (ours)"],
    "IID":    [0.620,0.681,0.704,0.712,0.722,0.738,0.735,0.752],
    "NonIID": [0.598,0.654,0.677,0.688,0.698,0.709,0.706,0.727],
}

TABLE_VII = {  # Visual Grounding, Open-I
    "method":    ["Local","FedAvg","Per-FedAvg","pFedLVM","AdaptiveDualBranch","pFedLLM (ours)"],
    "IID Dice":  [0.523,0.562,0.577,0.583,0.591,0.621],
    "IID IoU":   [0.401,0.432,0.447,0.452,0.459,0.489],
    "NonIID Dice":[0.511,0.548,0.563,0.571,0.578,0.607],
    "NonIID IoU": [0.389,0.417,0.433,0.439,0.445,0.476],
}

TABLE_VIII = {  # DP Noise vs Performance
    "sigma":   [0.0,  0.1,  0.3,  0.5,  0.8],
    "BLEU-4":  [0.298,0.291,0.283,0.262,0.219],
    "AUC-ROC": [0.928,0.924,0.919,0.901,0.861],
    "VQA Acc": [0.752,0.748,0.743,0.726,0.684],
}

TABLE_IX = {    # RDP budget
    "sigma":   [0.0,  0.1,  0.3,  0.5,  0.8],
    "epsilon": [None, 4.7,  2.1,  1.5,  1.2],
    "BLEU-4":  [0.298,0.291,0.283,0.262,0.219],
    "AUC-ROC": [0.928,0.924,0.919,0.901,0.861],
    "VQA Acc": [0.752,0.748,0.743,0.726,0.684],
}

TABLE_X = {     # Rare-class augmentation
    "class":   ["Cardiomegaly","Lung Lesion","Consolidation"],
    "w/o aug": [0.54, 0.47, 0.52],
    "w/ aug":  [0.61, 0.55, 0.59],
}

TABLE_XI = {    # Inversion attack resistance
    "method":  ["FedAvg","Per-FedAvg","pFedLLM"],
    "attack":  ["Gradient Inv.","Gradient Inv.","Feature Inv."],
    "SSIM":    [0.81, 0.75, 0.19],
    "PSNR":    [24.7, 22.9, 11.3],
}

TABLE_XII = {   # Backbone convergence
    "backbone":          ["Med-Gemini","GLM-4.5V","Med-R1"],
    "gpu_hours":         [18.2, 12.9, 21.4],
    "rounds_to_converge":[37,   29,   34],
    "comm_per_round_KB": [8.0,  6.9,  8.0],
}

# ─────────────────────────────────────────────────────────────────────────────

def hbar(width=62):
    return "═" * width

def divider(width=62):
    return "─" * width


def print_table_iv():
    print("\n" + hbar())
    print("  TABLE IV — Radiology Report Generation on MIMIC-CXR")
    print(hbar())
    print(f"  {'Method':<26} {'IID BLEU':>9} {'IID ROUGE':>10} {'NonIID BLEU':>12} {'NonIID ROUGE':>13}")
    print("  " + divider(70))
    methods = TABLE_IV["method"]
    for i, m in enumerate(methods):
        mark = " ◄" if "ours" in m else ""
        print(f"  {m+mark:<26} "
              f"{TABLE_IV['IID BLEU-4'][i]:>9.3f} "
              f"{TABLE_IV['IID ROUGE-L'][i]:>10.3f} "
              f"{TABLE_IV['NonIID BLEU-4'][i]:>12.3f} "
              f"{TABLE_IV['NonIID ROUGE-L'][i]:>13.3f}")
    print(hbar())


def print_table_v():
    print("\n" + hbar())
    print("  TABLE V — Disease Classification on MIMIC-CXR")
    print(hbar())
    print(f"  {'Method':<26} {'IID Acc':>8} {'IID AUC':>8} {'IID F1':>7} "
          f"{'NID Acc':>8} {'NID AUC':>8} {'NID F1':>7}")
    print("  " + divider(72))
    for i, m in enumerate(TABLE_V["method"]):
        mark = " ◄" if "ours" in m else ""
        print(f"  {m+mark:<26} "
              f"{TABLE_V['IID Acc'][i]:>8.3f} "
              f"{TABLE_V['IID AUC'][i]:>8.3f} "
              f"{TABLE_V['IID F1'][i]:>7.3f} "
              f"{TABLE_V['NonIID Acc'][i]:>8.3f} "
              f"{TABLE_V['NonIID AUC'][i]:>8.3f} "
              f"{TABLE_V['NonIID F1'][i]:>7.3f}")
    print(hbar())


def print_table_vi():
    print("\n" + hbar())
    print("  TABLE VI — VQA Accuracy on Med-MAT")
    print(hbar())
    print(f"  {'Method':<26} {'IID':>8} {'NonIID':>9}")
    print("  " + divider(45))
    for i, m in enumerate(TABLE_VI["method"]):
        mark = " ◄" if "ours" in m else ""
        print(f"  {m+mark:<26} {TABLE_VI['IID'][i]:>8.3f} {TABLE_VI['NonIID'][i]:>9.3f}")
    print(hbar())


def print_table_vii():
    print("\n" + hbar())
    print("  TABLE VII — Visual Grounding on Open-I")
    print(hbar())
    print(f"  {'Method':<26} {'IID Dice':>9} {'IID IoU':>8} {'NID Dice':>9} {'NID IoU':>8}")
    print("  " + divider(64))
    for i, m in enumerate(TABLE_VII["method"]):
        mark = " ◄" if "ours" in m else ""
        print(f"  {m+mark:<26} "
              f"{TABLE_VII['IID Dice'][i]:>9.3f} "
              f"{TABLE_VII['IID IoU'][i]:>8.3f} "
              f"{TABLE_VII['NonIID Dice'][i]:>9.3f} "
              f"{TABLE_VII['NonIID IoU'][i]:>8.3f}")
    print(hbar())


def print_table_viii_ix():
    print("\n" + hbar())
    print("  TABLES VIII & IX — DP Noise vs Utility / Privacy Budget")
    print(hbar())
    print(f"  {'σ':>6}  {'ε (RDP)':>10}  {'BLEU-4':>8}  {'AUC-ROC':>9}  {'VQA Acc':>9}")
    print("  " + divider(52))
    for i, sigma in enumerate(TABLE_IX["sigma"]):
        eps = TABLE_IX["epsilon"][i]
        es  = "  no DP" if eps is None else f"{eps:>10.1f}"
        print(f"  {sigma:>6.1f}  {es}  "
              f"{TABLE_IX['BLEU-4'][i]:>8.3f}  "
              f"{TABLE_IX['AUC-ROC'][i]:>9.3f}  "
              f"{TABLE_IX['VQA Acc'][i]:>9.3f}")
    print(hbar())


def print_table_x():
    print("\n" + hbar())
    print("  TABLE X — Rare-Class Performance with/without Augmentation (F1)")
    print(hbar())
    print(f"  {'Class':<22} {'w/o Aug':>9} {'w/ Aug':>9} {'Δ F1':>8}")
    print("  " + divider(50))
    for i, cls in enumerate(TABLE_X["class"]):
        wo = TABLE_X["w/o aug"][i]; w = TABLE_X["w/ aug"][i]
        print(f"  {cls:<22} {wo:>9.2f} {w:>9.2f} {w-wo:>+8.2f}")
    print(hbar())
    print("  Generative augmentation improves rare-class F1 by +11–17%")


def print_table_xi():
    print("\n" + hbar())
    print("  TABLE XI — Inversion Attack Reconstruction Metrics")
    print("  (SSIM ↓ and PSNR ↓ = harder to reconstruct = more private)")
    print(hbar())
    print(f"  {'Method':<16} {'Attack Type':<18} {'SSIM↓':>7} {'PSNR↓':>8}")
    print("  " + divider(52))
    for i, m in enumerate(TABLE_XI["method"]):
        mark = " ◄ best" if m == "pFedLLM" else ""
        print(f"  {m:<16} {TABLE_XI['attack'][i]:<18} "
              f"{TABLE_XI['SSIM'][i]:>7.2f} {TABLE_XI['PSNR'][i]:>8.1f}{mark}")
    print(hbar())


def print_table_xii():
    print("\n" + hbar())
    print("  TABLE XII — Backbone Compute & Convergence (server-side)")
    print(hbar())
    print(f"  {'Backbone':<14} {'GPU-hrs':>9} {'→Converge':>11} {'KB/round':>10}")
    print("  " + divider(48))
    for i, b in enumerate(TABLE_XII["backbone"]):
        mark = " ◄ MoE efficient" if b == "GLM-4.5V" else ""
        print(f"  {b:<14} {TABLE_XII['gpu_hours'][i]:>9.1f} "
              f"{TABLE_XII['rounds_to_converge'][i]:>11} "
              f"{TABLE_XII['comm_per_round_KB'][i]:>10.1f}{mark}")
    print(hbar())


def print_ablation_summary():
    print("\n" + hbar())
    print("  SECTION V.I — Ablation Study (reported improvements)")
    print(hbar())
    ablations = [
        ("Vision-only backbone (no LLM text)", "BLEU-4 ↓14%", "text semantics essential"),
        ("Shared head across clients",          "Perf.  ↓9%",  "per-client adaptation needed"),
        ("Feature dim d=256 → d=1024",          "Acc.   +2.4pp","larger d helps until saturation"),
        ("DP removed (σ=0→0.6)",                "AUC: 0.928→0.861","privacy-utility trade-off"),
        ("No generative augmentation",          "Rare-cls F1 ↓11-17%","augmentation critical for rare"),
        ("GLM-4.5V vs Med-Gemini",              "BLEU-4 +2.1% (NonIID)","3D-RoPE aids spatial reasoning"),
        ("MRG-LLM prompt + GLM-4.5V",           "BLEU-4 +4.3%","best personalisation combo"),
    ]
    for component, effect, insight in ablations:
        print(f"  • {component:<40} → {effect}")
        print(f"    Insight: {insight}")
    print(hbar())


def run_demo():
    """Quick demo showing all paper tables without training."""
    print("\n" + hbar(70))
    print("  pFedLLM — Paper Results Summary (All Tables)")
    print("  'Advancing Medical Imaging with LLM-Driven Personalized FL'")
    print(hbar(70))
    print_table_iv()
    print_table_v()
    print_table_vi()
    print_table_vii()
    print_table_viii_ix()
    print_table_x()
    print_table_xi()
    print_table_xii()
    print_ablation_summary()

    print("\n" + hbar())
    print("  KEY FINDINGS")
    print(hbar())
    print("  1. pFedLLM outperforms all baselines on 4 tasks (Tables IV–VII)")
    print(f"     Best: AUC 0.939 vs 0.887 (AdaptiveDualBranchNet) — +5.2pp")
    print(f"     Best: BLEU-4 0.264 vs 0.243 (GLM-4.5V) — +8.6% Non-IID")
    print()
    print("  2. Communication: 9–10 KB/sample vs >100 MB (FedAvg) — >1000× less")
    print()
    print("  3. Privacy: Feature inversion SSIM=0.19 vs gradient SSIM=0.81")
    print()
    print("  4. DP: ε=1.5 at σ=0.5 with <4% AUC degradation")
    print()
    print("  5. GLM-4.5V backbone: fastest convergence (29 rounds vs 37)")
    print(hbar())


def main():
    p = argparse.ArgumentParser(description="pFedLLM Evaluation")
    p.add_argument("--demo",         action="store_true", help="Show all paper tables")
    p.add_argument("--paper_tables", action="store_true", help="Same as --demo")
    p.add_argument("--checkpoint",   type=str, default=None)
    p.add_argument("--table",        type=str, default=None,
                   choices=["iv","v","vi","vii","viii","ix","x","xi","xii","ablation"])
    args = p.parse_args()

    if args.demo or args.paper_tables:
        run_demo(); return

    if args.table:
        dispatch = {
            "iv": print_table_iv,  "v": print_table_v,
            "vi": print_table_vi,  "vii": print_table_vii,
            "viii": print_table_viii_ix, "ix": print_table_viii_ix,
            "x": print_table_x,   "xi": print_table_xi,
            "xii": print_table_xii, "ablation": print_ablation_summary,
        }
        dispatch[args.table](); return

    if args.checkpoint:
        try:
            import torch
            ckpt = torch.load(args.checkpoint, map_location="cpu")
            print(f"\nCheckpoint round: {ckpt.get('round', '?')}")
            print("Test metrics:")
            for k, v in ckpt.get("metrics", {}).items():
                print(f"  {k}: {v:.4f}")
        except ImportError:
            print("[Error] PyTorch not installed. Use --demo for paper tables.")
        return

    # Default: show all tables
    run_demo()


if __name__ == "__main__":
    main()
