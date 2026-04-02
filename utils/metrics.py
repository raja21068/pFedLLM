"""
utils/metrics.py
─────────────────
All evaluation metrics used in pFedLLM experiments.

Tables:
  IV  — BLEU-4, ROUGE-L  (report generation)
  V   — Accuracy, AUC-ROC, F1  (disease classification)
  VI  — Accuracy  (VQA)
  VII — Dice, IoU  (visual grounding)
  XI  — SSIM, PSNR  (inversion attack resistance)
"""

import math
import torch
import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Report Generation:  BLEU-4  &  ROUGE-L
# ─────────────────────────────────────────────────────────────────────────────

def _ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def bleu_score(
    hypothesis: List[str],
    reference: List[str],
    max_n: int = 4,
    smooth: bool = True,
) -> float:
    """
    Sentence-level BLEU-N (corpus average when called iteratively).
    Uses add-1 smoothing by default (Chen & Cherry, 2014).
    """
    precisions = []
    for n in range(1, max_n + 1):
        hyp_counts = _ngram_counts(hypothesis, n)
        ref_counts = _ngram_counts(reference,  n)
        # Clipped count
        clipped = sum(min(cnt, ref_counts[ng]) for ng, cnt in hyp_counts.items())
        total   = max(len(hypothesis) - n + 1, 0)
        if total == 0:
            precisions.append(0.0)
            continue
        p_n = (clipped + 1) / (total + 1) if smooth else (clipped / total if clipped else 0.0)
        precisions.append(p_n)

    if not precisions or all(p == 0 for p in precisions):
        return 0.0

    # Brevity penalty
    bp = math.exp(min(0.0, 1.0 - len(reference) / max(len(hypothesis), 1)))
    log_avg = sum(math.log(p + 1e-9) for p in precisions) / len(precisions)
    return bp * math.exp(log_avg)


def rouge_l(hypothesis: List[str], reference: List[str]) -> float:
    """ROUGE-L (longest common subsequence) F1."""
    n, m = len(hypothesis), len(reference)
    if n == 0 or m == 0:
        return 0.0
    # LCS via DP
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if hypothesis[i-1] == reference[j-1] \
                       else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[n][m]
    prec  = lcs / n if n > 0 else 0.0
    rec   = lcs / m if m > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def corpus_bleu4(hypotheses: List[List[str]], references: List[List[str]]) -> float:
    """BLEU-4 averaged over a corpus."""
    return np.mean([bleu_score(h, r) for h, r in zip(hypotheses, references)])


def corpus_rouge_l(hypotheses: List[List[str]], references: List[List[str]]) -> float:
    """ROUGE-L averaged over a corpus."""
    return np.mean([rouge_l(h, r) for h, r in zip(hypotheses, references)])


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Disease Classification:  Accuracy, AUC-ROC, F1
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def classification_metrics(
    logits: np.ndarray,   # (N, C) raw logits
    labels: np.ndarray,   # (N, C) binary labels
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Multi-label classification metrics (Table V).
    Macro-averaged AUC-ROC, accuracy (sample-level), and F1.
    """
    probs = _sigmoid(logits)               # (N, C)
    preds = (probs >= threshold).astype(int)

    # Sample-level accuracy (exact match ratio)
    accuracy = (preds == labels).all(axis=1).mean()

    # Macro-averaged F1
    from functools import reduce
    eps = 1e-8
    tp  = (preds * labels).sum(axis=0)    # (C,)
    fp  = (preds * (1 - labels)).sum(axis=0)
    fn  = ((1 - preds) * labels).sum(axis=0)
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    macro_f1  = f1.mean()

    # Macro-averaged AUC-ROC (trapezoidal rule)
    aucs = []
    for c in range(labels.shape[1]):
        if labels[:, c].sum() == 0 or labels[:, c].sum() == len(labels):
            continue   # skip degenerate classes
        aucs.append(_auc_roc(labels[:, c], probs[:, c]))
    macro_auc = float(np.mean(aucs)) if aucs else 0.0

    return {
        "accuracy":  float(accuracy),
        "auc_roc":   macro_auc,
        "f1":        float(macro_f1),
    }


def _auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Binary AUC-ROC via trapezoidal rule."""
    order  = np.argsort(-y_score)
    y_true = y_true[order]
    npos   = y_true.sum()
    nneg   = len(y_true) - npos
    if npos == 0 or nneg == 0:
        return 0.5
    tp_cum  = np.cumsum(y_true)
    fp_cum  = np.cumsum(1 - y_true)
    tpr     = tp_cum / npos
    fpr     = fp_cum / nneg
    return float(np.trapz(tpr, fpr))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  VQA:  Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def vqa_accuracy(
    logits: np.ndarray,   # (N, A) raw logits
    labels: np.ndarray,   # (N,)   ground-truth answer index
) -> float:
    preds = logits.argmax(axis=1)
    return float((preds == labels).mean())


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Visual Grounding:  Dice  &  IoU
# ─────────────────────────────────────────────────────────────────────────────

def box_iou(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """
    Compute IoU between predicted and ground-truth bounding boxes.

    Boxes format: (x, y, w, h) normalised to [0, 1].
    Returns IoU per sample.
    """
    # Convert to (x1, y1, x2, y2)
    pred_x1 = pred_boxes[:, 0]
    pred_y1 = pred_boxes[:, 1]
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2]
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3]

    gt_x1 = gt_boxes[:, 0]
    gt_y1 = gt_boxes[:, 1]
    gt_x2 = gt_boxes[:, 0] + gt_boxes[:, 2]
    gt_y2 = gt_boxes[:, 1] + gt_boxes[:, 3]

    inter_x1 = np.maximum(pred_x1, gt_x1)
    inter_y1 = np.maximum(pred_y1, gt_y1)
    inter_x2 = np.minimum(pred_x2, gt_x2)
    inter_y2 = np.minimum(pred_y2, gt_y2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h

    pred_area = pred_boxes[:, 2] * pred_boxes[:, 3]
    gt_area   = gt_boxes[:, 2]   * gt_boxes[:, 3]
    union     = pred_area + gt_area - inter

    return inter / np.maximum(union, 1e-8)


def dice_score(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> float:
    """Dice coefficient computed from box overlap (approximation)."""
    iou = box_iou(pred_boxes, gt_boxes)
    dice = 2 * iou / (1 + iou + 1e-8)
    return float(dice.mean())


def grounding_metrics(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> Dict[str, float]:
    """Combined grounding metrics (Table VII)."""
    iou  = box_iou(pred_boxes, gt_boxes)
    dice = 2 * iou / (1 + iou + 1e-8)
    return {"dice": float(dice.mean()), "iou": float(iou.mean())}


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Privacy: SSIM & PSNR  (inversion attack resistance, Table XI)
# ─────────────────────────────────────────────────────────────────────────────

def psnr(original: np.ndarray, reconstructed: np.ndarray,
         max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (lower = safer for privacy)."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(max_val) - 10 * math.log10(mse)


def ssim(original: np.ndarray, reconstructed: np.ndarray,
         data_range: float = 1.0) -> float:
    """
    Structural Similarity Index (lower = safer; harder to reconstruct).
    Simplified single-scale version.
    """
    mu_x = original.mean()
    mu_y = reconstructed.mean()
    sigma_x  = original.std()
    sigma_y  = reconstructed.std()
    sigma_xy = np.mean((original - mu_x) * (reconstructed - mu_y))

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    numerator   = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x**2 + sigma_y**2 + C2)
    return float(numerator / denominator)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Aggregated metric tracker
# ─────────────────────────────────────────────────────────────────────────────

class MetricTracker:
    """Running average tracker for all metrics across rounds."""

    def __init__(self):
        self._sums:   Dict[str, float] = {}
        self._counts: Dict[str, int]   = {}

    def update(self, name: str, value: float, count: int = 1):
        self._sums[name]   = self._sums.get(name, 0.0) + value * count
        self._counts[name] = self._counts.get(name, 0)  + count

    def compute(self) -> Dict[str, float]:
        return {k: self._sums[k] / self._counts[k]
                for k in self._sums if self._counts[k] > 0}

    def reset(self):
        self._sums.clear()
        self._counts.clear()
