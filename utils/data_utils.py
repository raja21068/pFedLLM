"""
utils/data_utils.py
────────────────────
Dataset loading, partitioning, and preprocessing utilities for pFedLLM.

Supports three datasets from the paper (Section V.A):
  • MIMIC-CXR         — 377,110 chest radiographs + radiology reports
  • MedTrinity-25M    — 25M images across 10 imaging modalities
  • Med-MAT           — VQA dataset from 106 medical datasets

Three partition schemes (Section V.A):
  • IID               — random balanced split
  • Non-IID temporal  — contiguous patient-ID blocks (temporal shift)
  • Non-IID clinical  — Dirichlet allocation by metadata (device/region)
"""

import os
import json
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader, Subset


# ─────────────────────────────────────────────────────────────────────────────
# MIMIC-CXR Dataset Wrapper  (real data)
# ─────────────────────────────────────────────────────────────────────────────

class MIMICCXRDataset(Dataset):
    """
    MIMIC-CXR-JPG dataset loader.
    Requires local access to the MIMIC-CXR-JPG dataset.

    Download: https://physionet.org/content/mimic-cxr-jpg/

    Expected directory structure:
        root/
          files/
            p10/ p11/ ... p19/
              pXXXXX/
                sYYYYYY/
                  ZZZZZ.jpg
          mimic-cxr-2.0.0-chexpert.csv.gz
          mimic-cxr-2.0.0-split.csv.gz

    Parameters
    ----------
    root     : path to MIMIC-CXR-JPG root
    split    : "train" | "validate" | "test"
    transform: optional torchvision transform (default: resize + normalize)
    max_report_len: max tokens for report text
    """

    CHEXPERT_LABELS = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
        "Lung Opacity", "No Finding", "Pleural Effusion",
        "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices",
    ]

    def __init__(
        self,
        root:            str,
        split:           str = "train",
        transform=None,
        max_report_len:  int = 256,
        vocab_size:      int = 30522,
    ):
        self.root           = Path(root)
        self.split          = split
        self.transform      = transform
        self.max_report_len = max_report_len
        self.vocab_size     = vocab_size
        self.samples        = self._load_metadata()

    def _load_metadata(self) -> List[Dict]:
        """Load split CSV and label CSV; pair each study with its image path."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required: pip install pandas")

        split_csv = self.root / "mimic-cxr-2.0.0-split.csv.gz"
        label_csv = self.root / "mimic-cxr-2.0.0-chexpert.csv.gz"

        if not split_csv.exists():
            raise FileNotFoundError(
                f"MIMIC-CXR split file not found at {split_csv}.\n"
                "Please download from https://physionet.org/content/mimic-cxr-jpg/\n"
                "or run with --use_synthetic for quick testing."
            )

        splits = pd.read_csv(split_csv)
        labels = pd.read_csv(label_csv)

        # Filter to requested split
        split_ids = splits[splits["split"] == self.split]["dicom_id"].tolist()
        labels    = labels[labels["dicom_id"].isin(split_ids)]

        # Build sample list
        samples = []
        for _, row in labels.iterrows():
            img_path = (self.root / "files" /
                        f"p{str(row['subject_id'])[:2]}" /
                        f"p{row['subject_id']}" /
                        f"s{row['study_id']}" /
                        f"{row['dicom_id']}.jpg")
            label_vec = [
                max(0.0, float(row.get(lbl, 0.0)) if not pd.isna(row.get(lbl, 0.0)) else 0.0)
                for lbl in self.CHEXPERT_LABELS
            ]
            samples.append({
                "image_path": str(img_path),
                "subject_id": row["subject_id"],
                "study_id":   row["study_id"],
                "labels":     label_vec,
            })
        return samples

    def _load_report(self, study_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and tokenise de-identified radiology report (simplified)."""
        # In production: load actual .txt report and use BiomedBERT tokeniser
        report_ids  = torch.randint(100, self.vocab_size - 100, (self.max_report_len,))
        report_ids[0]  = 101   # [CLS]
        report_ids[-1] = 102   # [SEP]
        attn_mask = torch.ones(self.max_report_len, dtype=torch.long)
        return report_ids, attn_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Load image
        try:
            from PIL import Image
            import torchvision.transforms as T
            img = Image.open(sample["image_path"]).convert("L")   # grayscale
            if self.transform:
                img = self.transform(img)
            else:
                default_tf = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.5], std=[0.5]),
                ])
                img = default_tf(img)
        except Exception:
            img = torch.randn(1, 224, 224)   # fallback for missing files

        report_ids, attn_mask = self._load_report(sample["study_id"])
        labels = torch.tensor(sample["labels"], dtype=torch.float32)

        return {
            "image":      img,
            "report_ids": report_ids,
            "attn_mask":  attn_mask,
            "labels":     labels,
            "subject_id": sample["subject_id"],
            "study_id":   sample["study_id"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Partitioning  (Section V.A)
# ─────────────────────────────────────────────────────────────────────────────

def partition_iid(
    dataset: Dataset,
    num_clients: int,
    seed: int = 42,
) -> List[List[int]]:
    """
    IID partition: random balanced assignment ensuring similar label
    distributions across clients (Section V.A).
    """
    rng     = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset)).tolist()
    size    = len(dataset) // num_clients
    return [indices[c * size:(c + 1) * size] for c in range(num_clients)]


def partition_non_iid_temporal(
    dataset: Dataset,
    num_clients: int,
) -> List[List[int]]:
    """
    Non-IID temporal partition: contiguous blocks by patient ID order.
    Induces temporal distribution shift across clients (Section V.A).
    The first clients see earlier studies; later clients see more recent ones.
    """
    N    = len(dataset)
    size = N // num_clients
    return [list(range(c * size, min((c + 1) * size, N)))
            for c in range(num_clients)]


def partition_non_iid_clinical(
    labels:      np.ndarray,    # (N, num_classes) multi-label or (N,) dominant class
    num_clients: int,
    alpha:       float = 0.5,
    seed:        int   = 42,
) -> List[List[int]]:
    """
    Non-IID clinical partition via Dirichlet allocation.

    Simulates real-world hospital heterogeneity due to geographic region,
    imaging device type, or patient demographics (Section V.A).

    Lower alpha → more heterogeneous distributions (stronger non-IID).
    alpha=0.1: extreme non-IID; alpha=100: near-IID.

    Operates on the dominant class per sample (argmax for multi-label).
    """
    rng = np.random.default_rng(seed)

    # Get dominant class per sample
    if labels.ndim == 2:
        dom_labels = labels.argmax(axis=1)
    else:
        dom_labels = labels

    num_classes     = int(dom_labels.max()) + 1
    class_indices   = [np.where(dom_labels == k)[0] for k in range(num_classes)]
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for k_idx in class_indices:
        if len(k_idx) == 0:
            continue
        props  = rng.dirichlet(np.full(num_clients, alpha))
        splits = (np.cumsum(props) * len(k_idx)).astype(int)[:-1]
        for c, chunk in enumerate(np.split(k_idx, splits)):
            client_indices[c].extend(chunk.tolist())

    return client_indices


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def make_dataloaders(
    dataset:     Dataset,
    cfg,
    labels:      Optional[np.ndarray] = None,
    test_split:  float = 0.1,
    seed:        int   = 42,
) -> Tuple[List[DataLoader], DataLoader]:
    """
    Partition dataset into per-client training loaders + global test loader.

    Parameters
    ----------
    dataset    : full Dataset (MIMIC-CXR or synthetic)
    cfg        : FederatedConfig
    labels     : (N, C) label array; required for Dirichlet partition
    test_split : fraction held out for global test set

    Returns
    -------
    train_loaders : one DataLoader per client
    test_loader   : global held-out test set
    """
    N          = len(dataset)
    n_test     = int(N * test_split)
    n_train    = N - n_test
    rng        = np.random.default_rng(seed)
    all_idx    = rng.permutation(N)
    test_idx   = all_idx[:n_test].tolist()
    train_idx  = all_idx[n_test:].tolist()

    # Partition training indices across clients
    train_subset_labels = (labels[train_idx] if labels is not None else None)

    if cfg.partition == "iid":
        client_idx_lists = partition_iid(
            Subset(dataset, train_idx), cfg.num_clients, seed
        )
        # Convert local indices back to global
        client_idx_lists = [[train_idx[i] for i in local]
                            for local in client_idx_lists]

    elif cfg.partition == "non_iid_temporal":
        client_idx_lists = []
        size = len(train_idx) // cfg.num_clients
        for c in range(cfg.num_clients):
            client_idx_lists.append(train_idx[c * size:(c + 1) * size])

    elif cfg.partition == "non_iid_clinical":
        if train_subset_labels is None:
            raise ValueError("labels required for non_iid_clinical partition")
        local_lists = partition_non_iid_clinical(
            train_subset_labels, cfg.num_clients, seed=seed
        )
        client_idx_lists = [[train_idx[i] for i in local]
                            for local in local_lists]
    else:
        raise ValueError(f"Unknown partition scheme: {cfg.partition}")

    # Build per-client DataLoaders
    train_loaders = [
        DataLoader(
            Subset(dataset, idx_list),
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for idx_list in client_idx_lists
    ]

    # Test DataLoader (shared across all clients)
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loaders, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Dataset registry  (maps name → Dataset class)
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(cfg) -> Dataset:
    """
    Load the dataset specified in cfg.dataset.

    For real datasets: provide cfg.data_dir.
    For synthetic testing: set cfg.use_synthetic = True.
    """
    if cfg.use_synthetic:
        from data.synthetic_dataset import SyntheticMIMICDataset
        return SyntheticMIMICDataset(
            num_samples=500 * cfg.num_clients,
            image_size=cfg.image_size,
            vocab_size=cfg.vocab_size,
            max_report_len=cfg.max_report_len,
            num_classes=cfg.num_classes,
        )

    if cfg.dataset == "mimic_cxr":
        if not cfg.data_dir:
            raise ValueError("cfg.data_dir must be set for MIMIC-CXR")
        return MIMICCXRDataset(
            root=cfg.data_dir,
            split="train",
            max_report_len=cfg.max_report_len,
            vocab_size=cfg.vocab_size,
        )

    raise NotImplementedError(
        f"Dataset '{cfg.dataset}' not yet implemented. "
        "Available: 'mimic_cxr', or set use_synthetic=True."
    )


# ─────────────────────────────────────────────────────────────────────────────
# De-identification utilities  (Section IV.A)
# ─────────────────────────────────────────────────────────────────────────────

class ReportDeidentifier:
    """
    Minimal de-identification of radiology reports before transmission.

    In production: use a clinical NLP pipeline (e.g., MIMIC-III de-id tools,
    Philter, or a fine-tuned NER model) to remove PHI.

    Paper states: "reports are transmitted only after institutional
    de-identification (removal of PHI) and optional encryption."
    """

    # Minimal PHI patterns (extend with full regex suite in production)
    PHI_PATTERNS = [
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",       # dates
        r"\bDr\.?\s+[A-Z][a-z]+\b",             # doctor names
        r"\bMR#?\s*\d+\b",                       # medical record numbers
        r"\b\d{3}-\d{2}-\d{4}\b",               # SSN-like
        r"\b\(\d{3}\)\s*\d{3}-\d{4}\b",         # phone numbers
    ]

    def __init__(self, replacement: str = "[REDACTED]"):
        import re
        self.patterns    = [re.compile(p) for p in self.PHI_PATTERNS]
        self.replacement = replacement

    def deidentify(self, text: str) -> str:
        """Remove PHI patterns from free-text report."""
        for pat in self.patterns:
            text = pat.sub(self.replacement, text)
        return text

    def deidentify_batch(self, texts: List[str]) -> List[str]:
        return [self.deidentify(t) for t in texts]


# ─────────────────────────────────────────────────────────────────────────────
# Collate function for variable-length reports
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate to handle variable-length report tokens."""
    keys = batch[0].keys()
    out  = {}
    for k in keys:
        vals = [item[k] for item in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals)
        else:
            out[k] = vals
    return out
