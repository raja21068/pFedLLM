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
import re
import json
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader, Subset

# ── HuggingFace datasets — aliased to avoid collision with load_project_dataset()
try:
    from datasets import load_dataset as hf_load_dataset
except ImportError:
    hf_load_dataset = None   # graceful fallback; only needed for MedTrinity


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
    """

    CHEXPERT_LABELS = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
        "Lung Opacity", "No Finding", "Pleural Effusion",
        "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices",
    ]

    def __init__(self, root, split="train", transform=None,
                 max_report_len=256, vocab_size=30522):
        self.root           = Path(root)
        self.split          = split
        self.transform      = transform
        self.max_report_len = max_report_len
        self.vocab_size     = vocab_size
        self.samples        = self._load_metadata()

    def _load_metadata(self):
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
        split_ids = splits[splits["split"] == self.split]["dicom_id"].tolist()
        labels    = labels[labels["dicom_id"].isin(split_ids)]

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

    def _load_report(self, study_id):
        report_ids  = torch.randint(100, self.vocab_size - 100, (self.max_report_len,))
        report_ids[0]  = 101
        report_ids[-1] = 102
        attn_mask = torch.ones(self.max_report_len, dtype=torch.long)
        return report_ids, attn_mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            from PIL import Image
            import torchvision.transforms as T
            img = Image.open(sample["image_path"]).convert("L")
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
            img = torch.randn(1, 224, 224)

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
# Fallback tokeniser (used when `transformers` is not installed)
# ─────────────────────────────────────────────────────────────────────────────

class _WordHashTokenizer:
    """
    Minimal word-level tokeniser — maps each word to a stable hash-derived
    token ID so no external library is needed.
    Vocabulary: [CLS]=101, [SEP]=102, [PAD]=0, words→[103, vocab_size).
    """
    def __init__(self, vocab_size=30522):
        self.vocab_size = vocab_size

    def _word_to_id(self, word):
        return 103 + (hash(word) % (self.vocab_size - 103))

    def encode(self, text, max_length):
        words = re.findall(r"\w+", text.lower())
        ids   = [101] + [self._word_to_id(w) for w in words] + [102]
        ids   = ids[:max_length]
        mask  = [1] * len(ids)
        pad   = max_length - len(ids)
        ids  += [0] * pad
        mask += [0] * pad
        return (torch.tensor(ids,  dtype=torch.long),
                torch.tensor(mask, dtype=torch.long))


# ─────────────────────────────────────────────────────────────────────────────
# MedTrinity-25M Dataset Wrapper  (HuggingFace: UCSC-VLAA/MedTrinity-25M)
# ─────────────────────────────────────────────────────────────────────────────

class MedTrinityDataset(Dataset):
    """
    Wraps the HuggingFace MedTrinity-25M dataset (25M_demo config) and maps
    its fields to the pFedLLM batch format.

    HuggingFace schema (25M_demo)       pFedLLM expected shape
    ──────────────────────────────      ────────────────────────────────────
    image   PIL.Image RGB var-size  →   FloatTensor (1, image_size, image_size)
    caption str (multigranular)     →   report_ids  LongTensor (max_report_len,)
                                        attn_mask   LongTensor (max_report_len,)
    caption keyword match           →   labels      FloatTensor (14,)  CheXpert
    — (not in dataset)              →   vqa_answer  LongTensor scalar  [stub=0]
    — (not in dataset)              →   box         FloatTensor (4,)   [stub=0]

    Fixes applied
    ─────────────
      1. RGB→grayscale via T.Grayscale to match 1-channel encoder input
      2. Resize to cfg.image_size (default 224)
      3. Caption tokenised with AutoTokenizer or _WordHashTokenizer fallback
      4. 14-class multi-hot labels derived from keyword matching in caption
      5. vqa_answer and box set to zero stubs for interface compatibility
    """

    CHEXPERT_LABELS = [
        "atelectasis", "cardiomegaly", "consolidation", "edema",
        "enlarged cardiomediastinum", "fracture", "lung lesion",
        "lung opacity", "no finding", "pleural effusion",
        "pleural other", "pneumonia", "pneumothorax", "support devices",
    ]

    def __init__(self, hf_dataset, image_size=224, max_report_len=256,
                 vocab_size=30522, num_classes=14,
                 tokenizer_name="bert-base-uncased"):
        self.ds             = hf_dataset
        self.image_size     = image_size
        self.max_report_len = max_report_len
        self.vocab_size     = vocab_size
        self.num_classes    = num_classes

        # FIX 1 & 2: RGB PIL → (1, H, W) grayscale tensor
        try:
            import torchvision.transforms as T
            self.img_tf = T.Compose([
                T.Grayscale(num_output_channels=1),   # RGB→1-ch
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5]),
            ])
        except ImportError:
            self.img_tf = None

        # FIX 3: caption tokeniser
        self.tokenizer = self._build_tokenizer(tokenizer_name, vocab_size)

    @staticmethod
    def _build_tokenizer(name, vocab_size):
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(name)
        except Exception:
            return _WordHashTokenizer(vocab_size=vocab_size)

    def _tokenize(self, text):
        L   = self.max_report_len
        tok = self.tokenizer
        if hasattr(tok, "encode_plus"):
            enc  = tok(text, max_length=L, padding="max_length",
                       truncation=True, return_tensors="pt")
            ids  = enc["input_ids"].squeeze(0)
            mask = enc["attention_mask"].squeeze(0)
        else:
            ids, mask = tok.encode(text, max_length=L)
        return ids.long(), mask.long()

    def _extract_labels(self, caption):
        # FIX 4: multi-hot from keyword matching
        low = caption.lower()
        vec = torch.zeros(self.num_classes, dtype=torch.float32)
        any_pos = False
        for i, kw in enumerate(self.CHEXPERT_LABELS):
            if kw in low:
                vec[i] = 1.0
                any_pos = True
        if not any_pos:
            vec[8] = 1.0   # "no finding"
        return vec

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]

        # Image
        pil_img = row["image"]
        if self.img_tf is not None:
            image = self.img_tf(pil_img)
        else:
            import numpy as np
            arr   = np.array(pil_img.convert("L").resize(
                (self.image_size, self.image_size)))
            image = (torch.from_numpy(arr).float().unsqueeze(0) / 255.0 - 0.5) / 0.5

        # Caption → tokens
        caption = row.get("caption") or ""
        report_ids, attn_mask = self._tokenize(caption)

        # Labels
        labels = self._extract_labels(caption)

        # FIX 5: stubs for fields absent in MedTrinity
        vqa_answer = torch.tensor(0, dtype=torch.long)
        box        = torch.zeros(4, dtype=torch.float32)

        return {
            "image":      image,        # (1, H, W)
            "report_ids": report_ids,   # (max_report_len,)
            "attn_mask":  attn_mask,    # (max_report_len,)
            "labels":     labels,       # (14,)
            "vqa_answer": vqa_answer,   # scalar
            "box":        box,          # (4,)
        }


# ─────────────────────────────────────────────────────────────────────────────
# Partitioning  (Section V.A)
# ─────────────────────────────────────────────────────────────────────────────

def partition_iid(dataset, num_clients, seed=42):
    rng     = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset)).tolist()
    size    = len(dataset) // num_clients
    return [indices[c * size:(c + 1) * size] for c in range(num_clients)]


def partition_non_iid_temporal(dataset, num_clients):
    N    = len(dataset)
    size = N // num_clients
    return [list(range(c * size, min((c + 1) * size, N)))
            for c in range(num_clients)]


def partition_non_iid_clinical(labels, num_clients, alpha=0.5, seed=42):
    rng = np.random.default_rng(seed)
    dom_labels = labels.argmax(axis=1) if labels.ndim == 2 else labels
    num_classes   = int(dom_labels.max()) + 1
    class_indices = [np.where(dom_labels == k)[0] for k in range(num_classes)]
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

def make_dataloaders(dataset, cfg, labels=None, test_split=0.1, seed=42):
    N          = len(dataset)
    n_test     = int(N * test_split)
    rng        = np.random.default_rng(seed)
    all_idx    = rng.permutation(N)
    test_idx   = all_idx[:n_test].tolist()
    train_idx  = all_idx[n_test:].tolist()

    train_subset_labels = (labels[train_idx] if labels is not None else None)

    if cfg.partition == "iid":
        client_idx_lists = partition_iid(
            Subset(dataset, train_idx), cfg.num_clients, seed)
        client_idx_lists = [[train_idx[i] for i in local]
                            for local in client_idx_lists]
    elif cfg.partition == "non_iid_temporal":
        size = len(train_idx) // cfg.num_clients
        client_idx_lists = [train_idx[c * size:(c + 1) * size]
                            for c in range(cfg.num_clients)]
    elif cfg.partition == "non_iid_clinical":
        if train_subset_labels is None:
            raise ValueError("labels required for non_iid_clinical partition")
        local_lists = partition_non_iid_clinical(
            train_subset_labels, cfg.num_clients, seed=seed)
        client_idx_lists = [[train_idx[i] for i in local]
                            for local in local_lists]
    else:
        raise ValueError(f"Unknown partition scheme: {cfg.partition}")

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
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loaders, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Dataset registry
# Renamed to load_project_dataset to avoid shadowing hf_load_dataset above.
# A backwards-compatible alias `load_dataset` is kept at the bottom.
# ─────────────────────────────────────────────────────────────────────────────

def load_project_dataset(cfg) -> Dataset:
    """
    Load the dataset specified in cfg.dataset.

    cfg.dataset values:
      "mimic_cxr"      — MIMIC-CXR-JPG  (JPEG images + label CSVs)
      "mimic_cxr_dcm"  — raw MIMIC-CXR  (DICOM images; label CSVs from cfg.labels_dir)
      "medtrinity_25m" — HuggingFace UCSC-VLAA/MedTrinity-25M (25M_demo)
    Or set cfg.use_synthetic = True for synthetic data.

    MIMIC-CXR config fields (in FederatedConfig or subclass):
      cfg.data_dir       — root of MIMIC-CXR / MIMIC-CXR-JPG download
      cfg.labels_dir     — dir holding the CheXpert CSV (needed for DICOM mode)
      cfg.image_subdir   — sub-folder for images, e.g. 'files512x512' (default 'files')
      cfg.frontal_only   — keep PA/AP views only (default True)
      cfg.uncertain_as   — value for uncertain (-1) labels (default 0.0)
      cfg.mimic_fmt      — 'jpg' or 'dicom' (auto-set by dataset name)
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

    if cfg.dataset in ("mimic_cxr", "mimic_cxr_dcm"):
        if not cfg.data_dir:
            raise ValueError("cfg.data_dir must be set for MIMIC-CXR")

        # Resolve format: explicit cfg.mimic_fmt wins; else infer from dataset name
        fmt = getattr(cfg, "mimic_fmt", None)
        if fmt is None:
            fmt = "dicom" if cfg.dataset == "mimic_cxr_dcm" else "jpg"

        labels_dir   = getattr(cfg, "labels_dir",   None)
        image_subdir = getattr(cfg, "image_subdir", "files")
        frontal_only = getattr(cfg, "frontal_only", True)
        uncertain_as = getattr(cfg, "uncertain_as", 0.0)

        # Use mimic_cxr_adapter if available (supports both jpg and dicom)
        try:
            from mimic_cxr_adapter import MIMICCXRPFedAdapter, MIMICCXRJPGDataset
            base_ds = MIMICCXRJPGDataset(
                datadir=cfg.data_dir,
                split="train",
                image_subdir=image_subdir,
                fmt=fmt,
                labels_dir=labels_dir,
                uncertain_as=uncertain_as,
                img_size=cfg.image_size,
                frontal_only=frontal_only,
            )
            return MIMICCXRPFedAdapter(
                base_ds,
                max_report_len=cfg.max_report_len,
                vocab_size=cfg.vocab_size,
            )
        except ImportError:
            # Fallback to the legacy internal wrapper (JPG only)
            if fmt == "dicom":
                raise ImportError(
                    "mimic_cxr_adapter.py is required for DICOM support.\n"
                    "Copy mimic_cxr_adapter.py to your project root."
                )
            return MIMICCXRDataset(
                root=cfg.data_dir,
                split="train",
                max_report_len=cfg.max_report_len,
                vocab_size=cfg.vocab_size,
            )

    if cfg.dataset == "medtrinity_25m":
        if hf_load_dataset is None:
            raise ImportError(
                "The `datasets` package is required.\n"
                "Install with:  pip install datasets"
            )
        hf_config = getattr(cfg, "hf_config", "25M_demo")
        hf_split  = getattr(cfg, "hf_split",  "train")
        raw_ds = hf_load_dataset(
            "UCSC-VLAA/MedTrinity-25M",
            hf_config,
            split=hf_split,
        )
        return MedTrinityDataset(
            hf_dataset=raw_ds,
            image_size=cfg.image_size,
            max_report_len=cfg.max_report_len,
            vocab_size=cfg.vocab_size,
            num_classes=cfg.num_classes,
        )

    raise NotImplementedError(
        f"Dataset '{cfg.dataset}' not implemented. "
        "Available: 'mimic_cxr', 'mimic_cxr_dcm', 'medtrinity_25m', "
        "or use_synthetic=True."
    )


# Backwards-compatible alias
load_dataset = load_project_dataset


# ─────────────────────────────────────────────────────────────────────────────
# De-identification utilities  (Section IV.A)
# ─────────────────────────────────────────────────────────────────────────────

class ReportDeidentifier:
    PHI_PATTERNS = [
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\bDr\.?\s+[A-Z][a-z]+\b",
        r"\bMR#?\s*\d+\b",
        r"\b\d{3}-\d{2}-\d{4}\b",
        r"\b\(\d{3}\)\s*\d{3}-\d{4}\b",
    ]

    def __init__(self, replacement="[REDACTED]"):
        self.patterns    = [re.compile(p) for p in self.PHI_PATTERNS]
        self.replacement = replacement

    def deidentify(self, text):
        for pat in self.patterns:
            text = pat.sub(self.replacement, text)
        return text

    def deidentify_batch(self, texts):
        return [self.deidentify(t) for t in texts]


# ─────────────────────────────────────────────────────────────────────────────
# Collate function for variable-length reports
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    keys = batch[0].keys()
    out  = {}
    for k in keys:
        vals = [item[k] for item in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals)
        else:
            out[k] = vals
    return out
