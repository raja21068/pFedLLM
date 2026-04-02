"""
data/synthetic_dataset.py
──────────────────────────
Synthetic dataset that mimics MIMIC-CXR structure for quick testing.
Also contains the partitioning logic for IID / Non-IID splits
(Section V.A — Temporal and Clinical partitioning).

Real dataset usage:
    dataset = MIMICCXRDataset(root="/path/to/mimic-cxr-jpg")
    loaders  = partition_dataset(dataset, cfg)

Synthetic usage (no downloads needed):
    loaders = make_synthetic_loaders(cfg)
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# MIMIC-CXR disease labels  (14 classes from CheXpert label set)
# ─────────────────────────────────────────────────────────────────────────────

DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion",
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices",
]

# Classes with few samples → candidates for generative augmentation (Section V.C)
RARE_CLASSES = {"Cardiomegaly", "Lung Lesion", "Consolidation"}


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic sample dataclass
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticCXRSample:
    """Single synthetic (image, report_tokens, labels) tuple."""

    def __init__(self, image_size: int = 224, vocab_size: int = 30522,
                 max_report_len: int = 64, num_classes: int = 14,
                 client_id: int = 0):
        # Simulate domain shift: each client has a slightly different mean
        shift = client_id * 0.05
        self.image = torch.randn(1, image_size, image_size) + shift
        self.report_ids = torch.randint(100, vocab_size - 100,
                                        (max_report_len,))
        self.report_ids[0]  = 101   # [CLS]
        self.report_ids[-1] = 102   # [SEP]
        self.attention_mask = torch.ones(max_report_len, dtype=torch.long)
        # Multi-label with sparse positives (most studies have 1-3 findings)
        self.labels = torch.zeros(num_classes)
        n_pos = random.randint(1, 3)
        pos_idx = random.sample(range(num_classes), n_pos)
        for i in pos_idx:
            self.labels[i] = 1.0
        # VQA
        self.vqa_question = torch.randint(100, vocab_size - 100, (32,))
        self.vqa_answer   = torch.randint(0, 500, ()).long()
        # Grounding box (x, y, w, h) ∈ [0,1]
        x, y = random.random() * 0.6, random.random() * 0.6
        w, h = 0.1 + random.random() * 0.3, 0.1 + random.random() * 0.3
        self.box = torch.tensor([x, y, min(w, 1-x), min(h, 1-y)])


# ─────────────────────────────────────────────────────────────────────────────
# Torch Dataset wrappers
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticMIMICDataset(Dataset):
    """
    Fully synthetic dataset that reproduces the (I, R, y) structure.
    Supports all four tasks; samples are generated at init time.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        image_size: int = 224,
        vocab_size: int = 30522,
        max_report_len: int = 64,
        num_classes: int = 14,
        client_id: int = 0,
        seed: int = 42,
    ):
        rng = random.Random(seed + client_id)
        np_rng = np.random.default_rng(seed + client_id)
        torch.manual_seed(seed + client_id)

        # Build samples
        shift = client_id * 0.05
        self.images   = torch.randn(num_samples, 1, image_size, image_size) + shift
        self.reports  = torch.randint(100, vocab_size - 100,
                                      (num_samples, max_report_len))
        self.reports[:, 0]  = 101
        self.reports[:, -1] = 102
        self.attn_masks = torch.ones(num_samples, max_report_len, dtype=torch.long)

        # Multi-label (sparse) classification labels
        self.labels = torch.zeros(num_samples, num_classes)
        for i in range(num_samples):
            n_pos = rng.randint(1, 3)
            pos   = rng.sample(range(num_classes), n_pos)
            self.labels[i, pos] = 1.0

        # VQA answers (classification over 500 answer classes)
        self.vqa_answers = torch.randint(0, 500, (num_samples,))

        # Grounding boxes (x,y,w,h)
        raw = torch.rand(num_samples, 4)
        raw[:, 2:] = raw[:, 2:] * 0.4 + 0.1   # width & height ∈ [0.1, 0.5]
        self.boxes = raw

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "image":        self.images[idx],
            "report_ids":   self.reports[idx],
            "attn_mask":    self.attn_masks[idx],
            "labels":       self.labels[idx],
            "vqa_answer":   self.vqa_answers[idx],
            "box":          self.boxes[idx],
        }


# ─────────────────────────────────────────────────────────────────────────────
# IID / Non-IID partitioning  (Section V.A)
# ─────────────────────────────────────────────────────────────────────────────

def partition_iid(
    dataset: Dataset,
    num_clients: int,
    seed: int = 42,
) -> List[List[int]]:
    """
    IID partition: randomly assign samples with balanced label distributions.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset)).tolist()
    split_size = len(dataset) // num_clients
    return [indices[c * split_size:(c + 1) * split_size]
            for c in range(num_clients)]


def partition_non_iid_temporal(
    dataset: Dataset,
    num_clients: int,
) -> List[List[int]]:
    """
    Non-IID temporal partition: contiguous blocks by patient order.
    Induces temporal distribution shift across clients (Section V.A).
    """
    N = len(dataset)
    block = N // num_clients
    return [list(range(c * block, (c + 1) * block))
            for c in range(num_clients)]


def partition_non_iid_dirichlet(
    labels: np.ndarray,       # (N,) dominant class per sample
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> List[List[int]]:
    """
    Non-IID clinical partition via Dirichlet allocation.
    Lower α → more heterogeneous distributions (simulates geographic /
    device-type variation, Section V.A non-IID clinical).
    """
    rng = np.random.default_rng(seed)
    num_classes = labels.max() + 1
    class_indices = [np.where(labels == k)[0] for k in range(num_classes)]
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for k_idx in class_indices:
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        proportions = (np.cumsum(proportions) * len(k_idx)).astype(int)[:-1]
        splits = np.split(k_idx, proportions)
        for c, split in enumerate(splits):
            client_indices[c].extend(split.tolist())

    return client_indices


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_loaders(cfg) -> Tuple[List[DataLoader], DataLoader]:
    """
    Create per-client training DataLoaders and a global test DataLoader.

    Returns
    -------
    train_loaders : List[DataLoader]  — one per client
    test_loader   : DataLoader        — shared test set
    """
    samples_per_client = max(200, cfg.batch_size * 10)
    test_samples       = max(500, cfg.batch_size * 20)

    train_loaders = []
    for c in range(cfg.num_clients):
        ds = SyntheticMIMICDataset(
            num_samples=samples_per_client,
            image_size=cfg.image_size,
            vocab_size=cfg.vocab_size,
            max_report_len=cfg.max_report_len,
            num_classes=cfg.num_classes,
            client_id=c,
            seed=cfg.seed,
        )
        loader = DataLoader(ds, batch_size=cfg.batch_size,
                            shuffle=True, drop_last=True,
                            num_workers=0)
        train_loaders.append(loader)

    test_ds = SyntheticMIMICDataset(
        num_samples=test_samples,
        image_size=cfg.image_size,
        vocab_size=cfg.vocab_size,
        max_report_len=cfg.max_report_len,
        num_classes=cfg.num_classes,
        client_id=99,
        seed=cfg.seed + 1,
    )
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=0)

    return train_loaders, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Generative augmentation stub  (Section V.C)
# ─────────────────────────────────────────────────────────────────────────────

class GenerativeAugmentor:
    """
    Simulates the latent-diffusion–based rare-class augmentation.

    In production: a server-pretrained latent diffusion model conditioned on
    the class label generates synthetic compressed features h̃_c and paired
    report embeddings. Synthetic samples are mixed 1:1 with real data.

    Improvement: rare-class F1 +11–17% (Table X, Section V.C).
    """

    def __init__(self, embed_dim: int = 1024, noise_scale: float = 0.1,
                 seed: int = 42):
        self.embed_dim  = embed_dim
        self.noise_scale = noise_scale
        torch.manual_seed(seed)

        # Class-specific prototype embeddings (learnt from server data)
        self.class_prototypes: Dict[int, torch.Tensor] = {}

    def register_prototype(self, class_id: int, embedding: torch.Tensor):
        """Store the mean embedding for a class on the server."""
        self.class_prototypes[class_id] = embedding.detach().clone()

    def generate(
        self,
        class_id: int,
        num_samples: int,
        embed_dim: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate synthetic feature embeddings for a given rare class.

        Returns
        -------
        h_synth : (num_samples, embed_dim)
        """
        d = embed_dim or self.embed_dim
        if class_id in self.class_prototypes:
            proto = self.class_prototypes[class_id]
            noise = torch.randn(num_samples, d) * self.noise_scale
            return proto.unsqueeze(0).expand(num_samples, -1) + noise
        else:
            # No prototype yet: sample from Gaussian
            return torch.randn(num_samples, d) * 0.3

    def augment_batch(
        self,
        h_real: torch.Tensor,      # (B, d) real features
        labels_real: torch.Tensor, # (B, C) multi-label
        rare_class_ids: List[int],
        ratio: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mix real and synthetic features (1:1 ratio by default).
        Only augments samples belonging to rare classes.
        """
        synth_h, synth_y = [], []
        for cls_id in rare_class_ids:
            mask = labels_real[:, cls_id] == 1
            n_rare = mask.sum().item()
            if n_rare == 0:
                continue
            n_aug = max(1, int(n_rare * ratio))
            h_synth = self.generate(cls_id, n_aug, h_real.shape[-1])
            y_synth = torch.zeros(n_aug, labels_real.shape[-1])
            y_synth[:, cls_id] = 1.0
            synth_h.append(h_synth)
            synth_y.append(y_synth)

        if not synth_h:
            return h_real, labels_real

        h_all = torch.cat([h_real] + synth_h, dim=0)
        y_all = torch.cat([labels_real] + synth_y, dim=0)

        # Shuffle combined batch
        perm = torch.randperm(len(h_all))
        return h_all[perm], y_all[perm]
