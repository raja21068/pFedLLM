"""
mimic_cxr_adapter.py
─────────────────────
Drop-in adapter that lets the `cxr_classification` training scripts
(train_densenet.py, train_attention_sl.py, etc.) run on either:

  A) MIMIC-CXR-JPG   — official JPEG release + pre-computed labels
     physionet.org/content/mimic-cxr-jpg/

  B) MIMIC-CXR       — raw DICOM release (no labels CSV bundled)
     physionet.org/content/mimic-cxr/

Usage (drop-in replacement for mimic_cxr_jpg.py):
──────────────────────────────────────────────────
  # MIMIC-CXR-JPG (unchanged interface)
  from mimic_cxr_adapter import MIMICCXRJPGDataset, get_mimic_splits
  ds_train, ds_val, ds_test = get_mimic_splits('/data/mimic-cxr-jpg')

  # Raw MIMIC-CXR (DICOM) — same interface, different datadir + fmt flag
  from mimic_cxr_adapter import MIMICCXRJPGDataset, get_mimic_splits
  ds_train, ds_val, ds_test = get_mimic_splits(
      '/data/mimic-cxr',
      fmt='dicom',
      labels_dir='/data/mimic-cxr-jpg',   # CheXpert CSVs live here
  )

Integration with pFedLLM
────────────────────────
  from mimic_cxr_adapter import MIMICCXRJPGDataset
  from utils.data_utils import make_dataloaders
  full_ds = MIMICCXRJPGDataset('/data/mimic-cxr-jpg', split='train')
  train_loaders, test_loader = make_dataloaders(full_ds, cfg)
"""

from __future__ import annotations

import os
import re
import warnings
import functools
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# ── Optional imports (guarded) ────────────────────────────────────────────────
try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

try:
    from PIL import Image
    _PILLOW = True
except ImportError:
    _PILLOW = False

try:
    import pydicom
    _PYDICOM = True
except ImportError:
    _PYDICOM = False

try:
    import torchvision.transforms as T
    _TVision = True
except ImportError:
    _TVision = False


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CHEXPERT_LABELS: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]
NUM_LABELS = len(CHEXPERT_LABELS)   # 14

# Files that ship with MIMIC-CXR-JPG (also downloadable separately)
_SPLIT_CSV  = "mimic-cxr-2.0.0-split.csv.gz"
_LABEL_CSV  = "mimic-cxr-2.0.0-chexpert.csv.gz"
_NEGBIO_CSV = "mimic-cxr-2.0.0-negbio.csv.gz"   # alternative if chexpert absent


# ─────────────────────────────────────────────────────────────────────────────
# DICOM utilities
# ─────────────────────────────────────────────────────────────────────────────

def _dicom_to_pil(dcm_path: str) -> "Image.Image":
    """
    Load a DICOM file and return a uint8 PIL.Image (mode='L').

    Applies:
      1. Rescale slope / intercept  (Modality LUT)
      2. Window / level clipping    (VOI LUT; falls back to data range)
      3. Photometric inversion if MONOCHROME1
      4. Linear stretch to [0, 255]
    """
    if not _PYDICOM:
        raise ImportError(
            "pydicom is required for DICOM support.\n"
            "Install with:  pip install pydicom"
        )
    if not _PILLOW:
        raise ImportError("Pillow is required:  pip install Pillow")

    dcm = pydicom.dcmread(dcm_path)
    arr = dcm.pixel_array.astype(np.float32)

    # ── Modality LUT ─────────────────────────────────────────────────────────
    slope     = float(getattr(dcm, "RescaleSlope",     1.0))
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    # ── VOI LUT (window / level) ──────────────────────────────────────────────
    wc = getattr(dcm, "WindowCenter", None)
    ww = getattr(dcm, "WindowWidth",  None)
    if wc is not None and ww is not None:
        wc = float(wc[0]) if hasattr(wc, "__len__") else float(wc)
        ww = float(ww[0]) if hasattr(ww, "__len__") else float(ww)
        lo = wc - ww / 2.0
        hi = wc + ww / 2.0
    else:
        lo, hi = arr.min(), arr.max()

    arr = np.clip(arr, lo, hi)

    # ── Photometric interpretation ────────────────────────────────────────────
    photo = getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")
    if photo == "MONOCHROME1":
        arr = hi - arr + lo   # invert

    # ── Stretch to uint8 ─────────────────────────────────────────────────────
    rng = hi - lo
    if rng < 1e-6:
        arr[:] = 0.0
    else:
        arr = (arr - lo) / rng * 255.0
    arr = arr.astype(np.uint8)

    return Image.fromarray(arr, mode="L")


def _jpeg_to_pil(jpg_path: str) -> "Image.Image":
    """Load a JPEG chest X-ray as grayscale PIL image."""
    if not _PILLOW:
        raise ImportError("Pillow is required:  pip install Pillow")
    return Image.open(jpg_path).convert("L")


# ─────────────────────────────────────────────────────────────────────────────
# Label / split CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_labels_df(labels_dir: str) -> "pd.DataFrame":
    """
    Load CheXpert label CSV.  Falls back to NegBio labels if the
    CheXpert file is missing (e.g. older MIMIC-CXR release).
    """
    if not _PANDAS:
        raise ImportError("pandas is required:  pip install pandas")

    labels_dir = Path(labels_dir)
    chexpert = labels_dir / _LABEL_CSV
    negbio   = labels_dir / _NEGBIO_CSV

    if chexpert.exists():
        df = pd.read_csv(chexpert)
    elif negbio.exists():
        warnings.warn(
            f"{_LABEL_CSV} not found in {labels_dir}; "
            f"falling back to {_NEGBIO_CSV}.  "
            "Download the CheXpert file from MIMIC-CXR-JPG for best results.",
            UserWarning,
        )
        df = pd.read_csv(negbio)
    else:
        raise FileNotFoundError(
            f"No label file found in {labels_dir}.\n"
            f"Expected one of: {_LABEL_CSV}, {_NEGBIO_CSV}\n"
            "You can download mimic-cxr-2.0.0-chexpert.csv.gz from:\n"
            "  https://physionet.org/content/mimic-cxr-jpg/  (free, credentialed)\n"
            "and place it alongside your MIMIC-CXR DICOM files."
        )

    return df


def _load_split_df(data_dir: str) -> "pd.DataFrame":
    """Load the official train/validate/test split CSV."""
    if not _PANDAS:
        raise ImportError("pandas is required:  pip install pandas")

    path = Path(data_dir) / _SPLIT_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\n"
            "Download from:  https://physionet.org/content/mimic-cxr-jpg/"
        )
    return pd.read_csv(path)


def _build_label_vector(row: "pd.Series") -> np.ndarray:
    """
    Convert a label-CSV row into a float32 array of shape (14,).

    Convention (CheXpert):
      NaN / missing  →  0.0  (treated as absent)
      -1             →  0.0  (uncertain; treated as absent, common approach)
       0             →  0.0  (negative)
       1             →  1.0  (positive)
    """
    vec = np.zeros(NUM_LABELS, dtype=np.float32)
    for i, col in enumerate(CHEXPERT_LABELS):
        val = row.get(col, float("nan"))
        try:
            v = float(val)
            vec[i] = 1.0 if v == 1.0 else 0.0
        except (ValueError, TypeError):
            vec[i] = 0.0
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# Default image transforms
# ─────────────────────────────────────────────────────────────────────────────

def _default_transform(img_size: int = 224) -> Optional[Callable]:
    """Standard CXR transform: resize + tensor + normalize."""
    if not _TVision:
        return None
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Core Dataset class — mirrors MIMICCXRJPGDataset from mimic_cxr_jpg.py
# ─────────────────────────────────────────────────────────────────────────────

class MIMICCXRJPGDataset(Dataset):
    """
    Torch Dataset for MIMIC-CXR, supporting both the raw DICOM release
    and the MIMIC-CXR-JPG derived release.

    This class is a drop-in replacement for the `MIMICCXRJPGDataset` found
    in `cxr_classification/mimic_cxr_jpg.py`.  The constructor arguments
    are backwards-compatible; new arguments are keyword-only.

    Parameters
    ----------
    datadir : str
        Root of MIMIC-CXR or MIMIC-CXR-JPG download.

    split : str
        One of 'train', 'validate', 'test', or 'all'.
        Filters samples by the official MIMIC-CXR split.

    image_subdir : str
        Sub-directory under `datadir` that holds image files.
        Matches the `cxr_classification` convention.
        Default: 'files'  (the standard layout for both releases).
        For pre-resized images set e.g. 'files512x512'.

    transform : callable | None
        torchvision transform applied to the PIL image.
        If None a sensible default (resize-224, ToTensor, Normalize) is used.

    fmt : str
        'jpg'   — MIMIC-CXR-JPG  (loads .jpg files with PIL)
        'dicom' — raw MIMIC-CXR  (loads .dcm files with pydicom)

    labels_dir : str | None
        Directory containing the CheXpert / NegBio label CSVs.
        Required when fmt='dicom' because raw MIMIC-CXR does not bundle them.
        Ignored when fmt='jpg' (labels are read from datadir by default).

    uncertain_as : float
        Value to assign to uncertain labels (-1 in CheXpert).
        Common choices: 0.0 (ignore) or 1.0 (positive).  Default: 0.0.

    img_size : int
        Target spatial resolution fed to the default transform.
        Ignored if a custom transform is supplied.

    frontal_only : bool
        If True, keep only PA and AP views (drop lateral).
        Default: True (chest X-ray classification convention).

    Returns (per __getitem__)
    ─────────────────────────
    {
        "image"      : FloatTensor (1, img_size, img_size)   grayscale
        "labels"     : FloatTensor (14,)                     CheXpert labels
        "subject_id" : int
        "study_id"   : int
        "dicom_id"   : str
        "split"      : str
    }
    """

    def __init__(
        self,
        datadir:       str,
        split:         str = "train",
        image_subdir:  str = "files",
        transform:     Optional[Callable] = None,
        *,
        fmt:           str = "jpg",
        labels_dir:    Optional[str] = None,
        uncertain_as:  float = 0.0,
        img_size:      int   = 224,
        frontal_only:  bool  = True,
    ):
        assert fmt in ("jpg", "dicom"), f"fmt must be 'jpg' or 'dicom', got {fmt!r}"
        assert split in ("train", "validate", "test", "all"), \
            f"split must be train/validate/test/all, got {split!r}"

        self.datadir      = Path(datadir)
        self.split        = split
        self.image_subdir = image_subdir
        self.fmt          = fmt
        self.uncertain_as = uncertain_as
        self.img_size     = img_size
        self.frontal_only = frontal_only

        # Labels can live in a separate directory (useful for DICOM mode)
        self.labels_dir = Path(labels_dir) if labels_dir else self.datadir

        # Image transform
        self.transform = transform or _default_transform(img_size)

        # Build sample list
        self.samples: List[Dict] = self._build_sample_list()

    # ── Internal: build sample list ──────────────────────────────────────────

    def _build_sample_list(self) -> List[Dict]:
        """
        Join the split CSV with the label CSV on (subject_id, study_id).
        Optionally filter to frontal views only.
        """
        split_df  = _load_split_df(str(self.datadir))
        labels_df = _load_labels_df(str(self.labels_dir))

        # Filter by requested split
        if self.split != "all":
            split_df = split_df[split_df["split"] == self.split]

        # Optionally keep frontal views only
        if self.frontal_only and "ViewPosition" in split_df.columns:
            split_df = split_df[split_df["ViewPosition"].isin(["PA", "AP"])]

        # Merge labels onto split rows
        merged = split_df.merge(
            labels_df,
            on=["subject_id", "study_id"],
            how="inner",
        )

        samples = []
        for _, row in merged.iterrows():
            subject_id = int(row["subject_id"])
            study_id   = int(row["study_id"])
            dicom_id   = str(row["dicom_id"])

            # Build image path ─────────────────────────────────────────────
            # MIMIC-CXR directory layout:
            #   {image_subdir}/p{group}/p{subject_id}/s{study_id}/{dicom_id}.ext
            group_prefix = f"p{str(subject_id)[:2]}"
            ext = "dcm" if self.fmt == "dicom" else "jpg"

            img_path = (
                self.datadir
                / self.image_subdir
                / group_prefix
                / f"p{subject_id}"
                / f"s{study_id}"
                / f"{dicom_id}.{ext}"
            )

            label_vec = _build_label_vector(row)
            # Apply uncertain_as mapping
            if self.uncertain_as != 0.0:
                raw_vec = np.array([
                    float(row.get(col, 0.0) or 0.0) for col in CHEXPERT_LABELS
                ], dtype=np.float32)
                label_vec = np.where(raw_vec == -1.0, self.uncertain_as, label_vec)

            samples.append({
                "image_path": str(img_path),
                "subject_id": subject_id,
                "study_id":   study_id,
                "dicom_id":   dicom_id,
                "labels":     label_vec,
                "split":      row.get("split", self.split),
            })

        return samples

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]

        # Load image
        try:
            if self.fmt == "dicom":
                pil_img = _dicom_to_pil(s["image_path"])
            else:
                pil_img = _jpeg_to_pil(s["image_path"])

            if self.transform:
                image = self.transform(pil_img)
            else:
                # Fallback: pure numpy
                arr   = np.array(pil_img.resize((self.img_size, self.img_size)),
                                 dtype=np.float32) / 255.0
                image = torch.from_numpy(arr).unsqueeze(0)

        except Exception as exc:
            warnings.warn(
                f"Could not load {s['image_path']}: {exc}. "
                "Returning zero tensor.",
                RuntimeWarning,
            )
            image = torch.zeros(1, self.img_size, self.img_size)

        return {
            "image":      image,
            "labels":     torch.tensor(s["labels"], dtype=torch.float32),
            "subject_id": s["subject_id"],
            "study_id":   s["study_id"],
            "dicom_id":   s["dicom_id"],
            "split":      s["split"],
        }

    # ── Convenience ───────────────────────────────────────────────────────────

    def label_names(self) -> List[str]:
        return CHEXPERT_LABELS

    def __repr__(self) -> str:
        return (
            f"MIMICCXRJPGDataset("
            f"fmt={self.fmt!r}, split={self.split!r}, "
            f"n={len(self)}, labels={NUM_LABELS})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Split helpers — mirrors the API from cxr_classification/mimic_cxr_jpg.py
# ─────────────────────────────────────────────────────────────────────────────

def get_mimic_splits(
    datadir:      str,
    *,
    fmt:          str           = "jpg",
    labels_dir:   Optional[str] = None,
    image_subdir: str           = "files",
    transform:    Optional[Callable] = None,
    uncertain_as: float         = 0.0,
    img_size:     int           = 224,
    frontal_only: bool          = True,
    **kwargs,                                 # absorb any extra kwargs
) -> Tuple["MIMICCXRJPGDataset", "MIMICCXRJPGDataset", "MIMICCXRJPGDataset"]:
    """
    Return (train, validate, test) Dataset objects for MIMIC-CXR.

    This is the primary convenience function.  Pass the same arguments as
    the MIMICCXRJPGDataset constructor.

    Example — MIMIC-CXR-JPG:
        train, val, test = get_mimic_splits('/data/mimic-cxr-jpg')

    Example — raw DICOM + labels from MIMIC-CXR-JPG:
        train, val, test = get_mimic_splits(
            '/data/mimic-cxr',
            fmt='dicom',
            labels_dir='/data/mimic-cxr-jpg',
        )

    Example — pre-resized images (512×512):
        train, val, test = get_mimic_splits(
            '/data/mimic-cxr-jpg',
            image_subdir='files512x512',
            img_size=512,
        )
    """
    common = dict(
        datadir=datadir,
        fmt=fmt,
        labels_dir=labels_dir,
        image_subdir=image_subdir,
        transform=transform,
        uncertain_as=uncertain_as,
        img_size=img_size,
        frontal_only=frontal_only,
    )
    return (
        MIMICCXRJPGDataset(split="train",    **common),
        MIMICCXRJPGDataset(split="validate", **common),
        MIMICCXRJPGDataset(split="test",     **common),
    )


def get_patient_split(
    datadir:    str,
    val_frac:   float = 0.1,
    test_frac:  float = 0.1,
    seed:       int   = 42,
    **kwargs,
) -> Tuple["MIMICCXRJPGDataset", "MIMICCXRJPGDataset", "MIMICCXRJPGDataset"]:
    """
    Alternative split by patient ID (no data leakage across splits).
    Useful when the official split CSVs are unavailable or you need
    a custom distribution.

    Mirrors the patient-level split function in cxr_classification.
    """
    full = MIMICCXRJPGDataset(datadir=datadir, split="all", **kwargs)

    # Collect unique subject_ids
    all_subjects = sorted({s["subject_id"] for s in full.samples})
    rng = np.random.default_rng(seed)
    rng.shuffle(all_subjects)

    n        = len(all_subjects)
    n_test   = int(n * test_frac)
    n_val    = int(n * val_frac)
    n_train  = n - n_val - n_test

    train_sids = set(all_subjects[:n_train])
    val_sids   = set(all_subjects[n_train:n_train + n_val])
    test_sids  = set(all_subjects[n_train + n_val:])

    def _subset(sids):
        ds = MIMICCXRJPGDataset(datadir=datadir, split="all", **kwargs)
        ds.samples = [s for s in ds.samples if s["subject_id"] in sids]
        return ds

    return _subset(train_sids), _subset(val_sids), _subset(test_sids)


# ─────────────────────────────────────────────────────────────────────────────
# pFedLLM integration helper
# ─────────────────────────────────────────────────────────────────────────────

class MIMICCXRPFedAdapter(Dataset):
    """
    Thin wrapper around MIMICCXRJPGDataset that emits the full pFedLLM
    batch format:

        image        FloatTensor (1, H, W)
        report_ids   LongTensor  (max_report_len,)   — stub zeros
        attn_mask    LongTensor  (max_report_len,)   — ones for non-padding
        labels       FloatTensor (14,)
        vqa_answer   LongTensor  scalar               — stub 0
        box          FloatTensor (4,)                 — stub zeros

    Plug this into `make_dataloaders()` from utils/data_utils.py instead of
    the raw MIMICCXRJPGDataset when you also need the report / VQA fields.
    """

    def __init__(
        self,
        mimic_ds: MIMICCXRJPGDataset,
        max_report_len: int = 256,
        vocab_size: int = 30522,
    ):
        self.ds             = mimic_ds
        self.max_report_len = max_report_len
        self.vocab_size     = vocab_size

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict:
        item = self.ds[idx]
        L    = self.max_report_len

        # Stub report tokens (replace with real tokeniser if reports available)
        report_ids = torch.zeros(L, dtype=torch.long)
        report_ids[0]  = 101   # [CLS]
        report_ids[-1] = 102   # [SEP]
        attn_mask = torch.ones(L, dtype=torch.long)

        return {
            "image":      item["image"],
            "report_ids": report_ids,
            "attn_mask":  attn_mask,
            "labels":     item["labels"],
            "vqa_answer": torch.tensor(0, dtype=torch.long),
            "box":        torch.zeros(4, dtype=torch.float32),
        }
