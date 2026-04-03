"""
configs/default.py
──────────────────
All hyperparameters and configuration for pFedLLM.
Values match Table II from the paper unless noted.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FederatedConfig:
    # ── Federated Training ──────────────────────────────────────────
    num_clients: int = 10              # |C| - number of participating hospitals
    rounds: int = 50                   # R  - communication rounds
    local_epochs: int = 1              # E  - local epochs per round
    client_fraction: float = 1.0       # fraction of clients per round

    # ── Model Architecture ──────────────────────────────────────────
    embedding_dim: int = 1024          # d  - latent feature dimension
    image_size: int = 224              # input image spatial resolution
    image_channels: int = 1            # 1 for chest X-ray (grayscale)
    num_classes: int = 14              # CheXpert/MIMIC-CXR disease labels
    vocab_size: int = 30522            # BERT tokenizer vocab size
    max_report_len: int = 256          # max tokens in radiology report

    # Compressor: 3 conv blocks + GAP + linear (Section IV.C)
    compressor_channels: List[int] = field(
        default_factory=lambda: [32, 64, 128]
    )
    compressor_kernel: int = 3

    # Personalized head dimensions
    head_hidden_dim: int = 256         # MLP hidden dim for classification
    decoder_hidden_dim: int = 256      # Decoder hidden dim for report gen

    # ── Server LLM ──────────────────────────────────────────────────
    llm_backbone: str = "medgemini"    # "medgemini" | "glm45v" | "medr1"
    # For demo: use a small HuggingFace model as proxy
    hf_model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    llm_embed_dim: int = 768           # backbone hidden size
    llm_projection_dim: int = 1024     # projected to match client embedding_dim

    # ── Optimization ────────────────────────────────────────────────
    optimizer: str = "adam"
    learning_rate: float = 1e-4
    batch_size: int = 32
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # ── Differential Privacy (Section IV.F.c) ───────────────────────
    use_dp: bool = True
    dp_noise_multiplier: float = 0.6   # σ - Gaussian noise std
    dp_max_grad_norm: float = 1.0      # per-batch gradient clipping (C)
    dp_delta: float = 1e-5             # δ for (ε, δ)-DP

    # ── Data Partitioning ────────────────────────────────────────────
    partition: str = "non_iid_temporal"  # "iid" | "non_iid_temporal" | "non_iid_clinical"
    dirichlet_alpha: float = 0.5       # concentration for non-IID split
    use_synthetic: bool = False        # use synthetic data for quick testing

    # ── Generative Augmentation (Section V.C) ───────────────────────
    use_gen_aug: bool = True
    aug_ratio: float = 1.0             # 1:1 real-to-synthetic ratio
    rare_class_threshold: int = 50     # samples below this get augmented

    # ── Multi-Stage Fine-Tuning ──────────────────────────────────────
    pretrain_rounds: int = 5           # warm-up rounds (stage 1)
    finetune_rounds: int = 45          # main FL rounds (stage 2)

    # ── Aggregation ──────────────────────────────────────────────────
    aggregation: str = "weighted_avg"  # "weighted_avg" | "attention"

    # ── Tasks ────────────────────────────────────────────────────────
    tasks: List[str] = field(
        default_factory=lambda: [
            "report_generation",
            "disease_classification",
            "vqa",
            "visual_grounding",
        ]
    )
    primary_task: str = "disease_classification"

    # ── Logging & Checkpointing ───────────────────────────────────────
    log_every: int = 5
    eval_every: int = 10
    save_every: int = 10
    output_dir: str = "./outputs"
    seed: int = 42

    # ── Dataset ──────────────────────────────────────────────────────
    dataset: str = "mimic_cxr"        # "mimic_cxr" | "medtrinity" | "medmat"
    data_dir: Optional[str] = None
    num_workers: int = 4

    # ── MLRG Checkpoint (optional) ───────────────────────────────────
    mlrg_ckpt_path: Optional[str] = None  # path to mimic-cxr/finetune/best_model.ckpt
    
    
@dataclass
class AblationConfig:
    """For reproducing ablation studies from Section V.I."""
    ablation_type: str = "full"        # "full" | "no_dp" | "no_aug" | "shared_head"
                                        #   | "vision_only" | "dim_256" | "dim_512" | "dim_2048"
    base_config: FederatedConfig = field(default_factory=FederatedConfig)


# ── Convenience factory ──────────────────────────────────────────────────────

def get_config(name: str = "default") -> FederatedConfig:
    presets = {
        "default": FederatedConfig(),
        "fast": FederatedConfig(rounds=5, num_clients=3, use_synthetic=True,
                                 batch_size=8, local_epochs=1),
        "iid":  FederatedConfig(partition="iid"),
        "non_iid": FederatedConfig(partition="non_iid_temporal"),
        "no_dp": FederatedConfig(use_dp=False, dp_noise_multiplier=0.0),
        "glm45v": FederatedConfig(llm_backbone="glm45v"),
    }
    return presets.get(name, FederatedConfig())
