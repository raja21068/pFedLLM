"""
models/personalized_head.py
────────────────────────────
Personalized task-specific heads  H_ψc : [z_c, h_c] → ŷ

Each client keeps its own copy of the head, trained entirely locally.
Input to every head is the concatenation [h_c ‖ z_agg] ∈ ℝ^{2d},
where h_c is the client's own compressed feature and z_agg is the
differentially-private shared representation from the server.

Supported tasks (Section V):
    ① disease_classification   – 2-layer MLP  →  sigmoid per class
    ② report_generation        – lightweight transformer decoder
    ③ vqa                      – MLP classifier over answer vocabulary
    ④ visual_grounding         – MLP → 4-dim box regression (x,y,w,h)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. DISEASE CLASSIFICATION HEAD
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    2-layer MLP for multi-label disease classification.

    Input  : concat([h_c, z_agg]) ∈ ℝ^{2d}
    Output : logits ∈ ℝ^{num_classes}  (BCEWithLogitsLoss)

    Metrics: Accuracy, AUC-ROC, F1  (Table V)
    """

    def __init__(self, embed_dim: int = 1024, num_classes: int = 14,
                 hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        in_dim = 2 * embed_dim   # concatenation of h_c and z_agg

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, h_c: torch.Tensor, z_agg: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h_c   : (B, d)  local compressed feature
        z_agg : (B, d)  shared server representation (+ DP noise)

        Returns
        -------
        logits : (B, num_classes)
        """
        x = torch.cat([h_c, z_agg], dim=-1)   # (B, 2d)
        return self.net(x)

    def predict_proba(self, h_c, z_agg):
        return torch.sigmoid(self.forward(h_c, z_agg))


# ─────────────────────────────────────────────────────────────────────────────
# 2. REPORT GENERATION HEAD
# ─────────────────────────────────────────────────────────────────────────────

class ReportGenerationHead(nn.Module):
    """
    Lightweight transformer decoder for radiology report generation.

    Optionally incorporates MRG-LLM prompt-customization via conditional
    affine transformations (Section IV.C, ref [32]).

    Input  : concat([h_c, z_agg]) ∈ ℝ^{2d}  (as memory for cross-attention)
    Output : token logits ∈ ℝ^{T × vocab_size}

    Metrics: BLEU-4, ROUGE-L  (Table IV)
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 256,
        vocab_size: int = 30522,
        max_len: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_mrg_prompt: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len    = max_len
        self.vocab_size = vocab_size

        # Project combined feature into decoder memory
        self.memory_proj = nn.Linear(2 * embed_dim, hidden_dim)

        # Token embeddings + positional
        self.tok_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)

        # MRG-LLM style conditional affine: scale & shift conditioned on h_c
        self.use_mrg = use_mrg_prompt
        if use_mrg_prompt:
            self.gamma_proj = nn.Linear(embed_dim, hidden_dim)
            self.beta_proj  = nn.Linear(embed_dim, hidden_dim)

        # Transformer decoder layers
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.ln_out  = nn.LayerNorm(hidden_dim)
        self.head    = nn.Linear(hidden_dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(
        self,
        h_c: torch.Tensor,
        z_agg: torch.Tensor,
        tgt_ids: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h_c     : (B, d)
        z_agg   : (B, d)
        tgt_ids : (B, T)   teacher-forced token ids
        tgt_mask: (T, T)   causal mask (auto-generated if None)

        Returns
        -------
        logits : (B, T, vocab_size)
        """
        B, T = tgt_ids.shape
        device = tgt_ids.device

        # Build memory from combined feature
        mem = self.memory_proj(torch.cat([h_c, z_agg], dim=-1))  # (B, hidden)
        mem = mem.unsqueeze(1)                                     # (B, 1, hidden)

        # Token embeddings
        positions = torch.arange(T, device=device).unsqueeze(0)   # (1, T)
        x = self.tok_embed(tgt_ids) + self.pos_embed(positions)   # (B, T, hidden)

        # MRG-LLM conditional affine transform (instance-aware adaptation)
        if self.use_mrg:
            gamma = self.gamma_proj(h_c).unsqueeze(1)   # (B, 1, hidden)
            beta  = self.beta_proj(h_c).unsqueeze(1)
            x = gamma * x + beta

        # Causal mask
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)

        out = self.decoder(x, mem, tgt_mask=tgt_mask)  # (B, T, hidden)
        out = self.ln_out(out)
        return self.head(out)                           # (B, T, vocab_size)

    @torch.no_grad()
    def generate(
        self,
        h_c: torch.Tensor,
        z_agg: torch.Tensor,
        bos_id: int = 101,
        eos_id: int = 102,
        max_new_tokens: int = 128,
    ) -> torch.Tensor:
        """Greedy-decode a radiology report token-by-token."""
        B = h_c.size(0)
        device = h_c.device
        ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            logits = self.forward(h_c, z_agg, ids)      # (B, t, vocab)
            next_id = logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            if (next_id == eos_id).all():
                break
        return ids


# ─────────────────────────────────────────────────────────────────────────────
# 3. VISUAL QUESTION ANSWERING HEAD
# ─────────────────────────────────────────────────────────────────────────────

class VQAHead(nn.Module):
    """
    MLP classifier for closed-set VQA on Med-MAT.
    Accuracy is the primary metric (Table VI).

    Input  : concat([h_c, z_agg]) ∈ ℝ^{2d}
    Output : answer logits ∈ ℝ^{num_answers}
    """

    def __init__(self, embed_dim: int = 1024, num_answers: int = 500,
                 hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_answers),
        )

    def forward(self, h_c: torch.Tensor, z_agg: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h_c, z_agg], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUAL GROUNDING HEAD
# ─────────────────────────────────────────────────────────────────────────────

class VisualGroundingHead(nn.Module):
    """
    MLP regressor for bounding-box prediction (visual grounding on Open-I).
    Predicts normalised (x, y, w, h) ∈ [0,1]^4.
    Metrics: Dice, IoU  (Table VII)

    Input  : concat([h_c, z_agg]) ∈ ℝ^{2d}
    Output : box ∈ ℝ^4
    """

    def __init__(self, embed_dim: int = 1024, hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 4),
            nn.Sigmoid(),   # normalise to [0, 1]
        )

    def forward(self, h_c: torch.Tensor, z_agg: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h_c, z_agg], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED PERSONALIZED HEAD (wraps all four tasks)
# ─────────────────────────────────────────────────────────────────────────────

class PersonalizedHead(nn.Module):
    """
    Wrapper that dispatches to the appropriate task head.
    Each client instantiates one PersonalizedHead; parameters ψ_c are
    optimised locally and never transmitted to the server.
    """

    SUPPORTED_TASKS = {
        "disease_classification",
        "report_generation",
        "vqa",
        "visual_grounding",
    }

    def __init__(self, task: str, embed_dim: int = 1024, **task_kwargs):
        super().__init__()
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(f"Unknown task '{task}'. Choose from {self.SUPPORTED_TASKS}")

        self.task = task
        if task == "disease_classification":
            self.head = ClassificationHead(embed_dim=embed_dim, **task_kwargs)
        elif task == "report_generation":
            self.head = ReportGenerationHead(embed_dim=embed_dim, **task_kwargs)
        elif task == "vqa":
            self.head = VQAHead(embed_dim=embed_dim, **task_kwargs)
        elif task == "visual_grounding":
            self.head = VisualGroundingHead(embed_dim=embed_dim, **task_kwargs)

    def forward(self, h_c: torch.Tensor, z_agg: torch.Tensor,
                **kwargs) -> torch.Tensor:
        if self.task == "report_generation":
            return self.head(h_c, z_agg, **kwargs)
        return self.head(h_c, z_agg)

    def loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute task-appropriate loss function."""
        if self.task == "disease_classification":
            # Binary cross-entropy (multi-label)
            return F.binary_cross_entropy_with_logits(logits, targets.float(),
                                                       reduction=reduction)
        elif self.task == "report_generation":
            # MLE (cross-entropy), shift targets
            B, T, V = logits.shape
            return F.cross_entropy(logits.reshape(B * T, V),
                                   targets.reshape(B * T).long(),
                                   ignore_index=0, reduction=reduction)
        elif self.task == "vqa":
            return F.cross_entropy(logits, targets.long(), reduction=reduction)
        elif self.task == "visual_grounding":
            # IoU-aware regression: L1 + GIoU
            return F.l1_loss(logits, targets.float(), reduction=reduction)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_personalized_head(cfg, task: str = None) -> PersonalizedHead:
    """Build a PersonalizedHead from a FederatedConfig object."""
    task = task or cfg.primary_task
    kwargs = {}
    if task == "disease_classification":
        kwargs = dict(num_classes=cfg.num_classes, hidden_dim=cfg.head_hidden_dim)
    elif task == "report_generation":
        kwargs = dict(vocab_size=cfg.vocab_size, max_len=cfg.max_report_len,
                      hidden_dim=cfg.decoder_hidden_dim)
    elif task == "vqa":
        kwargs = dict(hidden_dim=cfg.head_hidden_dim)
    elif task == "visual_grounding":
        kwargs = dict(hidden_dim=cfg.head_hidden_dim)

    return PersonalizedHead(task=task, embed_dim=cfg.embedding_dim, **kwargs)
