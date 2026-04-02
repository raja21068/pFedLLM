"""
models/server_llm.py
─────────────────────
Server-side multimodal LLM  F_θ : (h_c, R) → z_c

The server hosts a pre-trained multimodal LLM (Med-Gemini / GLM-4.5V).
It accepts compressed image features h_c and de-identified report text R,
and returns a shared semantic embedding z_c.

Key properties (Section IV.B, IV.C):
  • F_θ is FIXED during federated rounds (no backpropagation on server).
  • Clients never receive F_θ's parameters (>10 GB).
  • Only z_c (≈4 KB) is returned per sample.
  • Differential privacy noise ε_DP is added to z_agg before broadcast.
  • Aggregation: z_agg = Σ_c (|D_c|/|D|) · z_c  +  ε_DP

In production: swap ServerLLM for the actual Med-Gemini or GLM-4.5V API.
For local development: uses a small BioMedBERT-style encoder as proxy.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Cross-modal Fusion Module
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalFusion(nn.Module):
    """
    Fuses the image embedding h_c and text (report) embedding t_c via
    cross-attention, mimicking the vision–language encoder in Med-Gemini.

    Input : h_img ∈ ℝ^{B×d_img}, h_txt ∈ ℝ^{B×T×d_txt}
    Output: z_c ∈ ℝ^{B×d_out}
    """

    def __init__(self, d_img: int, d_txt: int, d_out: int,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # Project both modalities to a common dimension
        d_joint = d_out
        self.img_proj = nn.Linear(d_img, d_joint)
        self.txt_proj = nn.Linear(d_txt, d_joint)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_joint, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.ln1 = nn.LayerNorm(d_joint)
        self.ff  = nn.Sequential(
            nn.Linear(d_joint, d_joint * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_joint * 4, d_joint),
        )
        self.ln2 = nn.LayerNorm(d_joint)

    def forward(
        self,
        h_img: torch.Tensor,          # (B, d_img)
        h_txt: torch.Tensor,          # (B, T, d_txt)
    ) -> torch.Tensor:                # (B, d_out)
        q = self.img_proj(h_img).unsqueeze(1)   # (B, 1, d_joint) — image as query
        kv = self.txt_proj(h_txt)               # (B, T, d_joint) — text as key/value

        attn_out, _ = self.cross_attn(q, kv, kv)   # (B, 1, d_joint)
        attn_out = self.ln1(q + attn_out)
        out = self.ln2(attn_out + self.ff(attn_out))
        return out.squeeze(1)                       # (B, d_joint)


# ─────────────────────────────────────────────────────────────────────────────
# Text Encoder (proxy for the LLM text decoder/encoder)
# ─────────────────────────────────────────────────────────────────────────────

class LightTextEncoder(nn.Module):
    """
    Lightweight BERT-style text encoder used as a local proxy for
    the actual Med-Gemini text decoder.

    In production, replace this with:
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained("microsoft/BiomedNLP-BiomedBERT-...")
    """

    def __init__(self, vocab_size: int = 30522, d_model: int = 256,
                 max_len: int = 256, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.tok_embed  = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed  = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.ln      = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(
        self,
        input_ids: torch.Tensor,        # (B, T)
        attention_mask: Optional[torch.Tensor] = None,   # (B, T)
    ) -> torch.Tensor:                  # (B, T, d_model)
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x    = self.tok_embed(input_ids) + self.pos_embed(pos)
        # Convert attention mask to padding mask for Transformer
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.ln(out)


# ─────────────────────────────────────────────────────────────────────────────
# Server-side Multimodal LLM
# ─────────────────────────────────────────────────────────────────────────────

class ServerLLM(nn.Module):
    """
    F_θ  — server-side multimodal LLM.

    Workflow (Algorithm 1, steps 10–13):
        1. Receive h_c (image features) and R (tokenised report) from client c.
        2. Encode R with the text encoder → h_txt.
        3. Fuse (h_c, h_txt) via cross-modal attention → z_c.
        4. Aggregate {z_c} into z_agg  (weighted average + DP noise).
        5. Broadcast z_agg to all clients.

    Parameters
    ----------
    embed_dim : int
        Dimension of client image embeddings (h_c).  Default 1024.
    text_hidden : int
        Hidden dimension of the text encoder.
    output_dim : int
        Dimension of z_c returned to clients.  Default 1024.
    freeze : bool
        If True, all parameters are frozen (simulates a fixed pre-trained LLM).
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        text_hidden: int = 256,
        output_dim: int = 1024,
        vocab_size: int = 30522,
        max_report_len: int = 256,
        freeze: bool = True,
        use_pretrained_hf: bool = False,
        hf_model_name: str = "",
    ):
        super().__init__()
        self.embed_dim  = embed_dim
        self.output_dim = output_dim

        # ── Text encoder ──────────────────────────────────────────────
        if use_pretrained_hf and hf_model_name:
            # Production: load actual BioMedBERT / Gemini text encoder
            try:
                from transformers import AutoModel
                self.text_encoder = AutoModel.from_pretrained(hf_model_name)
                txt_dim = self.text_encoder.config.hidden_size
            except Exception as e:
                print(f"[ServerLLM] HF model load failed ({e}); using proxy encoder.")
                self.text_encoder = LightTextEncoder(vocab_size, text_hidden, max_report_len)
                txt_dim = text_hidden
        else:
            self.text_encoder = LightTextEncoder(vocab_size, text_hidden, max_report_len)
            txt_dim = text_hidden

        # ── Cross-modal fusion ────────────────────────────────────────
        self.fusion = CrossModalFusion(
            d_img=embed_dim, d_txt=txt_dim, d_out=output_dim
        )

        # ── Output projection (optional extra non-linearity) ──────────
        self.out_proj = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
        )

        if freeze:
            self.freeze_parameters()

    def freeze_parameters(self):
        """Freeze all parameters — server does no backprop (Section IV.C.e)."""
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze_parameters(self):
        """Allow gradient updates during the pre-training / fine-tuning stage."""
        for p in self.parameters():
            p.requires_grad_(True)

    def encode_text(
        self,
        report_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode de-identified report tokens → text embeddings."""
        with torch.no_grad():
            if hasattr(self.text_encoder, 'config'):
                # HuggingFace model
                out = self.text_encoder(report_ids, attention_mask=attention_mask)
                return out.last_hidden_state    # (B, T, hidden)
            else:
                return self.text_encoder(report_ids, attention_mask)   # (B, T, hidden)

    @torch.no_grad()
    def forward(
        self,
        h_c: torch.Tensor,                        # (B, d) client image feature
        report_ids: torch.Tensor,                  # (B, T) tokenised report
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:                             # (B, output_dim)
        """
        F_θ(h_c, R) → z_c

        The server runs this in no-grad mode; it is not part of the
        optimisation graph (Section IV.C.e: 'Fθ remains fixed').
        """
        h_txt = self.encode_text(report_ids, attention_mask)   # (B, T, txt_dim)
        z_c   = self.fusion(h_c, h_txt)                        # (B, output_dim)
        z_c   = self.out_proj(z_c)
        return z_c

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# GLM-4.5V Backbone Stub (Section IV.C.b)
# ─────────────────────────────────────────────────────────────────────────────

class GLM45VBackbone(nn.Module):
    """
    Stub representing GLM-4.5V (3D-RoPE + MoE) server backbone.
    Replace the forward() body with the real GLM-4.5V API call.

    Reduces communication by 10-15% vs Med-Gemini due to MoE activation
    sparsity; converges in 29 rounds vs 37 (Table XII).
    """

    def __init__(self, embed_dim: int = 1024, output_dim: int = 1024,
                 num_experts: int = 8, top_k_experts: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k_experts

        # Mixture-of-Experts routing
        self.gate   = nn.Linear(embed_dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.GELU(),
                          nn.Linear(embed_dim * 2, embed_dim))
            for _ in range(num_experts)
        ])
        self.proj = nn.Linear(embed_dim, output_dim)

    @torch.no_grad()
    def forward(self, h_c: torch.Tensor, **kwargs) -> torch.Tensor:
        """Simplified MoE forward (image-only; text ignored in stub)."""
        # Gating
        logits = self.gate(h_c)                          # (B, E)
        topk_val, topk_idx = logits.topk(self.top_k, dim=-1)
        weights = F.softmax(topk_val, dim=-1)            # (B, top_k)

        # Sparse expert computation
        out = torch.zeros_like(h_c)
        for k in range(self.top_k):
            idx = topk_idx[:, k]   # (B,) which expert
            w   = weights[:, k:k+1]
            for e in range(self.num_experts):
                mask = (idx == e)
                if mask.any():
                    out[mask] += w[mask] * self.experts[e](h_c[mask])

        return self.proj(out)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_server_llm(cfg) -> ServerLLM:
    return ServerLLM(
        embed_dim=cfg.embedding_dim,
        output_dim=cfg.embedding_dim,
        vocab_size=cfg.vocab_size,
        max_report_len=cfg.max_report_len,
        freeze=True,
        use_pretrained_hf=False,
    )
