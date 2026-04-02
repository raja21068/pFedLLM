"""
models/generative_augmentor.py
───────────────────────────────
Server-pretrained latent-diffusion–based generative augmentor for rare classes.

Section V.C (paper):
    "For clients with few samples of a rare class (e.g., Cardiomegaly),
     we synthesize additional training data. A server-pretrained latent
     diffusion model, conditioned on the class label, generates fake
     compressed features ĥ_c along with paired report embeddings.
     These synthetic samples are mixed 1:1 with real data locally,
     improving rare-class F1 by 11–17%."

Architecture used here:
    Conditional VAE (cVAE) as a lightweight proxy for the full latent
    diffusion model described in the paper. In production, swap
    `LatentDiffusionProxy` for an actual DDPM/DDIM implementation.

Table X reference results:
    Class           w/o Aug   w/ Aug   Δ
    Cardiomegaly    0.54      0.61     +0.07
    Lung Lesion     0.47      0.55     +0.08
    Consolidation   0.52      0.59     +0.07
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Conditional VAE Encoder / Decoder
# ─────────────────────────────────────────────────────────────────────────────

class ConditionalEncoder(nn.Module):
    """q(z | h_c, y)  —  encodes real features + class label into latent."""

    def __init__(self, feat_dim: int, num_classes: int, latent_dim: int = 64):
        super().__init__()
        in_dim = feat_dim + num_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(),
            nn.Linear(256, 128),    nn.GELU(),
        )
        self.mu_head  = nn.Linear(128, latent_dim)
        self.log_head = nn.Linear(128, latent_dim)

    def forward(self, h: torch.Tensor, y_onehot: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        x  = torch.cat([h, y_onehot], dim=-1)
        hid = self.net(x)
        return self.mu_head(hid), self.log_head(hid)


class ConditionalDecoder(nn.Module):
    """p(ĥ_c | z, y)  —  decodes latent + class label → synthetic feature."""

    def __init__(self, feat_dim: int, num_classes: int, latent_dim: int = 64):
        super().__init__()
        in_dim = latent_dim + num_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(),
            nn.Linear(128, 256),    nn.GELU(),
            nn.Linear(256, feat_dim),
        )

    def forward(self, z: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, y_onehot], dim=-1)
        return self.net(x)


class ConditionalVAE(nn.Module):
    """
    Conditional VAE as lightweight proxy for the latent diffusion model.

    Pretrained on server data; used during federated rounds to generate
    synthetic h_c features for rare classes. Clients never receive
    the VAE parameters directly — they receive the generated features.
    """

    def __init__(self, feat_dim: int = 1024, num_classes: int = 14,
                 latent_dim: int = 64):
        super().__init__()
        self.feat_dim   = feat_dim
        self.num_classes = num_classes
        self.latent_dim  = latent_dim

        self.encoder = ConditionalEncoder(feat_dim, num_classes, latent_dim)
        self.decoder = ConditionalDecoder(feat_dim, num_classes, latent_dim)

    def reparameterise(self, mu: torch.Tensor,
                       log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        h: torch.Tensor,           # (B, feat_dim)
        y_onehot: torch.Tensor,    # (B, num_classes)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        h_recon : (B, feat_dim)   reconstruction
        mu      : (B, latent_dim)
        log_var : (B, latent_dim)
        """
        mu, log_var = self.encoder(h, y_onehot)
        z           = self.reparameterise(mu, log_var)
        h_recon     = self.decoder(z, y_onehot)
        return h_recon, mu, log_var

    @torch.no_grad()
    def generate(
        self,
        class_id:    int,
        num_samples: int,
        device:      torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Sample num_samples synthetic features for class `class_id`.

        Returns
        -------
        h_synth : (num_samples, feat_dim)
        """
        y_onehot = torch.zeros(num_samples, self.num_classes, device=device)
        y_onehot[:, class_id] = 1.0

        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decoder(z, y_onehot)

    def loss(
        self,
        h_recon: torch.Tensor,
        h_orig:  torch.Tensor,
        mu:      torch.Tensor,
        log_var: torch.Tensor,
        beta:    float = 1.0,
    ) -> torch.Tensor:
        """
        ELBO loss = reconstruction loss + β · KL divergence.
        β=1 → standard VAE; β<1 → more weight on reconstruction fidelity.
        """
        recon_loss = F.mse_loss(h_recon, h_orig, reduction="mean")
        kl_loss    = -0.5 * torch.mean(
            1 + log_var - mu.pow(2) - log_var.exp()
        )
        return recon_loss + beta * kl_loss


# ─────────────────────────────────────────────────────────────────────────────
# High-level GenerativeAugmentor  (used by FederatedClient)
# ─────────────────────────────────────────────────────────────────────────────

class GenerativeAugmentor:
    """
    Wraps the cVAE to provide a simple augment() interface used in
    FederatedClient.train_step().

    The augmentor is pre-trained on the server before federated rounds;
    clients call augment() locally using generated features.

    Augmentation ratio: 1:1 real-to-synthetic (Section V.C).
    """

    def __init__(
        self,
        feat_dim:    int = 1024,
        num_classes: int = 14,
        latent_dim:  int = 64,
        device:      torch.device = torch.device("cpu"),
        seed:        int = 42,
    ):
        torch.manual_seed(seed)
        self.device      = device
        self.num_classes = num_classes
        self.feat_dim    = feat_dim

        self.model = ConditionalVAE(feat_dim, num_classes, latent_dim).to(device)
        self._trained = False

    # ── Pre-training (server-side) ────────────────────────────────────────

    def pretrain(
        self,
        h_all:       torch.Tensor,    # (N, feat_dim) real features
        y_all:       torch.Tensor,    # (N, num_classes) labels
        num_epochs:  int = 20,
        lr:          float = 1e-3,
        batch_size:  int = 64,
    ) -> List[float]:
        """
        Train the cVAE on server data.
        In production: use the actual MIMIC-CXR feature bank.
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        losses: List[float] = []
        N = len(h_all)

        for epoch in range(num_epochs):
            perm = torch.randperm(N)
            epoch_loss = 0.0; steps = 0
            for i in range(0, N - batch_size, batch_size):
                idx       = perm[i:i + batch_size]
                h_b       = h_all[idx].to(self.device)
                y_b       = y_all[idx].float().to(self.device)
                h_r, mu, lv = self.model(h_b, y_b)
                loss      = self.model.loss(h_r, h_b, mu, lv)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item(); steps += 1
            avg = epoch_loss / max(steps, 1)
            losses.append(avg)
            if (epoch + 1) % 5 == 0:
                print(f"  [Augmentor pretrain epoch {epoch+1}/{num_epochs}] "
                      f"loss = {avg:.4f}")

        self._trained = True
        self.model.eval()
        return losses

    # ── Augmentation (client-side call) ──────────────────────────────────

    @torch.no_grad()
    def augment(
        self,
        h_real:      torch.Tensor,    # (B, feat_dim) real features
        labels_real: torch.Tensor,    # (B, num_classes) real labels
        rare_class_ids: List[int],    # class indices to augment
        ratio:       float = 1.0,     # synthetic-to-real ratio per rare class
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mix real and synthetic features for rare classes.

        Returns
        -------
        h_aug    : (B', feat_dim)   combined real + synthetic features
        y_aug    : (B', num_classes) corresponding labels
        """
        h_list = [h_real]
        y_list = [labels_real]

        for cls_id in rare_class_ids:
            mask  = labels_real[:, cls_id] == 1
            n_real = mask.sum().item()
            if n_real == 0:
                continue
            n_synth = max(1, int(n_real * ratio))

            if self._trained:
                h_synth = self.model.generate(cls_id, n_synth, self.device)
            else:
                # Fallback: Gaussian noise around real class mean
                proto   = h_real[mask].mean(0, keepdim=True)
                noise   = torch.randn(n_synth, self.feat_dim,
                                      device=self.device) * 0.1
                h_synth = proto.expand(n_synth, -1) + noise

            y_synth = torch.zeros(n_synth, self.num_classes,
                                  device=self.device)
            y_synth[:, cls_id] = 1.0

            h_list.append(h_synth)
            y_list.append(y_synth)

        if len(h_list) == 1:
            return h_real, labels_real

        h_all = torch.cat(h_list, dim=0)
        y_all = torch.cat(y_list, dim=0)

        # Shuffle combined batch
        perm = torch.randperm(len(h_all), device=self.device)
        return h_all[perm], y_all[perm]

    def state_dict(self) -> Dict:
        return {"model": self.model.state_dict(), "trained": self._trained}

    def load_state_dict(self, state: Dict):
        self.model.load_state_dict(state["model"])
        self._trained = state.get("trained", False)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_augmentor(cfg, device: torch.device) -> GenerativeAugmentor:
    return GenerativeAugmentor(
        feat_dim=cfg.embedding_dim,
        num_classes=cfg.num_classes,
        device=device,
    )
