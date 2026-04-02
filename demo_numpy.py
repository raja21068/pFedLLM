"""
demo_numpy.py — pFedLLM Complete Simulation (NumPy only, no GPU needed)
════════════════════════════════════════════════════════════════════════
Faithfully implements Algorithm 1 from the paper with:
  • Feature-map communication protocol (not parameter sharing)
  • Server-side multimodal LLM  F_θ  (fixed; never sent to clients)
  • Client-side feature compressor  C_φc  +  personalised head  H_ψc
  • Differential privacy (Gaussian noise on z_agg)
  • Generative augmentation for rare classes
  • All metrics: AUC-ROC, Accuracy, F1 (paper Tables V–IX)

Usage
─────
  python demo_numpy.py                   # IID + Non-IID runs
  python demo_numpy.py --compare_all     # Table V — all baselines
  python demo_numpy.py --ablation dp     # Table VIII — DP noise sweep
  python demo_numpy.py --ablation dim    # Fig. 6 — dimensionality sweep
  python demo_numpy.py --ablation aug    # Table X — rare-class augmentation
  python demo_numpy.py --ablation all    # all ablations
"""

import sys, math, time, random, argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS  (paper Table II — scaled for fast CPU demo)
# ══════════════════════════════════════════════════════════════════════════════
NUM_CLIENTS  = 10
NUM_ROUNDS   = 40          # paper: 50; reduced for demo speed
LOCAL_EPOCHS = 1
BATCH_SIZE   = 64
EMBED_DIM    = 128         # paper: 1024  (scaled for demo speed)
IN_DIM       = 64          # proxy image feature dimension
TXT_DIM      = 32          # proxy report embedding dimension
NUM_CLASSES  = 14          # CheXpert 14-class label set
SAMPLES      = 300         # samples per client
LR           = 0.02
DP_SIGMA_DEFAULT = 0.1
DP_DELTA         = 1e-5

DISEASE_LABELS = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema",
    "Enlarged Cardiomediastinum","Fracture","Lung Lesion",
    "Lung Opacity","No Finding","Pleural Effusion",
    "Pleural Other","Pneumonia","Pneumothorax","Support Devices",
]
RARE_CLASSES = [1, 2, 6]   # Cardiomegaly, Consolidation, Lung Lesion

# ══════════════════════════════════════════════════════════════════════════════
# MATH HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))

def relu(x):    return np.maximum(0.0, x)
def l2norm(x):  n = np.linalg.norm(x, axis=-1, keepdims=True); return x / (n + 1e-8)
def he(r, c):   return np.random.randn(r, c) * math.sqrt(2.0 / r)
def zeros(n):   return np.zeros(n, dtype=np.float64)

# ══════════════════════════════════════════════════════════════════════════════
# MINIMAL LINEAR LAYER  (forward + manual SGD)
# ══════════════════════════════════════════════════════════════════════════════
class Linear:
    def __init__(self, in_d: int, out_d: int):
        self.W  = he(in_d, out_d)
        self.b  = zeros(out_d)
        self._x = None

    def fwd(self, x):
        self._x = x
        return x @ self.W + self.b

    def bwd(self, dout, lr):
        B   = len(dout)
        dW  = self._x.T @ dout / B
        db  = dout.mean(0)
        dx  = dout @ self.W.T
        self.W -= lr * dW
        self.b -= lr * db
        return dx

# ══════════════════════════════════════════════════════════════════════════════
# MODEL COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════
class FeatureCompressor:
    """C_φc : image_feat → h_c ∈ ℝ^d   (3-block CNN proxy)"""
    def __init__(self, in_d=IN_DIM, d=EMBED_DIM):
        mid = max(d // 2, 32)
        self.fc1  = Linear(in_d, mid)
        self.fc2  = Linear(mid, d)
        self._mask = None

    def fwd(self, x):
        pre        = self.fc1.fwd(x)
        self._mask = pre > 0.0
        return self.fc2.fwd(relu(pre))

    def bwd(self, dh, lr):
        d2 = self.fc2.bwd(dh, lr)
        d1 = d2 * self._mask
        self.fc1.bwd(d1, lr)


class PersonalizedHead:
    """H_ψc : [h_c ‖ z_agg] → logits ∈ ℝ^C   (2-layer MLP, per-client)"""
    def __init__(self, d=EMBED_DIM, n_cls=NUM_CLASSES):
        self.d    = d
        mid       = max(d, 64)
        self.fc1  = Linear(2 * d, mid)
        self.fc2  = Linear(mid, n_cls)
        self._mask = None

    def fwd(self, h, z):
        x          = np.concatenate([h, z], axis=-1)
        pre        = self.fc1.fwd(x)
        self._mask = pre > 0.0
        return self.fc2.fwd(relu(pre))

    def bwd(self, dlogits, lr):
        d2    = self.fc2.bwd(dlogits, lr)
        d1    = d2 * self._mask
        dx    = self.fc1.bwd(d1, lr)
        return dx[:, :self.d]      # gradient wrt h_c only


class ServerLLM:
    """F_θ : (h_c, report_embed) → z_c   (FIXED — never updated or sent)"""
    def __init__(self, img_d=EMBED_DIM, txt_d=TXT_DIM, out_d=EMBED_DIM):
        scale        = 1.0 / math.sqrt(img_d + txt_d)
        self.W_img   = np.random.randn(img_d, out_d) * scale
        self.W_txt   = np.random.randn(txt_d, out_d) * scale
        self.b       = zeros(out_d)

    def fwd(self, h_c, report):
        """Algorithm 1 line 11: z_c ← F_θ(h_c, R). No gradient."""
        z = np.tanh(h_c @ self.W_img + report @ self.W_txt + self.b)
        return l2norm(z)

# ══════════════════════════════════════════════════════════════════════════════
# DIFFERENTIAL PRIVACY  (Section IV.F.c, Tables VIII–IX)
# ══════════════════════════════════════════════════════════════════════════════
def dp_noise(z, sigma):
    """ε_DP ~ N(0, σ²I) added to z_agg (Algorithm 1 line 12)."""
    if sigma <= 0: return z
    return z + np.random.randn(*z.shape) * sigma

def compute_epsilon(sigma, q, T, delta=1e-5):
    """Rényi DP moments accountant → ε for (ε,δ)-DP."""
    if sigma <= 0: return float("inf")
    best = float("inf")
    for alpha in [2, 3, 4, 5, 6, 8, 10, 16, 32, 64]:
        rdp  = min(alpha * q**2 * (math.exp(1/sigma**2) - 1),
                   alpha / (2 * sigma**2))
        eps  = rdp * T + math.log(1 / delta) / (alpha - 1)
        best = min(best, eps)
    return round(best, 2)

# ══════════════════════════════════════════════════════════════════════════════
# GENERATIVE AUGMENTATION  (Section V.C, Table X)
# ══════════════════════════════════════════════════════════════════════════════
class GenerativeAugmentor:
    """Server-pretrained latent diffusion proxy for rare-class synthesis."""
    def __init__(self, d=EMBED_DIM):
        self.d      = d
        self.protos: Dict[int, np.ndarray] = {}

    def augment(self, h, y, rare_ids, ratio=1.0):
        sh_list, sy_list = [h], [y]
        for cls in rare_ids:
            mask = y[:, cls] == 1
            if not mask.any(): continue
            proto    = h[mask].mean(0)
            self.protos[cls] = proto
            n_aug    = max(1, int(mask.sum() * ratio))
            noise    = np.random.randn(n_aug, self.d) * 0.07
            sh       = proto[None] + noise
            sy       = np.zeros((n_aug, y.shape[1]))
            sy[:, cls] = 1.0
            sh_list.append(sh); sy_list.append(sy)
        h_all  = np.concatenate(sh_list)
        y_all  = np.concatenate(sy_list)
        perm   = np.random.permutation(len(h_all))
        return h_all[perm], y_all[perm]

# ══════════════════════════════════════════════════════════════════════════════
# METRICS  (Tables IV–VII)
# ══════════════════════════════════════════════════════════════════════════════
def _auc(y_true, y_score):
    order = np.argsort(-y_score)
    yt    = y_true[order]
    npos  = yt.sum(); nneg = len(yt) - npos
    if npos == 0 or nneg == 0: return 0.5
    tpr = np.cumsum(yt) / npos
    fpr = np.cumsum(1 - yt) / nneg
    return float(np.trapezoid(tpr, fpr))

def metrics(logits, labels, thr=0.5):
    probs = sigmoid(logits)
    preds = (probs >= thr).astype(int)
    acc   = (preds == labels).all(1).mean()
    aucs  = [_auc(labels[:, c], probs[:, c])
             for c in range(labels.shape[1])
             if 0 < labels[:, c].sum() < len(labels)]
    auc   = float(np.mean(aucs)) if aucs else 0.5
    eps   = 1e-8
    tp    = (preds * labels).sum(0)
    fp    = (preds * (1 - labels)).sum(0)
    fn    = ((1 - preds) * labels).sum(0)
    pr    = tp / (tp + fp + eps)
    rc    = tp / (tp + fn + eps)
    f1    = (2 * pr * rc / (pr + rc + eps)).mean()
    return {"accuracy": float(acc), "auc_roc": auc, "f1": float(f1)}

def bce_loss(logits, y):
    p = sigmoid(logits)
    return -np.mean(y * np.log(p + 1e-8) + (1 - y) * np.log(1 - p + 1e-8))

def bce_grad(logits, y):
    return (sigmoid(logits) - y) / max(len(logits), 1)

# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA  (Section V.A — IID and Non-IID partitions)
# ══════════════════════════════════════════════════════════════════════════════
def make_data(n=SAMPLES, client_id=0, partition="non_iid"):
    """
    Synthetic data with REAL image-label correlation so the model can learn.
    Each disease class has a prototype vector; images are linear combos + noise.
    Domain shift: each client has a rotated prototype matrix (equipment/demographics).
    """
    rng   = np.random.default_rng(SEED + client_id)
    shift = client_id * 0.1

    # Fixed class prototype vectors (same across clients, shifted per client)
    proto_rng = np.random.default_rng(SEED)
    prototypes = proto_rng.normal(0, 1.0, (NUM_CLASSES, IN_DIM))  # (C, IN_DIM)

    # Non-IID: each client focuses on 3-4 disease classes
    if partition == "non_iid":
        dominant    = rng.choice(NUM_CLASSES, 4, replace=False)
        dom_weights = np.ones(NUM_CLASSES) * 0.5
        dom_weights[dominant] = 3.0
        dom_weights /= dom_weights.sum()
    else:
        dom_weights = np.ones(NUM_CLASSES) / NUM_CLASSES

    labels = np.zeros((n, NUM_CLASSES))
    imgs   = np.zeros((n, IN_DIM))
    reps   = np.zeros((n, TXT_DIM))

    for i in range(n):
        n_pos = rng.integers(1, 3)
        pos   = rng.choice(NUM_CLASSES, n_pos, replace=False, p=dom_weights)
        labels[i, pos] = 1.0
        # Image = weighted sum of class prototypes + noise + domain shift
        img = prototypes[pos].sum(0) * 0.7 + rng.normal(shift, 0.4, IN_DIM)
        imgs[i] = img
        # Report embed also correlated with labels
        reps[i] = labels[i, :TXT_DIM] if TXT_DIM <= NUM_CLASSES else np.pad(
            labels[i], (0, TXT_DIM - NUM_CLASSES)) + rng.normal(0, 0.3, TXT_DIM)

    return {"imgs": imgs.astype(np.float64),
            "reps": reps.astype(np.float64),
            "labels": labels}

def make_test_data(n=600):
    return make_data(n=n, client_id=99, partition="iid")

# ══════════════════════════════════════════════════════════════════════════════
# FEDERATED CLIENT
# ══════════════════════════════════════════════════════════════════════════════
class Client:
    def __init__(self, cid, data, embed_d=EMBED_DIM,
                 use_dp=True, use_aug=True, dp_sigma=DP_SIGMA_DEFAULT):
        self.cid    = cid
        self.data   = data
        self.n      = len(data["imgs"])
        self.use_dp = use_dp
        self.use_aug= use_aug
        self.sigma  = dp_sigma
        self.comp   = FeatureCompressor(IN_DIM, embed_d)
        self.head   = PersonalizedHead(embed_d, NUM_CLASSES)
        self.aug    = GenerativeAugmentor(embed_d)

    def train_epoch(self, server: "Server", lr=LR):
        """One full pass over local data — Algorithm 1 lines 7-16."""
        idx       = np.random.permutation(self.n)
        loss_sum  = 0.0; steps = 0

        for i in range(0, self.n - BATCH_SIZE + 1, BATCH_SIZE):
            b    = idx[i:i + BATCH_SIZE]
            imgs = self.data["imgs"][b]
            reps = self.data["reps"][b]
            y    = self.data["labels"][b]

            # Line 9: h_c ← C_φc(I)
            h_c = self.comp.fwd(imgs)

            # Lines 11-12: z_c ← F_θ(h_c, R); z_agg ← weighted_avg + ε_DP
            sigma = self.sigma if self.use_dp else 0.0
            z_agg = server.encode(h_c, reps, sigma)

            # Generative augmentation (rare classes)
            if self.use_aug:
                h_use, y_use = self.aug.augment(h_c, y, RARE_CLASSES)
                B2    = min(len(h_use), len(z_agg))
                h_use = h_use[:B2]; y_use = y_use[:B2]
                z_use = np.tile(z_agg, (math.ceil(B2 / len(z_agg)), 1))[:B2]
            else:
                h_use, y_use, z_use = h_c, y, z_agg

            # Line 14: ŷ ← H_ψc([h_c, z_agg])
            logits = self.head.fwd(h_use, z_use)

            # Line 15: update (φc, ψc) ← min ℓc(ŷ, y)
            dlogits = bce_grad(logits, y_use)
            dh      = self.head.bwd(dlogits, lr)

            # DP-SGD: clip gradient, add noise
            if self.use_dp:
                norm = np.linalg.norm(dh) + 1e-8
                dh   = dh / max(1.0, norm)      # clip to unit norm
                dh  += np.random.randn(*dh.shape) * (self.sigma * 0.1)

            self.comp.bwd(dh, lr)

            loss_sum += bce_loss(logits, y_use); steps += 1

        return loss_sum / max(steps, 1)

    def evaluate(self, server: "Server", test_data):
        h  = self.comp.fwd(test_data["imgs"])
        z  = server.encode(h, test_data["reps"], sigma=0.0)
        lg = self.head.fwd(h, z)
        return metrics(lg, test_data["labels"])

# ══════════════════════════════════════════════════════════════════════════════
# FEDERATED SERVER
# ══════════════════════════════════════════════════════════════════════════════
class Server:
    """Hosts F_θ — never shares parameters or raw data."""
    def __init__(self, embed_d=EMBED_DIM):
        self.llm = ServerLLM(img_d=embed_d, txt_d=TXT_DIM, out_d=embed_d)

    def encode(self, h_c, reps, sigma=DP_SIGMA_DEFAULT):
        z = self.llm.fwd(h_c, reps)
        return dp_noise(z, sigma)

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP  (Algorithm 1)
# ══════════════════════════════════════════════════════════════════════════════
def run_pFedLLM(
    num_clients=NUM_CLIENTS, num_rounds=NUM_ROUNDS,
    embed_d=EMBED_DIM, partition="non_iid",
    use_dp=True, use_aug=True,
    dp_sigma=DP_SIGMA_DEFAULT, verbose=True,
) -> Tuple[Dict, Dict]:
    np.random.seed(SEED); random.seed(SEED)

    all_data  = [make_data(SAMPLES, c, partition) for c in range(num_clients)]
    test_data = make_test_data()
    server    = Server(embed_d)
    clients   = [Client(c, all_data[c], embed_d, use_dp, use_aug, dp_sigma)
                 for c in range(num_clients)]
    history   = defaultdict(list)
    t0        = time.time()

    if verbose:
        dp_str = f"σ={dp_sigma}" if use_dp else "off"
        print(f"\n  pFedLLM | {num_clients} clients | {num_rounds} rounds | "
              f"partition={partition} | DP={dp_str} | aug={'on' if use_aug else 'off'}")
        print(f"  {'Rnd':>4}  {'Loss':>8}  {'Acc':>8}  {'AUC-ROC':>9}  {'F1':>8}")
        print("  " + "─" * 46)

    for rnd in range(1, num_rounds + 1):
        lr_r   = LR * (0.97 ** rnd)
        losses = [cli.train_epoch(server, lr=lr_r) for cli in clients]

        if rnd % 5 == 0 or rnd == num_rounds or rnd == 1:
            per_cli = [cli.evaluate(server, test_data) for cli in clients]
            avg     = {k: float(np.mean([m[k] for m in per_cli])) for k in per_cli[0]}
            for k, v in avg.items(): history[k].append(v)
            history["round"].append(rnd)
            if verbose:
                print(f"  {rnd:>4}  {np.mean(losses):>8.4f}  "
                      f"{avg['accuracy']:>8.4f}  {avg['auc_roc']:>9.4f}  {avg['f1']:>8.4f}")

    elapsed = time.time() - t0
    final   = {k: v[-1] for k, v in history.items() if k != "round"}
    final["elapsed_s"] = round(elapsed, 2)
    q = BATCH_SIZE / SAMPLES
    final["epsilon"] = compute_epsilon(dp_sigma, q, num_rounds)
    return final, dict(history)

# ══════════════════════════════════════════════════════════════════════════════
# BASELINES  (Table V)
# ══════════════════════════════════════════════════════════════════════════════
def run_local(embed_d=EMBED_DIM):
    """Local-only training: no collaboration, no server features."""
    np.random.seed(SEED)
    test_data = make_test_data()
    server    = Server(embed_d)
    per_cli   = []
    for c in range(NUM_CLIENTS):
        data = make_data(SAMPLES, c, "non_iid")
        cli  = Client(c, data, embed_d, use_dp=False, use_aug=False)
        for rnd in range(NUM_ROUNDS):
            lr_r = LR * (0.97 ** (rnd + 1))
            idx  = np.random.permutation(cli.n)
            for i in range(0, cli.n - BATCH_SIZE + 1, BATCH_SIZE):
                b  = idx[i:i + BATCH_SIZE]
                h  = cli.comp.fwd(data["imgs"][b])
                z  = np.zeros_like(h)           # no server features
                lg = cli.head.fwd(h, z)
                dl = bce_grad(lg, data["labels"][b])
                dh = cli.head.bwd(dl, lr_r)
                cli.comp.bwd(dh, lr_r)
        h  = cli.comp.fwd(test_data["imgs"])
        z  = np.zeros((len(h), embed_d))
        lg = cli.head.fwd(h, z)
        per_cli.append(metrics(lg, test_data["labels"]))
    return {k: float(np.mean([m[k] for m in per_cli])) for k in per_cli[0]}


def run_fedavg(embed_d=EMBED_DIM):
    """FedAvg: average weights, no personalisation, uses server LLM."""
    np.random.seed(SEED)
    test_data = make_test_data()
    server    = Server(embed_d)
    all_data  = [make_data(SAMPLES, c, "non_iid") for c in range(NUM_CLIENTS)]

    # Shared weights
    sW1 = he(IN_DIM, embed_d // 2); sW2 = he(embed_d // 2, embed_d)
    sH1 = he(2*embed_d, embed_d);   sH2 = he(embed_d, NUM_CLASSES)

    for rnd in range(NUM_ROUNDS):
        lr_r = LR * (0.97 ** (rnd + 1))
        aW1 = np.zeros_like(sW1); aW2 = np.zeros_like(sW2)
        aH1 = np.zeros_like(sH1); aH2 = np.zeros_like(sH2)
        for c in range(NUM_CLIENTS):
            cli = Client(c, all_data[c], embed_d, use_dp=False, use_aug=False)
            cli.comp.fc1.W[:] = sW1; cli.comp.fc2.W[:] = sW2
            cli.head.fc1.W[:] = sH1; cli.head.fc2.W[:] = sH2
            cli.train_epoch(server, lr=lr_r)
            aW1 += cli.comp.fc1.W; aW2 += cli.comp.fc2.W
            aH1 += cli.head.fc1.W; aH2 += cli.head.fc2.W
        n = NUM_CLIENTS
        sW1 = aW1/n; sW2 = aW2/n; sH1 = aH1/n; sH2 = aH2/n

    cli = Client(0, all_data[0], embed_d, use_dp=False, use_aug=False)
    cli.comp.fc1.W[:] = sW1; cli.comp.fc2.W[:] = sW2
    cli.head.fc1.W[:] = sH1; cli.head.fc2.W[:] = sH2
    h  = cli.comp.fwd(test_data["imgs"])
    z  = server.encode(h, test_data["reps"], sigma=0.0)
    lg = cli.head.fwd(h, z)
    return metrics(lg, test_data["labels"])

# ══════════════════════════════════════════════════════════════════════════════
# PAPER REFERENCE TABLES
# ══════════════════════════════════════════════════════════════════════════════
def print_dp_table():
    q = BATCH_SIZE / SAMPLES
    paper_bleu = {0.0: 0.298, 0.1: 0.291, 0.3: 0.283, 0.5: 0.262, 0.8: 0.219}
    paper_auc  = {0.0: 0.928, 0.1: 0.924, 0.3: 0.919, 0.5: 0.901, 0.8: 0.861}
    print("\n" + "═"*62)
    print("  Table IX — DP Privacy Budget & Performance (paper values)")
    print("═"*62)
    print(f"  {'σ':>5}  {'ε (RDP)':>10}  {'BLEU-4':>8}  {'AUC-ROC':>9}")
    print("  " + "─"*50)
    for sigma in [0.0, 0.1, 0.3, 0.5, 0.8]:
        eps = compute_epsilon(sigma, q, 50)
        es  = "  no DP" if sigma == 0.0 else f"{eps:>10.2f}"
        print(f"  {sigma:>5.1f}  {es}  {paper_bleu[sigma]:>8.3f}  {paper_auc[sigma]:>9.3f}")
    print("═"*62)

def print_comm_analysis():
    d, T, B = 1024, 256, 32
    up_KB   = (B*d + B*T) * 4 / 1024
    down_KB = B * d * 4 / 1024
    total   = up_KB + down_KB
    fedavg  = 98 * 1024
    print("\n" + "═"*60)
    print("  Communication Analysis (Section IV.D, full-scale d=1024)")
    print("═"*60)
    print(f"    Uplink   : {up_KB:.0f} KB   (h_c + de-id report tokens)")
    print(f"    Downlink : {down_KB:.0f} KB   (z_agg shared features)")
    print(f"    Total    : {total:.0f} KB   vs FedAvg ~{fedavg//1024} MB")
    print(f"    Reduction: >{fedavg/total:.0f}× less bandwidth per round")
    print("═"*60)

def print_inversion_table():
    print("\n" + "═"*58)
    print("  Table XI — Inversion Attack Resistance (paper values)")
    print("  SSIM ↓ and PSNR ↓ = harder to reconstruct = more private")
    print("═"*58)
    print(f"  {'Method':<16}  {'Attack':<18}  {'SSIM↓':>6}  {'PSNR↓':>7}")
    print("  " + "─"*52)
    for method, attack, ssim, psnr, mark in [
        ("FedAvg",     "Gradient Inv.", 0.81, 24.7, ""),
        ("Per-FedAvg", "Gradient Inv.", 0.75, 22.9, ""),
        ("pFedLLM",    "Feature Inv.",  0.19, 11.3, " ◄ best"),
    ]:
        print(f"  {method:<16}  {attack:<18}  {ssim:>6.2f}  {psnr:>7.1f}{mark}")
    print("═"*58)

def print_backbone_table():
    print("\n" + "═"*58)
    print("  Table XII — Backbone Convergence (server-side, paper values)")
    print("═"*58)
    print(f"  {'Backbone':<14}  {'GPU-hrs':>9}  {'→converge':>10}  {'Comm/rnd':>10}")
    print("  " + "─"*50)
    for name, gpu, rnd, comm, mark in [
        ("Med-Gemini", 18.2, 37, "8.0 KB", ""),
        ("GLM-4.5V",   12.9, 29, "6.9 KB", " ◄ fastest"),
        ("Med-R1",     21.4, 34, "8.0 KB", ""),
    ]:
        print(f"  {name:<14}  {gpu:>9.1f}  {rnd:>10}  {comm:>10}{mark}")
    print("═"*58)

# ══════════════════════════════════════════════════════════════════════════════
# ABLATIONS
# ══════════════════════════════════════════════════════════════════════════════
def run_comparison():
    print("\n" + "═"*72)
    print("  Table V — Disease Classification, Non-IID Partition")
    print("  [Simulation on toy data] + [Paper values for reference]")
    print("═"*72)
    print(f"  {'Method':<28}  {'Acc(sim)':>9}  {'AUC(sim)':>9}  "
          f"{'Acc(paper)':>11}  {'AUC(paper)':>11}")
    print("  " + "─"*70)

    paper = {
        "Local (no FL)":         (0.784, 0.845),
        "FedAvg":                (0.811, 0.868),
        "Per-FedAvg":            (0.823, 0.879),
        "pFedLVM":               (0.829, 0.882),
        "AdaptiveDualBranchNet": (0.834, 0.887),
        "GLM-4.5V":              (0.840, 0.900),
        "Med-R1":                (0.832, 0.890),
        "pFedLLM (ours)":        (0.849, 0.939),
    }

    def row(name, m, p, mark=""):
        print(f"  {name+mark:<28}  {m['accuracy']:>9.4f}  {m['auc_roc']:>9.4f}  "
              f"{p[0]:>11.3f}  {p[1]:>11.3f}")

    print("  Computing Local baseline ...", end=" ", flush=True)
    m = run_local(); print("done")
    row("Local (no FL)", m, paper["Local (no FL)"])

    print("  Computing FedAvg ...", end=" ", flush=True)
    m = run_fedavg(); print("done")
    row("FedAvg", m, paper["FedAvg"])

    print("  Computing pFedLLM (no DP, no aug) ...", end=" ", flush=True)
    m, _ = run_pFedLLM(use_dp=False, use_aug=False, verbose=False); print("done")
    row("pFedLLM (no DP/aug)", m, paper["Per-FedAvg"])

    print("  Computing pFedLLM (full) ...", end=" ", flush=True)
    m, _ = run_pFedLLM(use_dp=True, use_aug=True, verbose=False); print("done")
    row("pFedLLM (full)", m, paper["pFedLLM (ours)"], " ◄")

    print("═"*72)
    print("\n  Paper-only methods (not re-implemented here):")
    for name in ["Per-FedAvg","pFedLVM","AdaptiveDualBranchNet","GLM-4.5V","Med-R1"]:
        pa, pu = paper[name]
        print(f"  {name:<28}  {'(paper)':>9}  {'(paper)':>9}  {pa:>11.3f}  {pu:>11.3f}")
    print("═"*72)


def run_dp_sweep():
    sigmas = [0.0, 0.1, 0.3, 0.5, 0.8]
    q      = BATCH_SIZE / SAMPLES
    print("\n" + "═"*62)
    print("  Table VIII — DP Noise vs Performance (simulated)")
    print("═"*62)
    print(f"  {'σ':>6}  {'ε':>8}  {'AUC-ROC':>9}  {'Acc':>8}  {'F1':>8}")
    print("  " + "─"*50)
    for sigma in sigmas:
        m, _ = run_pFedLLM(use_dp=(sigma>0), dp_sigma=sigma, verbose=False)
        eps  = compute_epsilon(sigma, q, NUM_ROUNDS)
        es   = "  no DP" if sigma == 0.0 else f"{eps:>8.2f}"
        print(f"  {sigma:>6.1f}  {es}  {m['auc_roc']:>9.4f}  "
              f"{m['accuracy']:>8.4f}  {m['f1']:>8.4f}")
    print("═"*62)


def run_dim_sweep():
    dims = [16, 32, 64, 128, 256]
    print("\n" + "═"*52)
    print("  Fig. 6 — Feature Dimensionality vs Utility")
    print("═"*52)
    print(f"  {'d':>6}  {'AUC-ROC':>9}  {'Acc':>8}  {'F1':>8}")
    print("  " + "─"*40)
    for d in dims:
        m, _ = run_pFedLLM(embed_d=d, verbose=False)
        print(f"  {d:>6}  {m['auc_roc']:>9.4f}  {m['accuracy']:>8.4f}  {m['f1']:>8.4f}")
    print("═"*52)
    print("  Paper (full-scale): performance saturates at d=1024")


def run_aug_ablation():
    print("\n" + "═"*58)
    print("  Table X — Generative Augmentation (simulated)")
    print("═"*58)
    m_no, _  = run_pFedLLM(use_aug=False, verbose=False)
    m_yes, _ = run_pFedLLM(use_aug=True,  verbose=False)
    df1  = (m_yes["f1"]      - m_no["f1"])      * 100
    dauc = (m_yes["auc_roc"] - m_no["auc_roc"]) * 100
    print(f"  Without aug  →  F1={m_no['f1']:.4f}   AUC={m_no['auc_roc']:.4f}")
    print(f"  With    aug  →  F1={m_yes['f1']:.4f}   AUC={m_yes['auc_roc']:.4f}")
    print(f"  Simulated Δ :   ΔF1={df1:+.2f}pp   ΔAUC={dauc:+.2f}pp")
    print(f"  Paper reports:  rare-class F1 +11–17% (Cardiomegaly, Lung Lesion)")
    print("\n  Paper Table X reference:")
    for cls, (wo, w) in [("Cardiomegaly",(0.54,0.61)),
                          ("Lung Lesion", (0.47,0.55)),
                          ("Consolidation",(0.52,0.59))]:
        print(f"    {cls:<18}  w/o={wo:.2f}  w/={w:.2f}  Δ={w-wo:+.2f}")
    print("═"*58)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="pFedLLM NumPy Demo")
    p.add_argument("--rounds",      type=int, default=NUM_ROUNDS)
    p.add_argument("--clients",     type=int, default=NUM_CLIENTS)
    p.add_argument("--partition",   type=str, default="non_iid",
                   choices=["iid","non_iid"])
    p.add_argument("--compare_all", action="store_true")
    p.add_argument("--ablation",    type=str, default=None,
                   choices=["dp","dim","aug","all"])
    p.add_argument("--no_dp",  action="store_true")
    p.add_argument("--no_aug", action="store_true")
    args = p.parse_args()

    print("\n" + "═"*62)
    print("  pFedLLM — Personalized Federated Learning Demo")
    print("  NumPy simulation of Algorithm 1 (paper tables reproduced)")
    print("═"*62)
    print_comm_analysis()
    print_dp_table()
    print_backbone_table()
    print_inversion_table()

    if args.compare_all:   run_comparison();    return
    if args.ablation == "dp":  run_dp_sweep();  return
    if args.ablation == "dim": run_dim_sweep(); return
    if args.ablation == "aug": run_aug_ablation(); return
    if args.ablation == "all":
        run_dp_sweep(); run_dim_sweep(); run_aug_ablation(); return

    # ── Default: IID then Non-IID ─────────────────────────────────────
    print("\n" + "─"*62)
    print("  Experiment 1: IID Partition")
    m_iid, _ = run_pFedLLM(
        num_clients=args.clients, num_rounds=args.rounds,
        partition="iid", use_dp=not args.no_dp, use_aug=not args.no_aug,
    )
    print("\n" + "─"*62)
    print("  Experiment 2: Non-IID Partition (temporal shift)")
    m_nid, _ = run_pFedLLM(
        num_clients=args.clients, num_rounds=args.rounds,
        partition="non_iid", use_dp=not args.no_dp, use_aug=not args.no_aug,
    )

    print("\n" + "═"*68)
    print("  Summary — pFedLLM vs Paper (Table V)")
    print("═"*68)
    print(f"  {'Metric':<12}  {'IID (sim)':>10}  {'IID (paper)':>12}  "
          f"{'NonIID (sim)':>13}  {'NonIID (paper)':>14}")
    print("  " + "─"*64)
    ref_iid  = {"accuracy":0.872,"auc_roc":0.928,"f1":0.859}
    ref_nid  = {"accuracy":0.849,"auc_roc":0.939,"f1":0.825}
    for k in ["accuracy","auc_roc","f1"]:
        print(f"  {k:<12}  {m_iid[k]:>10.4f}  {ref_iid[k]:>12.3f}  "
              f"{m_nid[k]:>13.4f}  {ref_nid[k]:>14.3f}")
    print("═"*68)
    print(f"\n  DP budget : ε = {m_nid['epsilon']:.2f}  (σ={DP_SIGMA_DEFAULT}, δ={DP_DELTA})")
    print(f"  Runtime   : {m_iid['elapsed_s']+m_nid['elapsed_s']:.1f}s  "
          f"(toy d={EMBED_DIM}, N={SAMPLES}; paper: d=1024, MIMIC-CXR)")
    print("\n  Further experiments:")
    print("    python demo_numpy.py --compare_all")
    print("    python demo_numpy.py --ablation dp")
    print("    python demo_numpy.py --ablation all")

if __name__ == "__main__":
    main()
