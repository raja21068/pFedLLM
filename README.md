# pFedLLM: Advancing Medical Imaging with LLM-Driven Personalized Federated Learning

> Implementation of **"pFedLLM: Advancing Medical Imaging with LLM-Driven Personalized Federated Learning"**

---

## Overview

pFedLLM is a personalized, LLM-driven federated framework for medical imaging and report generation that addresses three core challenges in distributed clinical AI:

| Challenge | pFedLLM Solution |
|---|---|
| Heterogeneous feature representations | Feature-map communication (not parameter sharing) |
| Robust privacy preservation | Feature-level DP + Gaussian mechanism (Table IX) |
| Underuse of multimodal LLMs in FL | Server-side Med-Gemini/GLM-4.5V, never distributed |

### Architecture (Algorithm 1)

```
CLIENTS (Hospitals)                       CENTRAL CLOUD SERVER
───────────────────────────────────────   ────────────────────────────────────
Raw Image I  ──► C_φc ──► h_c ────────►  Server-side Multimodal LLM  F_θ
De-id Report R            ─────────────►  ─ Med-Gemini / GLM-4.5V (3D-RoPE, MoE)
                                           ─ Cross-modal fusion (image × text)
                          ◄──────────────  z_agg = Σ_c (|D_c|/|D|)·z_c + ε_DP
[h_c ‖ z_agg] ──► H_ψc ──► ŷ             (weighted avg + Gaussian DP noise)
Update (φ_c, ψ_c) locally only
Raw data, labels, LLM weights never leave their respective machines
```

**Communication cost:** ~9–10 KB/sample vs >100 MB for FedAvg → **>1000× reduction**

---

## Results

### Disease Classification — MIMIC-CXR (Table V)

| Method | IID Acc | IID AUC | NonIID Acc | NonIID AUC | NonIID F1 |
|---|---|---|---|---|---|
| Local (no FL) | 0.819 | 0.872 | 0.784 | 0.845 | 0.763 |
| FedAvg | 0.838 | 0.891 | 0.811 | 0.868 | 0.786 |
| Per-FedAvg | 0.846 | 0.903 | 0.823 | 0.879 | 0.797 |
| AdaptiveDualBranchNet | 0.857 | 0.912 | 0.834 | 0.887 | 0.808 |
| GLM-4.5V | 0.859 | 0.915 | 0.840 | 0.900 | 0.820 |
| **pFedLLM (ours)** | **0.872** | **0.928** | **0.849** | **0.939** | **0.825** |

### Report Generation — MIMIC-CXR (Table IV)

| Method | IID BLEU-4 | IID ROUGE-L | NonIID BLEU-4 | NonIID ROUGE-L |
|---|---|---|---|---|
| FedAvg | 0.225 | 0.272 | 0.207 | 0.256 |
| GLM-4.5V | 0.275 | 0.321 | 0.243 | 0.297 |
| **pFedLLM (ours)** | **0.298** | **0.346** | **0.264** | **0.319** |

---

## Project Structure

```
pFedLLM/
├── configs/
│   └── default.py                 # All hyperparameters (Table II)
├── models/
│   ├── feature_compressor.py      # C_φc — CNN + optional tiny ViT
│   ├── personalized_head.py       # H_ψc — 4 task heads
│   ├── server_llm.py              # F_θ  — cross-modal LLM + GLM-4.5V stub
│   └── generative_augmentor.py    # cVAE rare-class synthesis (Table X)
├── federated/
│   ├── client.py                  # FederatedClient — compress→send→update
│   ├── server.py                  # FederatedServer — encode→aggregate→broadcast
│   └── aggregation.py             # weighted_avg / attention / similarity
├── data/
│   └── synthetic_dataset.py       # Synthetic MIMIC-CXR + IID/NonIID splits
├── utils/
│   ├── differential_privacy.py    # Gaussian mechanism + DP-SGD + RDP accountant
│   ├── metrics.py                 # BLEU-4, ROUGE-L, AUC-ROC, F1, Dice, IoU
│   ├── data_utils.py              # MIMIC-CXR loader + Dirichlet partitioning
│   ├── visualization.py           # All paper figures (Figs. 3–8)
│   └── privacy_analysis.py        # PrivacyAccountant, inversion attack, MI attack
├── experiments/
│   └── run_experiment.py          # Ablations (Section V.I)
├── demo_numpy.py                  # ▶ Run first — zero dependencies beyond NumPy
├── train.py                       # Main training CLI
├── evaluate.py                    # All paper tables IV–XII
├── requirements.txt
├── setup.py
└── LICENSE
```

---

## Quick Start

### Option A — Zero install (NumPy only)

```bash
git clone https://github.com/your-username/pFedLLM.git
cd pFedLLM

python demo_numpy.py                   # IID + Non-IID runs
python demo_numpy.py --compare_all     # Table V baseline comparison
python demo_numpy.py --ablation dp     # Table VIII DP noise sweep
python demo_numpy.py --ablation dim    # Fig. 6 dimensionality sweep
python demo_numpy.py --ablation aug    # Table X rare-class augmentation
python demo_numpy.py --ablation all    # all ablations
```

### Option B — Full PyTorch training

```bash
pip install -r requirements.txt
# or: pip install -e .

# Synthetic data (no download required):
python train.py --config fast --use_synthetic

# Paper settings (10 clients, 50 rounds, Non-IID):
python train.py \
    --use_synthetic \
    --num_clients 10 \
    --rounds 50 \
    --partition non_iid_temporal \
    --dp_sigma 0.6 \
    --task disease_classification

# With real MIMIC-CXR data:
python train.py \
    --dataset mimic_cxr \
    --data_dir /path/to/mimic-cxr-jpg \
    --num_clients 10 \
    --rounds 50 \
    --partition non_iid_clinical
```

### Option C — View all paper tables

```bash
python evaluate.py --demo          # all tables + key findings summary
python evaluate.py --table v       # Table V (classification)
python evaluate.py --table ix      # Table IX (DP privacy budget)
```

---

## Hyperparameters (Table II)

| Parameter | Value | Notes |
|---|---|---|
| Rounds R | 50 | Communication rounds |
| Local epochs E | 1 | Epochs per client per round |
| Batch size | 32 | |
| Embedding dim d | 1024 | h_c, z_c dimensionality |
| Optimizer | Adam, lr=1e-4 | |
| DP σ | 0.6 | Gaussian noise multiplier |
| DP δ | 1e-5 | Privacy failure probability |
| Backbone F_θ | Med-Gemini / GLM-4.5V | Server LLM |
| Compressor C_φc | 3 conv blocks + GAP + Linear | |
| Augmentation | Generative 1:1 | cVAE for rare classes |

---

## Module Reference

### Configs

```python
from configs.default import FederatedConfig, get_config

cfg = get_config("default")   # full paper settings (50 rounds, 10 clients)
cfg = get_config("fast")      # quick test (5 rounds, 3 clients, synthetic)
cfg = get_config("non_iid")   # non-IID temporal partition
cfg = get_config("no_dp")     # disable differential privacy
```

### Models

```python
from models.feature_compressor  import FeatureCompressor
from models.personalized_head   import PersonalizedHead
from models.server_llm          import ServerLLM
from models.generative_augmentor import GenerativeAugmentor

# Compressor: image → h_c
comp = FeatureCompressor(in_channels=1, embed_dim=1024)
h_c  = comp(image_batch)              # (B,1,224,224) → (B,1024)

# Personalized heads (one per client, per task)
head = PersonalizedHead("disease_classification", embed_dim=1024, num_classes=14)
logits = head(h_c, z_agg)             # (B,14) — BCEWithLogitsLoss

head = PersonalizedHead("report_generation", embed_dim=1024, vocab_size=30522)
logits = head(h_c, z_agg, tgt_ids=tokens)   # (B,T,vocab)
report = head.head.generate(h_c, z_agg)     # greedy decode

head = PersonalizedHead("vqa",             embed_dim=1024, num_answers=500)
head = PersonalizedHead("visual_grounding",embed_dim=1024)
boxes = head(h_c, z_agg)             # (B,4) normalised [x,y,w,h]

# Server LLM (fixed; never backpropagated or distributed)
llm = ServerLLM(embed_dim=1024, freeze=True)
z_c = llm(h_c, report_ids, attention_mask)   # (B,1024)

# Generative augmentor (server pretrained; rare-class synthesis)
aug = GenerativeAugmentor(feat_dim=1024, num_classes=14)
aug.pretrain(h_bank, y_bank, num_epochs=20)
h_aug, y_aug = aug.augment(h_real, y_real, rare_class_ids=[1,2,6])
```

### Federated Training

```python
from federated.client      import FederatedClient
from federated.server      import FederatedServer
from federated.aggregation import aggregate, CommunicationTracker

server    = FederatedServer(cfg, device)
server_fn = server.build_server_fn(dataset_sizes)

client  = FederatedClient(client_id=0, cfg=cfg,
                           train_loader=loader, device=device)
metrics = client.local_train(server_fn, num_epochs=1)

# Aggregation strategies
z_agg = aggregate("weighted_avg",  client_features, dataset_sizes)
z_agg = aggregate("attention",     client_features, dataset_sizes,
                   attention_module=attn_module)
z_agg = aggregate("similarity",    client_features, dataset_sizes,
                   reference_cid=0)

# Communication accounting
tracker = CommunicationTracker(embed_dim=1024)
tracker.record_round(num_clients=10, batch_size=32)
# → CommunicationTracker(rounds=1, total=0.29 MB, FedAvg equiv=196 MB, reduction=675×)
```

### Differential Privacy

```python
from utils.differential_privacy import (
    GaussianMechanism, DPOptimizer, compute_epsilon
)
from utils.privacy_analysis import (
    PrivacyAccountant, FeatureInversionAttack, print_privacy_report
)

# Feature-level DP: add noise to z_agg before broadcast
gm      = GaussianMechanism(sigma=0.6, sensitivity=1.0)
z_noisy = gm(z_agg)

# Client-side DP-SGD
dp_opt = DPOptimizer(base_optimizer, model,
                      max_grad_norm=1.0, noise_multiplier=0.6)
dp_opt.step()

# Track cumulative ε during training
acct = PrivacyAccountant(noise_multiplier=0.6, sampling_rate=32/10000)
acct.step(num_steps=500)      # after 500 gradient steps
print(acct.epsilon)           # current ε

# Compute ε for Table IX
eps, alpha = compute_epsilon(sigma=0.5, sampling_rate=32/10000,
                              num_rounds=50, delta=1e-5)
# → (1.5, best_alpha)

# Simulate feature inversion attack (Table XI)
attacker = FeatureInversionAttack(embed_dim=1024, image_dim=224*224)
results  = attacker.attack(z_features, true_images, num_epochs=50)
# → {'ssim': 0.19, 'psnr': 11.3, 'mse': ...}

print_privacy_report(cfg)
```

### Metrics

```python
from utils.metrics import (
    corpus_bleu4, corpus_rouge_l,    # Table IV
    classification_metrics,           # Table V
    vqa_accuracy,                     # Table VI
    grounding_metrics,                # Table VII
    MetricTracker,
)

m = classification_metrics(logits_np, labels_np)
# → {'accuracy': 0.872, 'auc_roc': 0.928, 'f1': 0.859}

bleu = corpus_bleu4(hypotheses, references)      # float
rl   = corpus_rouge_l(hypotheses, references)    # float
gi   = grounding_metrics(pred_boxes, gt_boxes)   # {'dice':..., 'iou':...}
```

### Visualization

```python
from utils.visualization import (
    plot_all_figures,            # generate all Figs. 3-8
    plot_dp_noise_utility,       # Fig. 4
    plot_comm_accuracy_tradeoff, # Fig. 5
    plot_dimensionality_utility, # Fig. 6
    plot_per_client_gains,       # Fig. 7
    plot_client_drift_pca,       # Fig. 8
    plot_training_history,       # training curves
)

plot_all_figures(save_dir="./figures", history=training_history)
```

---

## Ablations (Section V.I)

```bash
python experiments/run_experiment.py --ablation all        # full suite
python experiments/run_experiment.py --ablation no_dp      # DP removed → AUC +0.8%
python experiments/run_experiment.py --ablation no_aug     # aug removed → F1 -11%
python experiments/run_experiment.py --ablation dim_256    # d=256 → Acc -2.4pp
python experiments/run_experiment.py --dp_sweep            # Table VIII
python experiments/run_experiment.py --dim_sweep           # Fig. 6
python experiments/run_experiment.py --compare_baselines   # print Table V
```

Key ablation findings (Table V.I):
- **Vision-only backbone** (no LLM text): BLEU-4 ↓ 14%
- **Shared head** (no per-client): accuracy ↓ 9%
- **d = 256 → 1024**: accuracy +2.4pp; saturates at d=1024
- **DP removed** (σ=0): AUC slightly +, but privacy lost
- **No generative augmentation**: rare-class F1 ↓ 11–17%
- **GLM-4.5V vs Med-Gemini**: NonIID BLEU-4 +2.1%, converges in 29 vs 37 rounds
- **GLM-4.5V + MRG-LLM prompts**: BLEU-4 +4.3% over Med-Gemini

---

## Datasets

| Dataset | Size | Tasks | Link |
|---|---|---|---|
| MIMIC-CXR-JPG | 377K CXR + reports | Report gen, classification | [PhysioNet](https://physionet.org/content/mimic-cxr-jpg/) |
| MedTrinity-25M | 25M images, 10 modalities | Multimodal | [HuggingFace](https://huggingface.co/datasets/UCSC-VLAA/MedTrinity-25M) |
| Med-MAT | 106 medical VQA datasets | VQA | [GitHub](https://github.com/FreedomIntelligence/Med-MAT) |

> **Note:** MIMIC-CXR requires credentialed PhysioNet access. Use `--use_synthetic` for testing.

---

## Citation

```bibtex
@article{pfedllm2025,
  title   = {pFedLLM: Advancing Medical Imaging with LLM-Driven
             Personalized Federated Learning},
  author  = {Anonymous Authors},
  year    = {2025},
}
```

---

## License

[MIT License](LICENSE)
