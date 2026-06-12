# Frequency-Spatial Vision Mamba Dual Watermarking for Medical Image Security

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-orange)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/dataset-CHAOS_T1%26T2-green)](https://www.kaggle.com/datasets/anhoangvo/chaos-t1-and-t2)

> **Paper**: Frequency–Spatial Vision Mamba Dual Watermarking for Medical Image Security

---

## Overview

This repository implements a complete, trainable PyTorch system for **dual watermark embedding** into abdominal MRI images. Two encrypted secret images are hidden inside a single CHAOS MRI cover image with:

- **High imperceptibility** — PSNR > 42 dB under attacks, > 45 dB in organ ROIs
- **High robustness** — NC > 0.90 under 7 attack types
- **Security** — AES-256-GCM encryption + HKDF per-image key derivation

---

## Architecture

```
cover MRI (256×256)
├── VisionMambaEncoder  ──→  F_s  (64×64×256)  ← Spatial branch (O(n) SSM)
└── DWTBranch (2-lvl Haar) → F_f  (64×64×256)  ← Frequency branch
                │
         LearnableGating
         α = Conv1×1 3-layer sigmoid
         F_fused = α⊙F_f + (1-α)⊙F_s
                │
    ROI mask → β(x,y) = 0.15·(1 - 0.95·W_roi)
                │
    wm1_bits + wm2_bits (AES-256-GCM encrypted)
    projected → embed with β-modulated residual
                │
         I_w = cover + residual·β
                │
    ┌─── clean decode ────┐
    └─── attacked decode ─┘  → L_clean + 0.5·L_adv + 0.1·L_imp
```

---

## Checklist (from paper)

| # | Component | Status |
|---|-----------|--------|
| 1 | Frequency-Spatial Vision Mamba Architecture | ✅ `src/models/vision_mamba.py` + `dwt_branch.py` |
| 2 | ROI-Preserving Embedding | ✅ `src/utils/roi_weight.py` + `src/models/unet_roi.py` |
| 3 | Learnable Gating Mechanism | ✅ `src/models/gating.py` |
| 4 | Adversarial Robustness Training | ✅ `src/utils/attacks.py` + `src/training/train.py` |
| 5 | AES-256-GCM + HKDF Encryption | ✅ `src/utils/crypto.py` |

---

## Dataset Setup

1. Download **CHAOS T1 & T2** from Kaggle:
   https://www.kaggle.com/datasets/anhoangvo/chaos-t1-and-t2

2. Place the dataset under:
   ```
   data/CHAOS_T1T2/
   ├── images/       ← PNG MRI slices
   └── masks/        ← PNG segmentation masks (0=bg,1=liver,2=LK,3=RK,4=spleen)
   ```

3. Mask pixel values: `{0, 63, 126, 189, 252}` → auto-remapped to `{0,1,2,3,4}`

---

## Environment Setup

```powershell
# Create & activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

Or use the one-click script:
```
.\setup_env.bat
```

---

## Project Structure

```
Frequency Spatial Vision Mamba/
├── src/
│   ├── models/
│   │   ├── vision_mamba.py   # 2D-SSM, 4-direction scan
│   │   ├── dwt_branch.py     # 2-level Haar DWT
│   │   ├── gating.py         # Learnable α gating
│   │   ├── encoder.py        # Dual watermark encoder
│   │   ├── decoder.py        # Dual watermark extractor
│   │   ├── unet_roi.py       # ResNet-34 U-Net segmentation
│   │   └── full_model.py     # WatermarkNet (assembled)
│   ├── utils/
│   │   ├── attacks.py        # 7-attack pool
│   │   ├── crypto.py         # AES-256-GCM + HKDF
│   │   ├── losses.py         # L_clean + L_adv + L_imp
│   │   ├── metrics.py        # PSNR, SSIM, NC
│   │   └── roi_weight.py     # Distance-transform β map
│   ├── data/
│   │   └── chaos_dataset.py  # CHAOS T1/T2 loader
│   └── training/
│       ├── config.py         # All hyperparameters
│       ├── train.py          # Min-max training loop
│       └── evaluate.py       # Per-attack evaluation
├── data/CHAOS_T1T2/          # ← Place dataset here
├── checkpoints/              # Saved model weights
├── logs/                     # TensorBoard logs
├── results/                  # Evaluation CSVs
├── notebooks/demo.ipynb      # Visual demo
├── requirements.txt
├── setup_env.bat
└── README.md
```

---

## Training

```powershell
.\venv\Scripts\activate

# Start training (100 epochs, adversarial from epoch 6)
python -m src.training.train
```

**Monitor with TensorBoard:**
```powershell
tensorboard --logdir logs
```

**Key hyperparameters** (in `src/training/config.py`):

| Param | Value | Description |
|-------|-------|-------------|
| `IMAGE_SIZE` | 256 | MRI resize |
| `FEAT_DIM` | 256 | Feature dimension |
| `BETA_BASE` | 0.15 | Base embedding strength |
| `GAMMA` | 0.95 | ROI suppression |
| `LAMBDA_ADV` | 0.5 | Adversarial loss weight |
| `LAMBDA_IMP` | 0.1 | Imperceptibility loss weight |
| `ADV_START_EPOCH` | 5 | Warmup before adversarial |

---

## Evaluation

```powershell
# Evaluate best checkpoint on test split
python -m src.training.evaluate --checkpoint checkpoints/best.pth --split test
```

**Paper targets:**

| Metric | Target | Condition |
|--------|--------|-----------|
| PSNR | > 42 dB | Under any attack |
| PSNR (ROI) | > 45 dB | Inside organ regions |
| SSIM | > 0.96 | Under any attack |
| NC | > 0.90 | Both watermarks |

---

## Attack Pool

| Attack | Parameters |
|--------|-----------|
| JPEG compression | QF = 10–90 |
| Gaussian noise | σ = 0.001–0.05 |
| Salt & pepper | density = 0.01–0.1 |
| Median filter | k = 3 or 5 |
| Rotation | ±1–5° |
| Scaling | 0.8–1.2× |
| Gaussian blur | σ = 0.5–2.0 |

---

## Security: AES-256-GCM + HKDF

```
master_key (32 bytes)
      +
cover_image_hash (SHA-256)
      ↓
   HKDF → per-image key (32 bytes)
      ↓
AES-256-GCM encrypt(watermark_bytes)
      ↓
   nonce (12B) ‖ ciphertext ‖ tag (16B) → watermark payload
```

- Tamper detection: GCM tag verification on extraction
- Invalid tag → `cryptography.exceptions.InvalidTag` raised

---

## License

MIT
