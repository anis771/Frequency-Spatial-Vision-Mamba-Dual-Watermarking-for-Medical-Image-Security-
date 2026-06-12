"""
test_smoke.py -- Quick end-to-end sanity check.
Run: venv/Scripts/python.exe test_smoke.py
Uses a TINY model config for CPU-only testing.
"""
import sys, torch
sys.path.insert(0, '.')

print("=" * 50)
print("Full Model Forward Pass (TINY CPU config)")
print("=" * 50)
from src.models.full_model import WatermarkNet

# Tiny config for CPU smoke test
IMG_SIZE = 64     # small image
FEAT_DIM = 32     # small feature dim
WM_BITS  = 32     # small bit vector
BATCH    = 1

model = WatermarkNet(
    img_size=IMG_SIZE, patch_size=4, in_chans=1,
    feat_dim=FEAT_DIM, wm_bits=WM_BITS, depth=1,
    use_pretrained_seg=False
)
total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {total_p:,}")

cover   = torch.rand(BATCH, 1, IMG_SIZE, IMG_SIZE)
wm1_b   = (torch.rand(BATCH, WM_BITS) > 0.5).float()
wm2_b   = (torch.rand(BATCH, WM_BITS) > 0.5).float()
gt_mask = torch.zeros(BATCH, IMG_SIZE, IMG_SIZE, dtype=torch.long)

out = model(cover, wm1_b, wm2_b, gt_mask=gt_mask, apply_attack=True)
print(f"watermarked      : {out['watermarked'].shape}")
print(f"alpha_enc        : {out['alpha_enc'].shape}")
print(f"wm1_clean_logits : {out['wm1_clean_logits'].shape}")
print(f"wm1_adv_logits   : {out['wm1_adv_logits'].shape}")
print(f"attack_used      : {out.get('attack_name', '?')}")

psnr_val = 10 * torch.log10(
    torch.tensor(1.0) / ((out['watermarked'] - cover) ** 2).mean()
)
print(f"Embed PSNR (untrained): {psnr_val.item():.2f} dB")

print()
print("=" * 50)
print("Attack Pool Test")
print("=" * 50)
from src.utils.attacks import apply_all_attacks
attacked = apply_all_attacks(cover)
for name, v in attacked.items():
    mn, mx = v.min().item(), v.max().item()
    print(f"  {name:20s}: {v.shape}  range=[{mn:.2f},{mx:.2f}]")

print()
print("=" * 50)
print("Loss Function Test")
print("=" * 50)
from src.utils.losses import TotalWatermarkLoss
criterion = TotalWatermarkLoss(lambda_adv=0.5, lambda_imp=0.1)
loss_dict = criterion(
    cover=cover,
    watermarked=out['watermarked'],
    wm1_true=wm1_b, wm2_true=wm2_b,
    wm1_clean_pred=out['wm1_clean_logits'],
    wm2_clean_pred=out['wm2_clean_logits'],
    wm1_adv_pred=out['wm1_adv_logits'],
    wm2_adv_pred=out['wm2_adv_logits'],
)
for k, v in loss_dict.items():
    print(f"  {k:12s}: {v if isinstance(v, float) else v.item():.4f}")

print()
print("=" * 50)
print("Metrics Test")
print("=" * 50)
from src.utils.metrics import compute_all_metrics
metrics = compute_all_metrics(
    cover, out['watermarked'],
    wm1_b.unsqueeze(-1).unsqueeze(-1),
    torch.sigmoid(out['wm1_clean_logits']).unsqueeze(-1).unsqueeze(-1),
    wm2_b.unsqueeze(-1).unsqueeze(-1),
    torch.sigmoid(out['wm2_clean_logits']).unsqueeze(-1).unsqueeze(-1),
)
for k, v in metrics.items():
    print(f"  {k:10s}: {v:.4f}")

print()
print("ALL TESTS PASSED [OK]")
