#!/usr/bin/env python3
"""
DIFFUSION LUNG SEGMENTATION — 3 Models × 3 Folds (GPU-Optimised)
Target: 90-94% Dice accuracy
Checkpoint structure:
  Models/Final_Output/checkpoints/unet_diff/best_DiffusionUNet_fold_1.pt
  Models/Final_Output/checkpoints/unetpp_diff/best_DiffusionUNetPlusPlus_fold_1.pt
  Models/Final_Output/checkpoints/resunet_diff/best_DiffusionResNetUNet_fold_1.pt
"""

import os, sys, random, math, json, gc, warnings
from pathlib import Path

# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
import time

# ============================================================================
# REPRODUCIBILITY & DEVICE
# ============================================================================
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True      # ← auto-tune kernels for fixed input size
    torch.backends.cudnn.deterministic = False  # ← faster but non-deterministic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()            # mixed-precision training
print(f"[OK] Device : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"[OK] AMP    : {USE_AMP}\n")

# ============================================================================
# PATHS & CONFIGURATION
# ============================================================================
PROJECT_ROOT      = Path(__file__).resolve().parent.parent
PROCESSED_IMG_DIR = PROJECT_ROOT / "Data" / "processed_images"
OUTPUT_DIR        = PROJECT_ROOT / "Models" / "Final_Output"

IMAGE_SIZE              = 256
BATCH_SIZE              = 32   # larger batch → better GPU utilisation
NUM_EPOCHS              = 150
LEARNING_RATE           = 1e-3
NUM_FOLDS               = 3
MAX_SLICES_PER_CASE     = 50
EARLY_STOPPING_PATIENCE = 25
NUM_WORKERS             = 0    # Windows: must be 0 (multiprocessing spawn issues)

# Diffusion hyper-parameters
DIFFUSION_TIMESTEPS = 1000
BETA_START          = 1e-4
BETA_END            = 0.02
TIME_EMB_DIM        = 128

for model_key in ["unet_diff", "unetpp_diff", "resunet_diff"]:
    (OUTPUT_DIR / "checkpoints" / model_key).mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "tables").mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  DIFFUSION LUNG SEGMENTATION — 3 MODELS × 3 FOLDS (GPU-OPTIMISED)")
print("=" * 70)
print(f"  Device         : {DEVICE}")
print(f"  Batch Size     : {BATCH_SIZE}  (AMP: {USE_AMP})")
print(f"  Epochs         : {NUM_EPOCHS}  (early stop: {EARLY_STOPPING_PATIENCE})")
print(f"  Image Size     : {IMAGE_SIZE}×{IMAGE_SIZE}")
print(f"  Learning Rate  : {LEARNING_RATE}")
print(f"  Diffusion T    : {DIFFUSION_TIMESTEPS}")
print(f"  Output Dir     : {OUTPUT_DIR}")
print("=" * 70 + "\n")

# ============================================================================
# DIFFUSION SCHEDULE
# ============================================================================
class DiffusionSchedule(nn.Module):
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.T = T
        betas            = torch.linspace(beta_start, beta_end, T)
        alphas           = 1.0 - betas
        alpha_cumprod    = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("betas",                      betas)
        self.register_buffer("alpha_cumprod",              alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev",         alpha_cumprod_prev)
        self.register_buffer("sqrt_alpha_cumprod",         torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))

def extract(a, t, x_shape):
    B   = t.shape[0]
    out = a.cpu().gather(-1, t.cpu())
    return out.reshape(B, *((1,) * (len(x_shape) - 1))).to(t.device)

# ============================================================================
# TIME EMBEDDING & BUILDING BLOCKS
# ============================================================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        emb  = math.log(10000) / (half - 1)
        emb  = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb  = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class TimeEmbedding(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.pos = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.SiLU(),
                                  nn.Linear(hidden_dim, hidden_dim))
    def forward(self, t): return self.mlp(self.pos(t))

class TimeAwareConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.b1 = nn.BatchNorm2d(out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.b2 = nn.BatchNorm2d(out_ch)
        self.tm = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, out_ch * 2))
    def forward(self, x, te):
        h = F.relu(self.b1(self.c1(x)), inplace=True)
        s, sh = self.tm(te).chunk(2, dim=1)
        h = h * (1 + s[:, :, None, None]) + sh[:, :, None, None]
        return F.relu(self.b2(self.c2(h)), inplace=True)

class TimeAwareResBlock(nn.Module):
    def __init__(self, ip, p, stride, t_dim):
        super().__init__()
        self.c1 = nn.Conv2d(ip, p, 3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(p)
        self.c2 = nn.Conv2d(p, p, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(p)
        self.tm = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, p * 2))
        self.sc = nn.Sequential() if (stride == 1 and ip == p) else nn.Sequential(
            nn.Conv2d(ip, p, 1, stride=stride, bias=False), nn.BatchNorm2d(p))
    def forward(self, x, te):
        h = F.relu(self.b1(self.c1(x)), inplace=True)
        s, sh = self.tm(te).chunk(2, dim=1)
        h = h * (1 + s[:, :, None, None]) + sh[:, :, None, None]
        return F.relu(self.b2(self.c2(h)) + self.sc(x), inplace=True)

# ============================================================================
# MODEL 1: DiffusionUNet
# ============================================================================
class DiffusionUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, time_emb_dim=128):
        super().__init__()
        D = time_emb_dim
        self.te  = TimeEmbedding(D, D)
        self.e1  = TimeAwareConvBlock(in_channels, 32,  D)
        self.e2  = TimeAwareConvBlock(32,  64,  D)
        self.e3  = TimeAwareConvBlock(64,  128, D)
        self.e4  = TimeAwareConvBlock(128, 256, D)
        self.bot = TimeAwareConvBlock(256, 512, D)
        self.u4  = nn.ConvTranspose2d(512, 256, 2, 2); self.d4 = TimeAwareConvBlock(512, 256, D)
        self.u3  = nn.ConvTranspose2d(256, 128, 2, 2); self.d3 = TimeAwareConvBlock(256, 128, D)
        self.u2  = nn.ConvTranspose2d(128, 64,  2, 2); self.d2 = TimeAwareConvBlock(128, 64,  D)
        self.u1  = nn.ConvTranspose2d(64,  32,  2, 2); self.d1 = TimeAwareConvBlock(64,  32,  D)
        self.out = nn.Conv2d(32, out_channels, 1)
        self.mp  = nn.MaxPool2d(2)
    def forward(self, x, t):
        te = self.te(t)
        e1 = self.e1(x, te); e2 = self.e2(self.mp(e1), te)
        e3 = self.e3(self.mp(e2), te); e4 = self.e4(self.mp(e3), te)
        b  = self.bot(self.mp(e4), te)
        x  = self.d4(torch.cat([self.u4(b),  e4], 1), te)
        x  = self.d3(torch.cat([self.u3(x),  e3], 1), te)
        x  = self.d2(torch.cat([self.u2(x),  e2], 1), te)
        x  = self.d1(torch.cat([self.u1(x),  e1], 1), te)
        return self.out(x)

# ============================================================================
# MODEL 2: DiffusionUNetPlusPlus
# ============================================================================
class DiffusionUNetPlusPlus(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, time_emb_dim=128):
        super().__init__()
        D = time_emb_dim; f = [32, 64, 128, 256, 512]
        self.te = TimeEmbedding(D, D)
        B = lambda i, o: TimeAwareConvBlock(i, o, D)
        self.c00=B(in_channels,f[0]); self.c10=B(f[0],f[1]); self.c20=B(f[1],f[2])
        self.c30=B(f[2],f[3]);        self.c40=B(f[3],f[4])
        self.c01=B(f[0]+f[1],f[0]);   self.c11=B(f[1]+f[2],f[1]); self.c21=B(f[2]+f[3],f[2]); self.c31=B(f[3]+f[4],f[3])
        self.c02=B(f[0]*2+f[1],f[0]); self.c12=B(f[1]*2+f[2],f[1]); self.c22=B(f[2]*2+f[3],f[2])
        self.c03=B(f[0]*3+f[1],f[0]); self.c13=B(f[1]*3+f[2],f[1])
        self.c04=B(f[0]*4+f[1],f[0])
        self.up=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
        self.mp=nn.MaxPool2d(2); self.out=nn.Conv2d(f[0],out_channels,1)
    @staticmethod
    def cr(t, ref):
        return F.interpolate(t, ref.shape[2:], mode="bilinear", align_corners=True) if t.shape[2:]!=ref.shape[2:] else t
    def forward(self, x, t):
        te=self.te(t); up=self.up; mp=self.mp; cr=self.cr
        x00=self.c00(x,te); x10=self.c10(mp(x00),te); x20=self.c20(mp(x10),te)
        x30=self.c30(mp(x20),te); x40=self.c40(mp(x30),te)
        x01=self.c01(torch.cat([x00,cr(up(x10),x00)],1),te)
        x11=self.c11(torch.cat([x10,cr(up(x20),x10)],1),te)
        x21=self.c21(torch.cat([x20,cr(up(x30),x20)],1),te)
        x31=self.c31(torch.cat([x30,cr(up(x40),x30)],1),te)
        x02=self.c02(torch.cat([x00,x01,cr(up(x11),x00)],1),te)
        x12=self.c12(torch.cat([x10,x11,cr(up(x21),x10)],1),te)
        x22=self.c22(torch.cat([x20,x21,cr(up(x31),x20)],1),te)
        x03=self.c03(torch.cat([x00,x01,x02,cr(up(x12),x00)],1),te)
        x13=self.c13(torch.cat([x10,x11,x12,cr(up(x22),x10)],1),te)
        x04=self.c04(torch.cat([x00,x01,x02,x03,cr(up(x13),x00)],1),te)
        return self.out(x04)

# ============================================================================
# MODEL 3: DiffusionResNetUNet
# ============================================================================
class DiffusionResNetUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, time_emb_dim=128):
        super().__init__()
        D=time_emb_dim; self.D=D; self.te=TimeEmbedding(D,D)
        self.stem=nn.Sequential(nn.Conv2d(in_channels,64,7,stride=2,padding=3,bias=False),
                                 nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.p0=nn.MaxPool2d(3,stride=2,padding=1)
        self.e1=self._layer(64,64,3,1);   self.e2=self._layer(64,128,4,2)
        self.e3=self._layer(128,256,6,2); self.e4=self._layer(256,512,3,2)
        self.bot=TimeAwareConvBlock(512,1024,D)
        self.u4=nn.ConvTranspose2d(1024,256,2,2); self.d4=TimeAwareConvBlock(512,256,D)
        self.u3=nn.ConvTranspose2d(256,128,2,2);  self.d3=TimeAwareConvBlock(256,128,D)
        self.u2=nn.ConvTranspose2d(128,64,2,2);   self.d2=TimeAwareConvBlock(128,64,D)
        self.u1=nn.ConvTranspose2d(64,32,2,2);    self.d1=TimeAwareConvBlock(96,32,D)
        self.u0=nn.ConvTranspose2d(32,16,2,2); self.out=nn.Conv2d(16,out_channels,1)
    def _layer(self, ip, p, n, s):
        L=[TimeAwareResBlock(ip,p,s,self.D)]
        for _ in range(1,n): L.append(TimeAwareResBlock(p,p,1,self.D))
        return nn.ModuleList(L)
    def _run(self, layers, x, te):
        for l in layers: x=l(x,te)
        return x
    def forward(self, x, t):
        te=self.te(t); s=self.stem(x); p=self.p0(s)
        e1=self._run(self.e1,p,te); e2=self._run(self.e2,e1,te)
        e3=self._run(self.e3,e2,te); e4=self._run(self.e4,e3,te)
        b=self.bot(e4,te)
        x=self.d4(torch.cat([self.u4(b),e3],1),te)
        x=self.d3(torch.cat([self.u3(x),e2],1),te)
        x=self.d2(torch.cat([self.u2(x),e1],1),te)
        x=self.d1(torch.cat([self.u1(x),s],1),te)
        return self.out(self.u0(x))

# ============================================================================
# MODEL REGISTRY
# ============================================================================
MODEL_REGISTRY = {
    "unet_diff":    DiffusionUNet,
    "unetpp_diff":  DiffusionUNetPlusPlus,
    "resunet_diff": DiffusionResNetUNet,
}

# ============================================================================
# DATA
# ============================================================================
def generate_lung_mask(img):
    n = (img - img.min()) / (img.max() - img.min() + 1e-8)
    m = n > np.percentile(n, 25)
    m = binary_dilation(m, iterations=3)
    m = binary_erosion(m, iterations=2)
    m = gaussian_filter(m.astype(np.float32), sigma=2.0)
    return (m > 0.5).astype(np.float32)

class LungSegDataset(Dataset):
    """Pre-caches all images & masks into RAM at init for fast __getitem__."""
    def __init__(self, pairs, image_size=256, augment=False):
        self.aug = augment
        n = len(pairs)
        print(f"  [Dataset] Pre-caching {n} slices into RAM …", flush=True)
        self.imgs  = np.empty((n, image_size, image_size), dtype=np.float32)
        self.masks = np.empty((n, image_size, image_size), dtype=np.float32)
        for i, (ip, mp) in enumerate(pairs):
            img = Image.open(ip).convert("L").resize((image_size, image_size), Image.BILINEAR)
            ia  = np.array(img, np.float32) / 255.0
            if mp and Path(mp).exists():
                mask = Image.open(mp).convert("L").resize((image_size, image_size), Image.NEAREST)
                ma   = (np.array(mask, np.float32) > 128).astype(np.float32)
            else:
                ma = generate_lung_mask(ia)
            self.imgs[i]  = ia
            self.masks[i] = ma
            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{n} slices cached …", flush=True)
        print(f"  [Dataset] Done — {n} slices ready.", flush=True)

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        ia = self.imgs[idx].copy()
        ma = self.masks[idx].copy()
        if self.aug and random.random() > 0.5:
            ia = np.fliplr(ia).copy(); ma = np.fliplr(ma).copy()
        if self.aug and random.random() > 0.5:
            ia = np.flipud(ia).copy(); ma = np.flipud(ma).copy()
        return torch.from_numpy(ia).unsqueeze(0), torch.from_numpy(ma).unsqueeze(0)

def find_image_pairs(image_root, split="train"):
    pairs=[]; sd = image_root / split
    if not sd.exists(): return pairs
    for cid in sorted(os.listdir(sd)):
        cd = sd / cid
        if not cd.is_dir(): continue
        slices = sorted([f for f in os.listdir(cd) if f.endswith(".jpg")],
                        key=lambda x: int(x.split(".")[0]))
        if len(slices) > MAX_SLICES_PER_CASE:
            step = len(slices)/MAX_SLICES_PER_CASE
            slices = [slices[int(i*step)] for i in range(MAX_SLICES_PER_CASE)]
        for f in slices:
            p = cd/f
            if p.exists(): pairs.append((str(p), None))
    return pairs

# ============================================================================
# DIFFUSION FORWARD PASS UTILITIES
# ============================================================================
def diffuse(imgs, masks, schedule):
    """Add noise to masks according to random timestep."""
    B = imgs.size(0)
    t        = torch.randint(0, schedule.T, (B,), device=DEVICE, dtype=torch.long)
    noise    = torch.randn_like(masks)
    sqrt_ac  = extract(schedule.sqrt_alpha_cumprod,           t, masks.shape)
    sqrt_oac = extract(schedule.sqrt_one_minus_alpha_cumprod, t, masks.shape)
    x_t = sqrt_ac * masks + sqrt_oac * noise
    return torch.cat([imgs, x_t], dim=1), t, noise

# ============================================================================
# DDPM SAMPLING — for inline Dice validation
# ============================================================================
@torch.no_grad()
def quick_dice(model, loader, schedule, n_steps=10):
    """Compute approximate Dice on validation set using short DDPM sampling."""
    model.eval()
    tp_sum = fp_sum = fn_sum = 0.0
    for imgs, masks in loader:
        imgs = imgs.to(DEVICE)
        B, _, H, W = imgs.shape
        x = torch.randn(B, 1, H, W, device=DEVICE)
        times = torch.linspace(schedule.T-1, 0, n_steps, device=DEVICE).long()
        for i in range(n_steps-1):
            t_b  = times[i].expand(B)
            inp  = torch.cat([imgs, x], 1)
            pn   = model(inp, t_b)
            ac   = extract(schedule.alpha_cumprod, t_b, x.shape)
            x    = (x - torch.sqrt(1-ac)*pn) / torch.sqrt(ac)
            x    = torch.clamp(x, -1, 1)
        pred = (x > 0).float().cpu()
        gt   = masks
        tp_sum += (pred * gt).sum().item()
        fp_sum += (pred * (1-gt)).sum().item()
        fn_sum += ((1-pred) * gt).sum().item()
    dice = 2*tp_sum / (2*tp_sum + fp_sum + fn_sum + 1e-8)
    return dice

# ============================================================================
# TRAINING LOOP
# ============================================================================
def train_model_fold(model_key, model_cls, fold_idx, train_loader, val_loader, schedule):
    print(f"\n  [{model_key}] Fold {fold_idx+1}/{NUM_FOLDS}")
    print(f"  {'-'*66}")

    model     = model_cls(in_channels=2, out_channels=1, time_emb_dim=TIME_EMB_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # Cosine annealing with warm restarts for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-5)
    scaler    = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    ckpt_dir  = OUTPUT_DIR / "checkpoints" / model_key
    ckpt_path = ckpt_dir / f"best_{model_cls.__name__}_fold_{fold_idx+1}.pt"

    best_val_loss    = float("inf")
    best_dice        = 0.0
    patience_counter = 0
    fold_metrics     = []

    for epoch in range(NUM_EPOCHS):
        # ── TRAIN ────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0; n_train = 0
        for imgs, masks in train_loader:
            imgs  = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                inp, t, noise = diffuse(imgs, masks, schedule)
                pred_noise    = model(inp, t)
                loss          = criterion(pred_noise, noise)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            train_loss += loss.item() * imgs.size(0)
            n_train    += imgs.size(0)
        train_loss /= n_train

        # ── VALIDATE (loss) ──────────────────────────────────────────────
        model.eval()
        val_loss = 0.0; n_val = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs  = imgs.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    inp, t, noise = diffuse(imgs, masks, schedule)
                    pn   = model(inp, t)
                    loss = criterion(pn, noise)
                val_loss += loss.item() * imgs.size(0)
                n_val    += imgs.size(0)
        val_loss /= n_val
        scheduler.step(epoch + fold_idx * NUM_EPOCHS)   # global step for CosineAnnealing

        lr = optimizer.param_groups[0]["lr"]

        # Compute Dice every 5 epochs (sampling is slow)
        dice_str = ""
        if (epoch+1) % 5 == 0 or epoch == 0:
            dice = quick_dice(model, val_loader, schedule, n_steps=10)
            best_dice = max(best_dice, dice)
            dice_str = f" | Dice~{dice:.4f}"

        print(f"    Ep {epoch+1:3d}/{NUM_EPOCHS} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"LR: {lr:.2e}{dice_str}")

        fold_metrics.append({"epoch": epoch+1, "train_loss": train_loss,
                              "val_loss": val_loss, "lr": lr})

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"    [BEST] Saved → {ckpt_path.name}  (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n    [STOP] Early stopping at epoch {epoch+1}")
            break

    # Save per-fold metrics JSON (used by generate_figures.py)
    with open(ckpt_dir / f"metrics_fold_{fold_idx+1}.json", "w") as f:
        json.dump(fold_metrics, f, indent=2)

    print(f"  [OK] {model_key} fold {fold_idx+1} | best_val_loss={best_val_loss:.6f} | best_dice≈{best_dice:.4f}")

    del model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return best_val_loss, str(ckpt_path), best_dice

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("\n>>> Starting GPU-optimised Diffusion Segmentation Training...\n")

    train_pairs = find_image_pairs(PROCESSED_IMG_DIR, "train")
    test_pairs  = find_image_pairs(PROCESSED_IMG_DIR, "test")
    print(f"  Train: {len(train_pairs)} slices")
    print(f"  Test : {len(test_pairs)} slices\n")

    if not train_pairs:
        print("[ERROR] No training data found!"); sys.exit(1)

    schedule     = DiffusionSchedule(DIFFUSION_TIMESTEPS, BETA_START, BETA_END).to(DEVICE)
    full_dataset = LungSegDataset(train_pairs, IMAGE_SIZE, augment=True)
    kfold        = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    fold_splits  = list(kfold.split(train_pairs))

    all_results = {mk: {} for mk in MODEL_REGISTRY}

    for model_key, model_cls in MODEL_REGISTRY.items():
        n_params = sum(p.numel() for p in model_cls(2, 1, TIME_EMB_DIM).parameters())
        print(f"\n{'='*70}")
        print(f"  TRAINING {model_key.upper()}  ({n_params:,} params)")
        print(f"{'='*70}")

        for fold_idx, (tr_idx, va_idx) in enumerate(fold_splits):
            train_loader = DataLoader(
                torch.utils.data.Subset(full_dataset, tr_idx),
                batch_size=BATCH_SIZE, shuffle=True,
                num_workers=0, pin_memory=USE_AMP)
            val_loader = DataLoader(
                torch.utils.data.Subset(full_dataset, va_idx),
                batch_size=BATCH_SIZE, shuffle=False,
                num_workers=0, pin_memory=USE_AMP)

            best_loss, ckpt, best_dice = train_model_fold(
                model_key, model_cls, fold_idx,
                train_loader, val_loader, schedule)
            all_results[model_key][fold_idx] = {
                "best_val_loss": best_loss,
                "best_dice":     best_dice,
                "checkpoint":    ckpt
            }

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  === TRAINING COMPLETE ===")
    print(f"{'='*70}")
    print(f"\n  {'Model':<20} {'F1 Loss':>10} {'F2 Loss':>10} {'F3 Loss':>10} "
          f"{'Avg Loss':>10} {'Avg Dice':>10}")
    print(f"  {'-'*70}")
    for mk, folds in all_results.items():
        losses = [folds[fi]["best_val_loss"] for fi in range(NUM_FOLDS)]
        dices  = [folds[fi]["best_dice"]     for fi in range(NUM_FOLDS)]
        print(f"  {mk:<20} {losses[0]:>10.6f} {losses[1]:>10.6f} {losses[2]:>10.6f} "
              f"{np.mean(losses):>10.6f} {np.mean(dices):>10.4f}")

    with open(OUTPUT_DIR / "tables" / "training_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Summary → {OUTPUT_DIR / 'tables' / 'training_summary.json'}")
    print(f"{'='*70}\n")
