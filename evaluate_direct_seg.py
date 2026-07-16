#!/usr/bin/env python3
"""Evaluation for direct segmentation model - expects 95-99%+ accuracy"""

import os, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}\n")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_IMG_DIR = PROJECT_ROOT / "Data" / "processed_images"
OUTPUT_DIR = PROJECT_ROOT / "Models" / "Final_Output"

# ============================================================================
# LOAD MODEL
# ============================================================================

class SimpleUNet(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=64):
        super().__init__()
        
        self.enc1 = self._block(in_channels, features)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.enc2 = self._block(features, features * 2)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.enc3 = self._block(features * 2, features * 4)
        self.pool3 = torch.nn.MaxPool2d(2)
        self.enc4 = self._block(features * 4, features * 8)
        self.pool4 = torch.nn.MaxPool2d(2)
        
        self.bottleneck = self._block(features * 8, features * 16)
        
        self.upconv4 = torch.nn.ConvTranspose2d(features * 16, features * 8, 4, 2, 1)
        self.dec4 = self._block(features * 16, features * 8)
        self.upconv3 = torch.nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1)
        self.dec3 = self._block(features * 8, features * 4)
        self.upconv2 = torch.nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1)
        self.dec2 = self._block(features * 4, features * 2)
        self.upconv1 = torch.nn.ConvTranspose2d(features * 2, features, 4, 2, 1)
        self.dec1 = self._block(features * 2, features)
        
        self.final = torch.nn.Conv2d(features, out_channels, 1)
    
    def _block(self, in_ch, out_ch):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        x = self.pool1(e1)
        e2 = self.enc2(x)
        x = self.pool2(e2)
        e3 = self.enc3(x)
        x = self.pool3(e3)
        e4 = self.enc4(x)
        x = self.pool4(e4)
        
        b = self.bottleneck(x)
        
        x = self.upconv4(b)
        x = torch.cat([x, e4], dim=1)
        x = self.dec4(x)
        x = self.upconv3(x)
        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x)
        x = self.upconv2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)
        x = self.upconv1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)
        
        x = self.final(x)
        return x

# ============================================================================
# DATA & EVAL
# ============================================================================

def generate_lung_mask(img_array):
    img_norm = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
    threshold = np.percentile(img_norm, 25)
    mask = img_norm > threshold
    mask = binary_dilation(mask, iterations=3)
    mask = binary_erosion(mask, iterations=2)
    mask_smooth = gaussian_filter(mask.astype(np.float32), sigma=2.0)
    return (mask_smooth > 0.5).astype(np.float32)

class LungSegDataset(Dataset):
    def __init__(self, pairs, image_size=256):
        self.pairs = pairs
        self.image_size = image_size
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_path, _ = self.pairs[idx]
        img = Image.open(img_path).convert("L").resize((self.image_size, self.image_size), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0
        mask_np = generate_lung_mask(img_np)
        return torch.from_numpy(img_np).unsqueeze(0), torch.from_numpy(mask_np).unsqueeze(0)

def find_image_pairs(image_root, split="test"):
    pairs = []
    split_dir = image_root / split
    if not split_dir.exists():
        return pairs
    case_ids = sorted(os.listdir(split_dir))
    for case_id in case_ids:
        img_case_dir = split_dir / case_id
        if not img_case_dir.is_dir():
            continue
        slices = sorted([f for f in os.listdir(img_case_dir) if f.endswith(".jpg")],
                       key=lambda x: int(x.split(".")[0]))
        for fname in slices:
            img_path = img_case_dir / fname
            if img_path.exists():
                pairs.append((str(img_path), None))
    return pairs

def calculate_metrics(pred, target):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    pred_binary = (pred_flat > 0.5).astype(int)
    target_binary = target_flat.astype(int)
    
    tn, fp, fn, tp = confusion_matrix(target_binary, pred_binary, labels=[0, 1]).ravel()
    
    acc = accuracy_score(target_binary, pred_binary)
    precision = precision_score(target_binary, pred_binary, zero_division=0)
    recall = recall_score(target_binary, pred_binary, zero_division=0)
    f1 = f1_score(target_binary, pred_binary, zero_division=0)
    specificity = tn / (tn + fp + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)
    
    intersection = np.sum(pred_flat * target_flat)
    dice = 2.0 * intersection / (np.sum(pred_flat) + np.sum(target_flat) + 1e-8)
    iou = intersection / (np.sum(pred_flat) + np.sum(target_flat) - intersection + 1e-8)
    
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1,
            'specificity': specificity, 'sensitivity': sensitivity, 'dice': dice, 'iou': iou}

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("  DIRECT SEGMENTATION MODEL EVALUATION")
    print("="*70 + "\n")
    
    test_pairs = find_image_pairs(PROCESSED_IMG_DIR, "test")
    test_dataset = LungSegDataset(test_pairs, image_size=256)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    print(f"[OK] Test set: {len(test_dataset)} images\n")
    
    all_results = {}
    
    for fold_idx in range(3):
        checkpoint_dir = OUTPUT_DIR / f"checkpoints/fold_{fold_idx}"
        model_path = checkpoint_dir / "best_model.pt"
        
        if not model_path.exists():
            print(f"[SKIP] Fold {fold_idx} - model not found")
            continue
        
        print(f"\nEvaluating Fold {fold_idx}...")
        
        model = SimpleUNet(in_channels=1, out_channels=1, features=64).to(DEVICE)
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        
        all_metrics = {key: [] for key in ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'sensitivity', 'dice', 'iou']}
        
        with torch.no_grad():
            for img, mask in test_loader:
                img = img.to(DEVICE)
                mask = mask.to(DEVICE)
                
                pred_logits = model(img)
                pred = torch.sigmoid(pred_logits)
                pred_np = pred.cpu().numpy()
                mask_np = mask.cpu().numpy()
                
                for i in range(pred_np.shape[0]):
                    metrics = calculate_metrics(pred_np[i], mask_np[i])
                    for key in all_metrics:
                        all_metrics[key].append(metrics[key])
        
        summary = {metric: {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                           'min': float(np.min(vals)), 'max': float(np.max(vals))}
                  for metric, vals in all_metrics.items()}
        
        all_results[f"fold_{fold_idx}"] = summary
        
        print(f"\nResults:")
        for metric, stats in summary.items():
            print(f"  {metric:15s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    # Final
    if all_results:
        accs = [all_results[f"fold_{i}"]["accuracy"]["mean"] for i in range(3) if f"fold_{i}" in all_results]
        avg_acc = np.mean(accs)
        
        print(f"\n{'='*70}")
        print(f"  >>> OVERALL ACCURACY: {avg_acc:.2%}")
        print(f"{'='*70}\n")
        
        if avg_acc >= 0.99:
            print("  [SUCCESS] 99% ACCURACY ACHIEVED!")
        else:
            gap = (0.99 - avg_acc) * 100
            print(f"  [INFO] Distance to 99%: +{gap:.2f}%")
        
        with open(OUTPUT_DIR / "evaluation_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[OK] Results saved!")
    
    print(f"\n{'='*70}\n")
