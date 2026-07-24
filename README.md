# Diffusion-Based Lung Segmentation on CT

Recovering clean lung masks from pure noise on 2D CT slices — comparing three U-Net engines inside one diffusion framework.

*Medical Imaging · Deep Learning — course project.*

---

## Overview

The task is **segmentation**: label every one of the 65,536 pixels in a 256×256 CT slice as *lung* or *not-lung*, producing a binary mask. Lung edges are noisy and irregular, and even expert radiologists disagree on the exact boundary — so the core challenge is recovering a clean, anatomically correct mask from imaging that is inherently fuzzy at the edges.

A scan is a stack of 2D slices; the model segments each one independently, from the top of the lungs to the base.

## Why diffusion for medical data

Diffusion reframes segmentation as **denoising** — recovering a clean mask from a corrupted one. Three properties make it a strong fit for medical imaging:

- **Robustness** — training to denoise across many noise levels teaches the model to handle ambiguous, low-contrast boundaries, exactly the conditions in real CT scans.
- **Iterative refinement** — the mask is built through many small denoising corrections rather than one rigid guess; the prediction is refined step by step instead of committed all at once.
- **Uncertainty for free** — sampling from different random starts yields several plausible masks. Where they agree, the model is confident; where they disagree marks the uncertain boundary — genuine clinical value.

## Approach

**Diffusion is the method; the U-Net is the engine.**

The key twist: it is *not* the CT scan that gets noised — it is the **mask**. The CT stays clean and is fed in as a hint at every step. Training predicts the added noise; inference denoises pure static back into a mask.

```
Noisy mask → Predict noise → DDIM step ×50 → Clean lung mask
```

### Three engines, one diffusion framework

| Engine | Description | Params | Speed / slice |
|---|---|---|---|
| **U-Net** | The baseline — a symmetric encoder–decoder whose skip connections hand spatial detail back to the decoder. | ~8.2M | ~73 ms |
| **U-Net++** | Replaces each long skip with a ladder of small refining blocks, so raw and abstract features fuse cleanly. | ~9.7M | ~110 ms |
| **ResNet-UNet** | A ResNet-block encoder adds depth and residual shortcuts for a sturdier, deeper backbone. | ~40.4M | ~146 ms |

### DDIM: making diffusion practical at test time

| Aspect | DDPM (original) | DDIM (used here) |
|---|---|---|
| Denoising steps | 1000, one at a time | 10–50 total |
| Step skipping | Not allowed | Skip freely, e.g. 1000 → 50 |
| Randomness | Fresh noise every step | Optional, controlled by `eta` |
| Reproducibility | Varies run to run | `eta = 0` gives identical output |
| Speed | Days for the full test set | 20–100× faster |

This project uses `INFERENCE_STEPS = 50`.

## Results

**Patient-level Dice is what counts** — it averages per case and is the clinically meaningful metric. Slice-level numbers pool all pixels and read optimistically higher, so always check the aggregation.

| Engine | Patient-level Dice |
|---|---|
| **U-Net++** | **0.9341** |
| U-Net | 0.9151 |
| ResNet-UNet | 0.9093 |

Slice-level (pixel-aggregated): Dice ~0.986 · IoU ~0.973 · Accuracy ~0.985.

Training and validation loss fall steeply then converge across all three folds — stable learning, no divergence. U-Net++ has the tightest, lowest validation-loss spread and reaches the lowest best validation loss; ResNet-UNet's spread is wider with a heavier tail. All three converge late, around epoch 140–150.

Qualitatively, predictions track the true lung fields closely, with most disagreement sitting at the fuzzy boundaries — exactly where diffusion's step-by-step refinement helps.

## Findings — U-Net++ is the production pick

- **Highest patient-level Dice (0.9341)** — the most clinically meaningful metric here.
- Nested dense skips fuse multi-scale features better, helping across varied anatomies.
- Moderate cost (~9.7M params, ~110 ms per slice) — fine for batch or near-real-time use.
- **U-Net** is the fallback when latency or memory is tight — fastest at ~73 ms, smallest footprint.
- **ResNet-UNet** has the most capacity but is ~5× larger and slowest — offline use only.

## Key takeaways

- **Diffusion is the method, not the model.** The U-Net is the engine that predicts noise at every step; diffusion is the procedure around it. The experiment simply asks which engine works best.
- **U-Net++ is the best trade-off** — leading on patient-level Dice at moderate size and speed.
- **Read the metrics carefully.** Slice-level Dice (~0.986) looks great but is optimistic; patient-level averages (~0.91–0.93) are the honest, clinically relevant numbers.

## Limitations

- Patient-level results come from a limited set of test cases — broader multi-centre validation is still needed.
- Only three folds; some validation Dice is estimated from short DDIM sampling, not full inference.
- Works on 2D slices only — no full 3D volumetric context between slices.
- Where ground-truth masks were missing, a heuristic lung mask was substituted.
- Inference is slower than one-shot U-Nets, even with DDIM at 50 steps.

## Future work

- Extend to full 3D diffusion for cross-slice consistency.
- Quantize (INT8) and batch slices to cut latency and memory for deployment.
- Ship uncertainty maps from multiple stochastic samples per case.
- Ablate DDIM steps (5 / 10 / 20 / 50) to fix the speed–quality trade-off.
- Validate on external datasets and add morphological mask cleanup.

## Tech stack

PyTorch · torchvision · NumPy · Matplotlib · Jupyter Notebook
