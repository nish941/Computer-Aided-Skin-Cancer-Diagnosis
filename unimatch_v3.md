# UniMatch V2 → SECOND Dataset: Full Analysis

> **Your role:** You own UniMatch V2 in a group project doing Semi-Supervised Semantic Change Detection on remote sensing satellite imagery. This document covers the full state of the codebase, what needs to be built, every bug found, and how to get the best results on SECOND.

---

## 1. What the Code Is Actually Doing Right Now

### Architecture

The existing codebase is a clean, well-structured semi-supervised **single-image semantic segmentation** framework. Here is what each component does:

**Backbone — DINOv2 (`dinov2.py`)**
A Vision Transformer pre-trained on 142M curated images. Called via `get_intermediate_layers()` which extracts feature maps from 4 specified transformer blocks (e.g. layers [2,5,8,11] for ViT-Base). Each feature map is shape `[B, N_patches, embed_dim]` where `embed_dim` is 384/768/1024/1536 depending on model size.

**Decoder — DPT (`dpt.py`)**
Dense Prediction Transformer head. Takes the 4 feature maps, reshapes them from patch tokens back to 2D grids, applies 1×1 projection convolutions, then resizes each to different spatial scales and fuses them bottom-up via `FeatureFusionBlock` (refinenet) layers. Final output is a `[B, nclass, H, W]` segmentation map bilinearly interpolated to full image resolution.

The `comp_drop=True` path inside `DPT.forward()` is the Complementary Dropout — the paper's core contribution. It generates a binomial binary mask over the channel dimension for the first half of the batch, uses `2.0 - mask` as the complement for the second half, scales by ×2 to preserve expectation, then multiplies each of the 4 feature maps by the same mask before decoding.

**Main training method — UniMatch V2 (`unimatch_v2.py`)**
- EMA teacher-student: teacher is a `deepcopy` of the model, updated via `param_ema = param_ema * ema_ratio + param * (1 - ema_ratio)` every iteration.
- Labeled batch: straightforward CE/OHEM loss on ground-truth masks.
- Unlabeled batch: teacher produces pseudo-labels on weak views (random crop + hflip), filtered by `conf_thresh=0.95`. Student trains on two strongly-augmented versions (color jitter + blur + CutMix) with complementary dropout. Loss averaged over both streams and normalized by valid (non-ignore) pixels.
- Two CutMix boxes (cutmix_box1, cutmix_box2) applied independently to the two strong views. Pseudo-labels and confidence maps are blended using the same boxes to match.

**Dataset — `semi.py`**
`SemiDataset` reads a text file where each line is `"img_path mask_path"`. For unlabeled mode, it creates a zeroed dummy mask. Applies: resize(0.5–2.0), crop to `crop_size`, hflip(p=0.5) for both weak and strong views. Strong views additionally get color jitter, grayscale, gaussian blur. Returns a 6-tuple: `(img_w, img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2)`.

**Evaluation — `supervised.py` shared `evaluate()`**
Supports `'original'` mode (resize to nearest multiple of 14, then resize predictions back) and `'sliding_window'` mode (tile the image with 2/3 overlap). Uses distributed `all_reduce` for correct multi-GPU metrics. Reports per-class IoU and mIoU for both student and EMA teacher models each epoch.

---

## 2. What's Already Working Correctly

| Component | Status | Notes |
|-----------|--------|-------|
| EMA teacher-student | ✅ Correct | Formula matches paper exactly: `min(1-1/(iter+1), 0.996)` |
| Complementary dropout | ✅ Correct | Binomial mask, complement, ×2 scaling — all per the paper |
| Dual-stream CutMix | ✅ Correct | Pseudo-labels blended with same boxes — subtle but right |
| Confidence thresholding | ✅ Correct | τ=0.95, normalizes by non-ignore pixels |
| Poly LR schedule | ✅ Correct | Separate LR groups for encoder (5e-6) and decoder (2e-4) |
| Checkpoint resume | ✅ Correct | Saves all state, `map_location='cpu'` on load |
| Distributed training | ✅ Correct | SyncBN, DDP, DistributedSampler, all_reduce in eval |
| TensorBoard logging | ✅ Complete | Loss, mask_ratio, per-class IoU logged every iter/epoch |
| OHEM loss | ✅ Available | `ProbOhemCrossEntropy2d` — important for class imbalance |
| Labeled upsampling | ✅ Correct | `nsample` repeats labeled data to match unlabeled dataset size |

---

## 3. Bugs Found in the Code

### BUG 1 — Generator instead of tuple in comp_drop `dpt.py:161`
**Severity: Critical**

```python
# CURRENT (WRONG):
features = (feature * dropout_mask.unsqueeze(1) for feature in features)
# This is a Python generator — consumed exactly once, then empty.
# If DPTHead.forward() ever iterates twice, or if len() is called,
# it silently produces wrong results or raises StopIteration.

# FIX:
features = tuple(feature * dropout_mask.unsqueeze(1) for feature in features)
```

### BUG 2 — EMA update on DDP wrapper instead of inner module `unimatch_v2.py:223`
**Severity: Critical**

```python
# CURRENT (WRONG):
for param, param_ema in zip(model.parameters(), model_ema.parameters()):
    param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
# model is DDP-wrapped. model.parameters() may include DDP internal tensors.
# Using .module accesses the actual DPT model directly.

# FIX:
for param, param_ema in zip(model.module.parameters(), model_ema.module.parameters()):
    param_ema.data.mul_(ema_ratio).add_(param.detach().data * (1 - ema_ratio))
for buf, buf_ema in zip(model.module.buffers(), model_ema.module.buffers()):
    buf_ema.copy_(buf_ema * ema_ratio + buf.detach() * (1 - ema_ratio))
```

### BUG 3 — mask_ratio logged from original ignore_mask, not CutMixed `unimatch_v2.py:213`
**Severity: Medium (monitoring/analysis bug)**

```python
# CURRENT (WRONG):
mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() \
             / (ignore_mask != 255).sum()
# Uses original ignore_mask. The actual training loss uses ignore_mask_cutmixed1/2.
# The logged ratio does NOT reflect the actual pseudo-label usage.

# FIX:
used1 = ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255)).sum()
used2 = ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255)).sum()
total = (ignore_mask_cutmixed1 != 255).sum() + (ignore_mask_cutmixed2 != 255).sum()
mask_ratio = (used1 + used2).float().item() / (total.float().item() + 1e-10)
```

### BUG 4 — `is_best` uses `>=` causing stale overwrites `unimatch_v2.py:255`
**Severity: Low**

```python
# CURRENT (INCONSISTENT):
is_best = mIoU >= previous_best   # saves best.pth on tie
if mIoU == previous_best:         # updates best_epoch on exact equality
    best_epoch = epoch
# On a tie: best.pth is overwritten (possibly with a less stable model)
# but best_epoch IS updated. Inconsistent.

# FIX:
is_best = mIoU > previous_best
```

### BUG 5 — ignore_mask created with float64 dtype `semi.py:66`
**Severity: Low**

```python
# CURRENT (WRONG DTYPE):
ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))
# np.zeros() defaults to float64 → PIL creates mode 'F' image
# When converted via torch.from_numpy(...).long(), float values
# behave unexpectedly in boolean operations like (ignore_mask != 255)

# FIX:
ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0]), dtype=np.uint8))
```

### BUG 6 — Infinite loop possible in obtain_cutmix_box `transform.py:72`
**Severity: Low (rare but real)**

```python
# CURRENT (UNSAFE):
while True:
    ratio = np.random.uniform(ratio_1, ratio_2)
    cutmix_w = int(np.sqrt(size / ratio))
    cutmix_h = int(np.sqrt(size * ratio))
    x = np.random.randint(0, img_size)
    y = np.random.randint(0, img_size)
    if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
        break
# If cutmix_w or cutmix_h > img_size (possible with extreme ratios),
# this loops forever.

# FIX:
for _ in range(300):
    ratio = np.random.uniform(ratio_1, ratio_2)
    cutmix_w = int(np.sqrt(size / ratio))
    cutmix_h = int(np.sqrt(size * ratio))
    x = np.random.randint(0, img_size)
    y = np.random.randint(0, img_size)
    if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
        break
else:  # fallback: anchor top-left
    cutmix_w = min(int(np.sqrt(size)), img_size)
    cutmix_h = min(int(np.sqrt(size)), img_size)
    x, y = 0, 0
mask[y:y + cutmix_h, x:x + cutmix_w] = 1
```

### BUG 7 — Sliding window hardcodes 19 classes `supervised.py:30`
**Severity: Critical if using SECOND (6 classes)**

```python
# CURRENT (HARDCODED):
final = torch.zeros(b, 19, h, w).cuda()
# Hardcoded to Cityscapes. Silently produces completely wrong output
# for any other dataset when eval_mode == 'sliding_window'.

# FIX:
final = torch.zeros(b, cfg['nclass'], h, w).cuda()
```

### BUG 8 — No gradient clipping anywhere
**Severity: Medium (especially for small SECOND labeled sets)**

```python
# Add in unimatch_v2.py, fixmatch.py, supervised.py before optimizer.step():
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

## 4. What's Completely Missing for SECOND

SECOND is a **bi-temporal semantic change detection** dataset. UniMatch V2 is a **single-image semantic segmentation** framework. Every single item below needs to be built from scratch.

### 4.1 SECOND Dataset Loader

The SECOND dataset structure:
```
SECOND/
├── train/
│   ├── im1/      ← T1 images (before)
│   ├── im2/      ← T2 images (after)
│   ├── label1/   ← T1 land-cover masks (0-5)
│   └── label2/   ← T2 land-cover masks (0-5)
└── test/
    ├── im1/
    ├── im2/
    ├── label1/
    └── label2/
```

The new dataset class must:
1. Load **pairs** of images (T1 + T2) and their **pairs** of masks (label1 + label2)
2. Apply geometric augmentations (crop, flip, rotate) with the **same random seed to T1 and T2** to keep them spatially aligned
3. Apply color/photometric augmentations **independently** to T1 and T2 (they were taken at different times with different conditions)

```python
# dataset/second.py
import math, os, random
from copy import deepcopy

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.transform import normalize, resize, crop, hflip, blur, obtain_cutmix_box


class SecondDataset(Dataset):
    def __init__(self, root, mode, size=None, id_path=None, nsample=None):
        self.root = root
        self.mode = mode
        self.size = size

        if mode in ['train_l', 'train_u']:
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None and nsample > len(self.ids):
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('splits/second/val.txt', 'r') as f:
                self.ids = f.read().splitlines()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        parts = self.ids[item].split(' ')
        # parts: [t1_img, t2_img, t1_label, t2_label]
        img_t1 = Image.open(os.path.join(self.root, parts[0])).convert('RGB')
        img_t2 = Image.open(os.path.join(self.root, parts[1])).convert('RGB')

        if self.mode == 'train_u':
            h, w = img_t1.size[1], img_t1.size[0]
            mask_t1 = Image.fromarray(np.zeros((h, w), dtype=np.uint8))
            mask_t2 = Image.fromarray(np.zeros((h, w), dtype=np.uint8))
        else:
            mask_t1 = Image.fromarray(np.array(Image.open(os.path.join(self.root, parts[2]))))
            mask_t2 = Image.fromarray(np.array(Image.open(os.path.join(self.root, parts[3]))))

        if self.mode == 'val':
            img_t1, mask_t1 = normalize(img_t1, mask_t1)
            img_t2, mask_t2 = normalize(img_t2, mask_t2)
            return img_t1, img_t2, mask_t1, mask_t2, self.ids[item]

        # Geometric augmentations: SAME seed for T1 and T2 (spatial alignment!)
        seed = random.randint(0, 2**31)
        random.seed(seed); img_t1, mask_t1 = resize(img_t1, mask_t1, (0.5, 2.0))
        random.seed(seed); img_t2, mask_t2 = resize(img_t2, mask_t2, (0.5, 2.0))

        ignore_value = 254 if self.mode == 'train_u' else 255
        random.seed(seed); img_t1, mask_t1 = crop(img_t1, mask_t1, self.size, ignore_value)
        random.seed(seed); img_t2, mask_t2 = crop(img_t2, mask_t2, self.size, ignore_value)

        random.seed(seed); img_t1, mask_t1 = hflip(img_t1, mask_t1, p=0.5)
        random.seed(seed); img_t2, mask_t2 = hflip(img_t2, mask_t2, p=0.5)

        # Rotation augmentation (satellite imagery has no canonical orientation)
        k = random.randint(0, 3)
        if k > 0:
            random.seed(seed)
            img_t1 = img_t1.rotate(k * 90); mask_t1 = mask_t1.rotate(k * 90)
            img_t2 = img_t2.rotate(k * 90); mask_t2 = mask_t2.rotate(k * 90)

        if self.mode == 'train_l':
            img_t1, mask_t1 = normalize(img_t1, mask_t1)
            img_t2, mask_t2 = normalize(img_t2, mask_t2)
            return img_t1, mask_t1, img_t2, mask_t2

        # Unlabeled: build weak + two strong views for both T1 and T2
        img_w_t1 = deepcopy(img_t1)
        img_w_t2 = deepcopy(img_t2)

        # Strong view 1 — independent color aug for T1 and T2
        img_s1_t1 = deepcopy(img_t1)
        img_s1_t2 = deepcopy(img_t2)
        if random.random() < 0.8:
            img_s1_t1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1_t1)
        if random.random() < 0.8:
            img_s1_t2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1_t2)
        img_s1_t1 = transforms.RandomGrayscale(p=0.2)(img_s1_t1)
        img_s1_t2 = transforms.RandomGrayscale(p=0.2)(img_s1_t2)
        img_s1_t1 = blur(img_s1_t1, p=0.5)
        img_s1_t2 = blur(img_s1_t2, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1_t1.size[0], p=0.5)

        # Strong view 2
        img_s2_t1 = deepcopy(img_t1)
        img_s2_t2 = deepcopy(img_t2)
        if random.random() < 0.8:
            img_s2_t1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2_t1)
        if random.random() < 0.8:
            img_s2_t2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2_t2)
        img_s2_t1 = transforms.RandomGrayscale(p=0.2)(img_s2_t1)
        img_s2_t2 = transforms.RandomGrayscale(p=0.2)(img_s2_t2)
        img_s2_t1 = blur(img_s2_t1, p=0.5)
        img_s2_t2 = blur(img_s2_t2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2_t1.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask_t1.size[1], mask_t1.size[0]), dtype=np.uint8))
        img_s1_t1, ignore_mask = normalize(img_s1_t1, ignore_mask)
        img_s1_t2 = normalize(img_s1_t2)
        img_s2_t1 = normalize(img_s2_t1)
        img_s2_t2 = normalize(img_s2_t2)

        mask_t1_tensor = torch.from_numpy(np.array(mask_t1)).long()
        mask_t2_tensor = torch.from_numpy(np.array(mask_t2)).long()
        ignore_mask[mask_t1_tensor == 254] = 255
        ignore_mask[mask_t2_tensor == 254] = 255

        return (normalize(img_w_t1), normalize(img_w_t2),
                img_s1_t1, img_s1_t2, img_s2_t1, img_s2_t2,
                ignore_mask, cutmix_box1, cutmix_box2)
```

### 4.2 Siamese Model

```python
# model/semseg/siamese_dpt.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.dinov2 import DINOv2
from model.semseg.dpt import DPTHead


class SiameseDPT(nn.Module):
    """
    Shared-weight Siamese DPT for bi-temporal Semantic Change Detection.
    
    Both T1 and T2 pass through the same DINOv2 encoder and DPT decoder.
    Returns separate segmentation maps for T1 and T2.
    Change map is computed at inference via Post-Classification Comparison (PCC):
        change_map = (argmax(seg_t1) != argmax(seg_t2))
    """
    def __init__(self, encoder_size='base', nclass=6, features=128,
                 out_channels=[96, 192, 384, 768], use_bn=False):
        super().__init__()
        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11], 'base': [2, 5, 8, 11],
            'large': [4, 11, 17, 23], 'giant': [9, 19, 29, 39]
        }
        self.encoder_size = encoder_size
        self.backbone = DINOv2(model_name=encoder_size)
        self.head = DPTHead(nclass, self.backbone.embed_dim, features,
                            use_bn=use_bn, out_channels=out_channels)
        self.binomial = torch.distributions.binomial.Binomial(probs=0.5)

    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _forward_one(self, x, dropout_mask=None):
        patch_h = x.shape[-2] // 14
        patch_w = x.shape[-1] // 14
        features = self.backbone.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder_size]
        )
        if dropout_mask is not None:
            # FIX for BUG 1: use tuple, not generator
            features = tuple(f * dropout_mask.unsqueeze(1) for f in features)
        out = self.head(features, patch_h, patch_w)
        return F.interpolate(out, (patch_h * 14, patch_w * 14),
                             mode='bilinear', align_corners=True)

    def forward(self, img_t1, img_t2, comp_drop=False):
        if comp_drop:
            # img_t1 and img_t2 both have shape [B, 3, H, W]
            # where B contains [stream1_batch, stream2_batch] stacked
            bs = img_t1.shape[0]
            bs_half = bs // 2
            dim = self.backbone.embed_dim

            mask1 = self.binomial.sample((bs_half, dim)).to(img_t1.device) * 2.0
            mask2 = 2.0 - mask1
            dropout_mask = torch.cat([mask1, mask2], dim=0)  # [B, dim]

            seg_t1 = self._forward_one(img_t1, dropout_mask)
            seg_t2 = self._forward_one(img_t2, dropout_mask)
        else:
            seg_t1 = self._forward_one(img_t1)
            seg_t2 = self._forward_one(img_t2)

        return seg_t1, seg_t2

    @staticmethod
    def get_change_map(seg_t1, seg_t2):
        """Post-classification comparison. Call at inference only."""
        label_t1 = seg_t1.argmax(dim=1)
        label_t2 = seg_t2.argmax(dim=1)
        change_map = (label_t1 != label_t2).long()
        return label_t1, label_t2, change_map
```

### 4.3 SECOND Config File

```yaml
# configs/second.yaml
dataset: second
data_root: /your/path/to/SECOND
nclass: 6
crop_size: 512

epochs: 60
batch_size: 2          # per GPU; 4 GPUs = effective batch 8
lr: 0.000005
lr_multi: 40.0
criterion:
  name: HybridDiceCE
  kwargs:
    ignore_index: 255
    dice_weight: 0.5
conf_thresh: 0.95

backbone: dinov2_base
lock_backbone: False
freeze_epochs: 10      # freeze backbone for first 10 epochs, then unfreeze
```

### 4.4 Class Names

Add to `util/classes.py`:

```python
CLASSES['second'] = [
    'non-vegetated ground',   # 0  ~dominant
    'tree',                   # 1
    'low vegetation',         # 2
    'water',                  # 3  ~0.26% of pixels
    'building',               # 4  ~dominant
    'playground',             # 5  ~0.12% of pixels
]
```

### 4.5 Split Generation

```python
# generate_second_splits.py
import os, random

def generate(second_root, output_dir, seed=42, ratios=[5, 10, 20, 40, 100]):
    files = sorted(f for f in os.listdir(os.path.join(second_root, 'train', 'im1'))
                   if f.endswith('.png'))
    all_pairs = [f"train/im1/{f} train/im2/{f} train/label1/{f} train/label2/{f}"
                 for f in files]
    
    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    
    for pct in ratios:
        n = max(10, int(len(all_pairs) * pct / 100))
        labeled   = all_pairs[:n]
        unlabeled = all_pairs[n:] if pct < 100 else all_pairs
        
        d = os.path.join(output_dir, f'{pct}pct')
        os.makedirs(d, exist_ok=True)
        open(f'{d}/labeled.txt',   'w').write('\n'.join(labeled))
        open(f'{d}/unlabeled.txt', 'w').write('\n'.join(unlabeled))
        print(f'{pct}%: {n} labeled pairs, {len(unlabeled)} unlabeled')
    
    # Val split (test set)
    val_files = sorted(f for f in os.listdir(os.path.join(second_root, 'test', 'im1'))
                       if f.endswith('.png'))
    val_pairs = [f"test/im1/{f} test/im2/{f} test/label1/{f} test/label2/{f}"
                 for f in val_files]
    os.makedirs(output_dir, exist_ok=True)
    open(os.path.join(output_dir, 'val.txt'), 'w').write('\n'.join(val_pairs))

if __name__ == '__main__':
    generate('/path/to/SECOND', 'splits/second')
```

---

## 5. Improvements for SECOND — Ordered by Impact

### 5.1 Hybrid Dice + CE Loss (Highest Impact)

SECOND has extreme class imbalance. Water is 0.26% of pixels, playground 0.12%. Plain CE loss produces gradients almost entirely from buildings and ground, leaving rare classes with negligible updates. Dice loss gives every class equal gradient weight.

```python
# util/losses.py
import torch
import torch.nn as nn


class HybridDiceCELoss(nn.Module):
    def __init__(self, ignore_index=255, dice_weight=0.5, num_classes=6):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def dice_loss(self, pred, target):
        pred_soft = pred.softmax(dim=1)
        valid_mask = (target != self.ignore_index)
        dice_sum = 0.0
        valid_count = 0

        for cls in range(self.num_classes):
            gt = ((target == cls) & valid_mask).float()
            if gt.sum() < 1.0:
                continue  # skip absent classes to avoid trivial dice=1
            pr = pred_soft[:, cls] * valid_mask.float()
            inter = (pr * gt).sum()
            dice_sum += 1.0 - (2.0 * inter + 1.0) / (pr.sum() + gt.sum() + 1.0)
            valid_count += 1

        return dice_sum / max(valid_count, 1)

    def forward(self, pred, target):
        return self.ce(pred, target) + self.dice_weight * self.dice_loss(pred, target)
```

Wire it in by adding to the criterion factory in the training loop:
```python
elif cfg['criterion']['name'] == 'HybridDiceCE':
    criterion_l = HybridDiceCELoss(**cfg['criterion']['kwargs']).cuda(local_rank)
```

### 5.2 MRF Spatial Regularization at Inference

Removes salt-and-pepper noise from the change map. Applied to the argmax label maps of T1 and T2 before comparing them via PCC.

```python
# util/mrf.py
import numpy as np


def apply_mrf_potts(seg_prob_np, lambda_smooth=5.0):
    """
    Apply MRF MAP inference with Potts pairwise model.
    seg_prob_np: numpy [nclass, H, W] softmax probabilities
    Returns: smoothed label map [H, W] as int
    
    Requires: pip install pygco --break-system-packages
    """
    try:
        import pygco
    except ImportError:
        # Fallback: median filter smoothing
        from scipy.ndimage import median_filter
        labels = seg_prob_np.argmax(axis=0)
        return median_filter(labels, size=3)

    nclass, H, W = seg_prob_np.shape
    n_pixels = H * W

    # Unary potentials: -log(softmax), scaled to int32
    unary = -np.log(seg_prob_np.reshape(nclass, -1).T + 1e-10)
    unary = np.ascontiguousarray((unary * 10).astype(np.int32))

    # Potts pairwise
    pairwise = ((1 - np.eye(nclass)) * int(lambda_smooth * 10)).astype(np.int32)

    # 4-connected grid edges
    rows = np.arange(n_pixels).reshape(H, W)
    edges_h = np.stack([rows[:, :-1].ravel(), rows[:, 1:].ravel()], axis=1)
    edges_v = np.stack([rows[:-1, :].ravel(), rows[1:, :].ravel()], axis=1)
    edges = np.ascontiguousarray(np.vstack([edges_h, edges_v]).astype(np.int32))
    edge_weights = np.ones(len(edges), dtype=np.int32)

    labels = pygco.cut_general_graph(edges, edge_weights, unary, pairwise,
                                     n_iter=-1, algorithm='expansion')
    return labels.reshape(H, W)
```

Usage in `evaluate_second()`:
```python
from util.mrf import apply_mrf_potts

# After getting seg_t1, seg_t2:
prob_t1 = seg_t1.softmax(1).cpu().numpy()  # [B, nclass, H, W]
prob_t2 = seg_t2.softmax(1).cpu().numpy()

for b in range(B):
    pred_t1[b] = apply_mrf_potts(prob_t1[b], lambda_smooth=5.0)
    pred_t2[b] = apply_mrf_potts(prob_t2[b], lambda_smooth=5.0)
```

### 5.3 Boundary-Aware Loss

```python
def boundary_loss(pred_prob, img_t1, img_t2):
    """
    Penalize misalignment between prediction edges and image intensity edges.
    pred_prob: [B, nclass, H, W] softmax probabilities
    img_t1, img_t2: [B, 3, H, W] input images (normalized)
    """
    def img_gradient(img):
        gray = 0.299*img[:,0] + 0.587*img[:,1] + 0.114*img[:,2]  # [B,H,W]
        grad_h = torch.abs(gray[:, :, 1:] - gray[:, :, :-1])
        grad_v = torch.abs(gray[:, 1:, :] - gray[:, :-1, :])
        return grad_h, grad_v

    def pred_gradient(prob):
        p = prob.max(dim=1)[0]  # [B, H, W]
        grad_h = torch.abs(p[:, :, 1:] - p[:, :, :-1])
        grad_v = torch.abs(p[:, 1:, :] - p[:, :-1, :])
        return grad_h, grad_v

    # Use the element-wise max gradient from T1 and T2 images
    ig_h1, ig_v1 = img_gradient(img_t1)
    ig_h2, ig_v2 = img_gradient(img_t2)
    ig_h = torch.max(ig_h1, ig_h2)
    ig_v = torch.max(ig_v1, ig_v2)

    pg_h, pg_v = pred_gradient(pred_prob)

    return (torch.mean(torch.abs(pg_h - ig_h)) +
            torch.mean(torch.abs(pg_v - ig_v))) / 2.0

# Add to labeled loss: loss_x += beta * boundary_loss(pred_x_t1.softmax(1), img_x_t1, img_x_t2)
# Start with beta = 0.1
```

### 5.4 Per-Class Confidence Thresholding for Rare Classes

Fixed τ=0.95 systematically excludes water and playground pseudo-labels because the model rarely reaches 95% confidence on rare classes. This prevents SSL from helping on the exact classes that need it most.

```python
class AdaptiveThreshold:
    def __init__(self, nclass, base=0.95, min_thresh=0.6, ema=0.99):
        self.thresholds = torch.ones(nclass) * base
        self.min_thresh = min_thresh
        self.ema_decay = ema

    def update(self, conf_map, pred_labels, ignore_mask):
        """Update per-class thresholds based on observed confidence distributions."""
        for cls in range(len(self.thresholds)):
            cls_mask = (pred_labels == cls) & (ignore_mask != 255)
            if cls_mask.sum() > 200:
                mean_conf = conf_map[cls_mask].mean().item()
                # Use 90% of mean confidence as threshold floor for this class
                new_thresh = max(self.min_thresh, mean_conf * 0.90)
                self.thresholds[cls] = (self.ema_decay * self.thresholds[cls] +
                                        (1 - self.ema_decay) * new_thresh)

    def get_mask(self, conf_map, pred_labels):
        thresh_map = self.thresholds[pred_labels.cpu()].to(conf_map.device)
        return conf_map >= thresh_map
```

### 5.5 Freeze-Then-Unfreeze Backbone

With only 5-10% of SECOND labeled (233-466 pairs), fine-tuning the entire DINOv2 backbone immediately can cause catastrophic forgetting of the pre-trained representations. Freeze the backbone for the first 10 epochs while the decoder learns to interpret the features, then unfreeze for joint fine-tuning.

```python
# In training loop:
if epoch == cfg.get('freeze_epochs', 0) and cfg.get('lock_backbone', False):
    for p in model.module.backbone.parameters():
        p.requires_grad = True
    # Rebuild optimizer to include backbone params with correct LR
    optimizer = AdamW([
        {'params': [p for p in model.module.backbone.parameters() if p.requires_grad],
         'lr': cfg['lr']},
        {'params': [p for n, p in model.module.named_parameters() if 'backbone' not in n],
         'lr': cfg['lr'] * cfg['lr_multi']}
    ], lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01)
    if rank == 0:
        logger.info(f'Epoch {epoch}: Backbone unfrozen for fine-tuning')
```

---

## 6. Building Accuracy vs. Training Data Experiments

This is your core analysis: showing how performance scales with labeled data fraction.

### Experiment Matrix

| Labeled % | # Pairs | Supervised | FixMatch | UniMatch V2 | V2+Dice+MRF |
|-----------|---------|------------|----------|-------------|-------------|
| 5% | 233 | ? | ? | ? | ? |
| 10% | 466 | ? | ? | ? | ? |
| 20% | 932 | ? | ? | ? | ? |
| 40% | 1865 | ? | ? | ? | ? |
| 100% | 4662 | ? | — | — | ? |

Track per experiment: **mIoU (T1)**, **mIoU (T2)**, **Building IoU**, **Water IoU**, **Playground IoU**, **Change F1**, **SeK score**.

### SECOND Evaluation Function

```python
# Add to supervised.py or a new evaluate_second.py

SECOND_BUILDING_IDX = 4
SECOND_WATER_IDX = 3
SECOND_PLAYGROUND_IDX = 5


def evaluate_second(model, loader, cfg):
    """Evaluate SiameseDPT on SECOND dataset.
    Returns dict with mIoU, per-class IoU, building precision/recall/F1.
    """
    model.eval()
    from util.utils import intersectionAndUnion, AverageMeter
    import torch.distributed as dist

    meter_t1 = {'inter': AverageMeter(), 'union': AverageMeter()}
    meter_t2 = {'inter': AverageMeter(), 'union': AverageMeter()}
    bldg_tp = bldg_fp = bldg_fn = 0

    with torch.no_grad():
        for img_t1, img_t2, mask_t1, mask_t2, _ in loader:
            img_t1, img_t2 = img_t1.cuda(), img_t2.cuda()

            # Pad to nearest multiple of 14 (DINOv2 patch size)
            ori_h, ori_w = img_t1.shape[-2:]
            new_h = int(ori_h / 14 + 0.5) * 14
            new_w = int(ori_w / 14 + 0.5) * 14
            img_t1 = F.interpolate(img_t1, (new_h, new_w), mode='bilinear', align_corners=True)
            img_t2 = F.interpolate(img_t2, (new_h, new_w), mode='bilinear', align_corners=True)

            seg_t1, seg_t2 = model(img_t1, img_t2)

            seg_t1 = F.interpolate(seg_t1, (ori_h, ori_w), mode='bilinear', align_corners=True)
            seg_t2 = F.interpolate(seg_t2, (ori_h, ori_w), mode='bilinear', align_corners=True)

            pred_t1 = seg_t1.argmax(1)
            pred_t2 = seg_t2.argmax(1)

            # mIoU
            for pred, mask, m in [(pred_t1, mask_t1, meter_t1), (pred_t2, mask_t2, meter_t2)]:
                inter, union, _ = intersectionAndUnion(
                    pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)
                ri = torch.from_numpy(inter).cuda()
                ru = torch.from_numpy(union).cuda()
                dist.all_reduce(ri); dist.all_reduce(ru)
                m['inter'].update(ri.cpu().numpy())
                m['union'].update(ru.cpu().numpy())

            # Building-specific precision/recall
            b_pred = ((pred_t1 == SECOND_BUILDING_IDX) |
                      (pred_t2 == SECOND_BUILDING_IDX)).cpu()
            b_gt   = ((mask_t1 == SECOND_BUILDING_IDX) |
                      (mask_t2 == SECOND_BUILDING_IDX))
            bldg_tp += (b_pred &  b_gt).sum().item()
            bldg_fp += (b_pred & ~b_gt).sum().item()
            bldg_fn += (~b_pred & b_gt).sum().item()

    iou_t1 = meter_t1['inter'].sum / (meter_t1['union'].sum + 1e-10) * 100.0
    iou_t2 = meter_t2['inter'].sum / (meter_t2['union'].sum + 1e-10) * 100.0

    prec = bldg_tp / (bldg_tp + bldg_fp + 1e-10)
    rec  = bldg_tp / (bldg_tp + bldg_fn + 1e-10)
    f1   = 2 * prec * rec / (prec + rec + 1e-10)

    return {
        'mIoU_T1':        float(np.mean(iou_t1)),
        'mIoU_T2':        float(np.mean(iou_t2)),
        'iou_class_T1':   iou_t1,
        'iou_class_T2':   iou_t2,
        'building_IoU_T1': float(iou_t1[SECOND_BUILDING_IDX]),
        'building_IoU_T2': float(iou_t2[SECOND_BUILDING_IDX]),
        'water_IoU':       float((iou_t1[SECOND_WATER_IDX] + iou_t2[SECOND_WATER_IDX]) / 2),
        'playground_IoU':  float((iou_t1[SECOND_PLAYGROUND_IDX] + iou_t2[SECOND_PLAYGROUND_IDX]) / 2),
        'building_F1':     float(f1),
        'building_prec':   float(prec),
        'building_rec':    float(rec),
    }
```

### Run Script

```bash
#!/bin/bash
# run_second_all.sh

ROOT=/path/to/SECOND
SPLITS=splits/second

for PCT in 5 10 20 40; do
  for METHOD in supervised fixmatch unimatch_v2; do
    SAVE=exp/second/${METHOD}/dinov2_base/${PCT}pct
    mkdir -p $SAVE
    python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=$((12300 + RANDOM % 1000)) \
        ${METHOD}_second.py \
        --config configs/second.yaml \
        --labeled-id-path   ${SPLITS}/${PCT}pct/labeled.txt \
        --unlabeled-id-path ${SPLITS}/${PCT}pct/unlabeled.txt \
        --save-path $SAVE \
        2>&1 | tee $SAVE/out.log
  done
done
```

---

## 7. What NOT to Change

The following are implemented correctly and should not be touched:

- **Complementary dropout formula** — binomial mask + complement + ×2 scaling matches the paper exactly. Just fix the generator bug.
- **EMA formula** — `min(1-1/(iter+1), 0.996)` is exactly what the paper specifies.
- **Poly LR decay** — `lr × (1 - iter/total)^0.9` is standard and correct.
- **Checkpoint/resume** — robust implementation, handles all edge cases correctly.
- **CutMix blending of pseudo-labels** — the blending of `mask_u_w_cutmixed` and `conf_u_w_cutmixed` is subtle but correct. Do not simplify.
- **Labeled sampler upsampling** — `nsample=len(trainset_u.ids)` ensures labeled and unlabeled batches are the same length. Correct.
- **dist/all_reduce in evaluation** — correct multi-GPU metric aggregation pattern.

---

## 8. Priority Order (Start Here)

**This hour:**
1. Fix BUG 7 — `nclass=19` hardcode → `cfg['nclass']` (2 min)
2. Fix BUG 5 — `dtype=np.uint8` in ignore_mask (2 min)
3. Fix BUG 1 — `tuple()` wrapper in comp_drop (1 min)
4. Fix BUG 8 — add `clip_grad_norm_` (5 min)
5. Add SECOND class names to `util/classes.py` (5 min)
6. Create `configs/second.yaml` (10 min)

**This week:**
7. Write `generate_second_splits.py` and generate 5/10/20/40% splits
8. Write `dataset/second.py` (SecondDataset with synchronized augmentations)
9. Write `model/semseg/siamese_dpt.py` (SiameseDPT)
10. Write `unimatch_v2_second.py` adapting the training loop
11. Write `evaluate_second()` with building/water/playground tracking
12. Add `HybridDiceCELoss` to `util/losses.py`

**After first results:**
13. Add MRF post-processing
14. Add boundary-aware loss
15. Add per-class adaptive thresholds
16. Run full ablation study
