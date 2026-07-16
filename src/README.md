# KMP-UNet (BUSI training pipeline)

A self-contained, dependency-light re-implementation of **KMP-UNet** — "A parallel
UNet integrating KAN and mamba for medical image segmentation" (Liu, Wu, Xu, Shi
& Zheng, *Scientific Reports*, 2026) — built on top of the training pipeline
from the [KM-UNet repo](https://github.com/anuoluwa-og/Breast-Cancer-App)
(`dataset.py`, `losses.py`, `metrics.py`, `utils.py`, `kan.py`, and the overall
`train.py` structure), so it can be trained directly on the same pre-processed
**BUSI** dataset that repo already ships (`KM_Unet/data/busi/{images,masks/0}`).

## Files

| File | Purpose |
|---|---|
| `archs.py` | **KMP-UNet** architecture (Conv-Block+LDConv, KAN block, PCMB, ESCA) |
| `kan.py` | Spline-based `KANLinear` / `KAN` layers (copied from the source repo, unmodified) |
| `dataset.py` | `Dataset` class expecting `images/*.png` + `masks/0/*.png` (copied, unmodified) |
| `losses.py` | `BCEDiceLoss` (copied, unmodified) |
| `metrics.py` | `iou_score`, `indicators` (IoU/DSC/HD/HD95/SE/SP/precision) (copied, unmodified) |
| `utils.py` | `AverageMeter`, `str2bool` (copied, unmodified) |
| `train.py` | Training loop, adapted for `KMPUNet`, defaults matching the paper's protocol |

## Architecture → paper mapping

`archs.py` implements every module described in the paper's Methods section:

- **Conv-Block** (Fig. 2b): `3x3 conv -> LDConv -> 3x3 conv`. Used for the first/last
  3 encoder/decoder stages (`enc_conv1..3`, `dec_conv1..3`), channels 8/16/32 as in the paper.
  - **LDConv**: implemented as a depthwise conv whose sampling offsets are predicted
    per-pixel and looked up via `grid_sample`. Parameter count grows linearly with the
    number of sampling points, which is what "linear deformable convolution" is about.
    This avoids the CUDA-only deformable-conv ops the original LDConv repo uses.
- **KAN Block** (Fig. 2c): flatten → real spline-based `KANLinear` (from `kan.py`) →
  reshape → depthwise conv → LayerNorm, with a residual connection added for stable
  optimization.
- **PCMB** (Fig. 3): channels are split into `G=4` groups, each group is processed by a
  **CMB** block = forward-Mamba + backward-Mamba + 3x3-conv + residual, fused by a
  Linear→LayerNorm→GELU MLP (Eq. 4), and the four group outputs are concatenated and
  fused again by a second MLP (Eq. 3).
  - **Mamba branch**: a minimal, dependency-free selective state-space scan
    (`MinimalMamba1D`) implemented in pure PyTorch — same recurrence as the official
    Mamba block, just without the fused CUDA kernel (`mamba_ssm`/`causal_conv1d`), so
    it needs no extra system dependencies and runs on CPU or GPU.
- **ESCA** (Fig. 4): parallel spatial-attention branch (multi-scale depthwise 1D convs,
  kernel sizes {3,5,7,9}, over pooled H/W vectors) and channel-attention branch
  (multi-head self-attention on the 1x1 GAP token), combined via a learned adaptive
  gate plus a residual + FFN refinement (Eq. 5–7). Used to refine both skip connections.
- **Parallel PCMB+KAN stages**: per the paper's ablation (Table 6, "parallel > serial")
  and Fig. 8 (2 stacked parallel layers is optimal), each encoder/decoder level runs
  PCMB and KAN **on the same input** and fuses the two outputs with a learned 1x1 conv,
  stacked twice (`ParallelPKBlockPair`).
- **Overall U-shape**: stem (3 Conv-Blocks, /8) → parallel stage @32ch (/8) → parallel
  stage @64ch (/16) → bottleneck @128ch (/32) → mirrored decoder, with **two ESCA-refined
  skip connections** (32ch and 64ch), matching the two ESCA arrows in Fig. 2a.

Everything above is a faithful reading of the paper's equations and figures, with the
simplifications noted above chosen specifically so the whole thing trains with only
`torch` installed — no `mamba_ssm`, `causal_conv1d`, `mmcv`, or custom CUDA kernels.

## Data layout

Point `--data_dir`/`--dataset` at a folder shaped like:

```
<data_dir>/<dataset>/
├── images/
│   ├── benign (1).png
│   ├── malignant (1).png
│   └── ...
└── masks/
    └── 0/
        ├── benign (1).png
        ├── malignant (1).png
        └── ...
```

If you already cloned the KM-UNet repo, this is exactly `KM_Unet/data/busi`, so you can
just run:

```bash
python train.py --data_dir /path/to/KM_Unet/data --dataset busi
```

## Training

```bash
pip install torch albumentations pyyaml pandas scikit-learn tqdm opencv-python
# optional, for full clinical metrics (HD/HD95/precision/specificity/sensitivity):
pip install medpy

python train.py \
    --data_dir /path/to/KM_Unet/data --dataset busi \
    --epochs 400 --batch_size 32 \
    --input_h 256 --input_w 256 \
    --optimizer AdamW --lr 1e-3 --scheduler CosineAnnealingLR \
    --loss BCEDiceLoss
```

This mirrors the paper's implementation details section: 256×256 inputs, random
flip+rotation augmentation, BCE+Dice loss, AdamW with lr 1e-3 and cosine annealing,
batch size 32, 400 epochs. Lower `--batch_size` if you run out of GPU memory (the model
is compact but the pure-PyTorch Mamba scan and LDConv sampling loop are not as
memory/latency-optimized as fused CUDA kernels).

Useful knobs:

- `--base_channels` (default 8): scales the whole network width; this is what
  gets you close to the paper's ~1.0M-parameter budget (default settings here land
  around ~1.4M parameters — increase/decrease to trade capacity for size).
- `--groups` (default 4): number of PCMB channel groups (paper's ablation found G=4 best).
- `--d_state` (default 8): SSM state dimension for the Mamba branches.

Outputs (`model.pth`, `log.csv`, `config.yml`) are written to
`outputs/<dataset>_<arch>/`.

## Verified

The full pipeline (data loading → model forward/backward → loss → metrics →
checkpointing) was smoke-tested end-to-end on a small subset of the real BUSI images
from the source repo before delivery.
