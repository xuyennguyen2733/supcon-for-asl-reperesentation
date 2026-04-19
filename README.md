# Stabilizing Sign Language Representation Via Contrastive Invariance Learning On Skeletal Sequences

CS 6958 Advanced Computer Vision - Spring 2026

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

This project uses [WLASL-100](https://github.com/dxli94/WLASL), the 100-class subset of Word-Level American Sign Language.

### Download via Kaggle

1. Create a Kaggle account and generate an API token at https://www.kaggle.com/settings. This downloads a `kaggle.json` file.
2. Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows).
3. Run:

```bash
kaggle datasets download -d risangbaskoro/wlasl-processed -p data/videos_wlasl100 --unzip
```

4. Download `WLASL_v0.3.json` from the [official repo](https://github.com/dxli94/WLASL) and place it in `data/`.

## Tools

All tool scripts must be run from the project root using `python -m` (not `python tools/...`), so that the `utils` and `models` packages resolve correctly.

### Extract keypoints from WLASL-100

```bash
python -m tools.wlasl100_to_keypoints [--num_labels 100]
```

Extracts MediaPipe Holistic keypoints from WLASL-100 videos, normalizes them (crops to upper body), saves as `.npy` files to `data/keypoints/{train,val,test}/{gloss}/{video_id}/`, and renders skeleton visualizations to `data/keypoint_renders/`.

### Extract keypoints from custom videos

```bash
python -m tools.sample_video_to_keypoints
```

Processes custom video samples from `data/samples/videos/{label}/*.mp4`, extracts and normalizes keypoints, saves to `data/samples/keypoints/`, and renders visualizations.

### Visualize augmentations

```bash
python -m tools.visualize_augmentations
```

Loads saved keypoints from `data/keypoints/train/`, applies random augmentations (horizontal flip, speed change, joint noise, spatial scale, temporal crop, joint dropout, rotation) to create two views per sample, and renders them to `data/augmented_renders/` for visual inspection.

## Training

### Pre-training (optional, BERT-style masked pose modeling)

Pre-trains the encoder by masking 15% of frame tokens and learning to reconstruct them (MSE loss). This gives the encoder an understanding of temporal pose dynamics before it sees any labels.

```bash
python -m models.pretrain [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 50 | Pre-training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--mask_ratio` | 0.15 | Fraction of frame tokens to mask |
| `--use_rope` | off | Use RoPE positional encoding |
| `--use_triplet` / `--no-use_triplet` | on | Pose-Triplet vs flat tokenization |
| `--save_dir` | checkpoints | Output directory |

### Fine-tuning / Training

Trains the transformer encoder with the specified loss and tokenization. Each sample is augmented on-the-fly with 7 augmentations (128 possible combinations per view). Under-represented classes are oversampled to ensure balanced training.

```bash
python train.py [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--weight_decay` | 1e-4 | AdamW weight decay |
| `--temperature` | 0.07 | SupCon temperature |
| `--ce_weight` | 0.1 | Weight of CE loss (lambda) |
| `--warmup_epochs` | 10 | LR warmup epochs |
| `--target_per_class` | 50 | Oversample each class to this many samples per epoch (0 to disable) |
| `--ce_only` | off | CE-only baseline (no contrastive loss) |
| `--use_triplet` / `--no-use_triplet` | on | Pose-Triplet vs flat tokenization |
| `--use_rope` | off | Use RoPE instead of absolute positional encoding |
| `--pretrained_path` | None | Path to pre-trained encoder checkpoint |
| `--seed` | 42 | Random seed for reproducibility |
| `--save_dir` | checkpoints | Directory for saving model checkpoints |

## Experiments

The experiments are designed to isolate the contribution of the two core ideas of this project: **Pose-Triplet tokenization** and **Supervised Contrastive Loss (SupCon)**.

### Experiment Design

**Core 2x2 grid** — isolates each contribution independently and combined:

| # | Experiment | Tokenization | Loss | Tests |
|---|---|---|---|---|
| 1 | Flat + CE | Flat | CE | Naive baseline |
| 2 | Flat + SupCon + CE | Flat | SupCon + CE | SupCon contribution alone |
| 3 | Triplet + CE | Pose-Triplet | CE | Triplet contribution alone |
| 4 | Triplet + SupCon + CE | Pose-Triplet | SupCon + CE | **Proposed method** |

**Enhancements** — further improvements on the proposed method:

| # | Experiment | Addition | Tests |
|---|---|---|---|
| 5 | Triplet + RoPE + SupCon + CE | RoPE | Relative vs absolute positional encoding |
| 6 | Triplet + Pre-trained + SupCon + CE | Masked pose pre-training | Self-supervised initialization |
| 7 | Triplet + RoPE + Pre-trained + SupCon + CE | RoPE + pre-training | Full pipeline |

**Key comparisons:**

| Compare | Isolates |
|---|---|
| 1 vs 2 | Does SupCon help with flat tokens? |
| 1 vs 3 | Does Pose-Triplet help with CE only? |
| 1 vs 4 | Combined effect (the full proposal) |
| 2 vs 4 | Triplet's added value on top of SupCon |
| 3 vs 4 | SupCon's added value on top of Triplet |
| 4 vs 5, 6, 7 | RoPE / pre-training further improvements |

### Running Individual Experiments

```bash
# 1. Flat + CE (naive baseline)
python train.py --no-use_triplet --ce_only --save_dir experiments/trained_models/1_flat_ce

# 2. Flat + SupCon + CE
python train.py --no-use_triplet --save_dir experiments/trained_models/2_flat_supcon_ce

# 3. Triplet + CE
python train.py --ce_only --save_dir experiments/trained_models/3_triplet_ce

# 4. Triplet + SupCon + CE (proposed method)
python train.py --save_dir experiments/trained_models/4_triplet_supcon_ce

# 5. Triplet + RoPE + SupCon + CE
python train.py --use_rope --save_dir experiments/trained_models/5_triplet_rope_supcon_ce

# 6. Triplet + Pre-trained + SupCon + CE
python -m models.pretrain --epochs 50 --save_dir experiments/trained_models/6_triplet_pt_supcon_ce
python train.py --pretrained_path experiments/trained_models/6_triplet_pt_supcon_ce/pretrained_encoder.pt \
    --save_dir experiments/trained_models/6_triplet_pt_supcon_ce

# 7. Triplet + RoPE + Pre-trained + SupCon + CE (full pipeline)
python -m models.pretrain --epochs 50 --use_rope --save_dir experiments/trained_models/7_triplet_rope_pt_supcon_ce
python train.py --use_rope \
    --pretrained_path experiments/trained_models/7_triplet_rope_pt_supcon_ce/pretrained_encoder.pt \
    --save_dir experiments/trained_models/7_triplet_rope_pt_supcon_ce
```

### Running All Experiments with run.py

`run.py` orchestrates all 7 experiments automatically. It detects available GPUs and parallelizes across them. With 1 GPU, experiments run sequentially in priority order (core 2x2 first, then enhancements).

```bash
# Run all 7 experiments
python run.py

# Preview commands without running
python run.py --dry_run

# Run specific experiments by number
python run.py --only 1 4        # just baseline and proposed
python run.py --only 1 2 3 4    # just the core 2x2 grid

# Override epoch counts
python run.py --epochs 50                    # shorter training
python run.py --epochs 100 --pretrain_epochs 80  # more pre-training
```

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 100 | Training epochs per experiment |
| `--pretrain_epochs` | 50 | Pre-training epochs (experiments 6, 7) |
| `--dry_run` | off | Print commands without executing |
| `--only` | all | Run specific experiments by number (e.g. `--only 1 4 5`) |

**Multi-GPU behavior:** With N GPUs detected, up to N experiments run concurrently, each pinned to a separate GPU via `CUDA_VISIBLE_DEVICES`. Pre-training steps (experiments 6, 7) run automatically before their training step within the same job.

**Output structure:**

```
experiments/trained_models/
├── 1_flat_ce/
│   ├── best_model.pt       # best checkpoint by val top-1 accuracy
│   └── train.log           # full training output
├── 2_flat_supcon_ce/
│   ├── best_model.pt
│   └── train.log
├── ...
├── 6_triplet_pt_supcon_ce/
│   ├── pretrained_encoder.pt   # pre-trained weights
│   ├── best_model.pt
│   └── train.log
└── 7_triplet_rope_pt_supcon_ce/
    ├── pretrained_encoder.pt
    ├── best_model.pt
    └── train.log
```
