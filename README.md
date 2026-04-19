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

Loads saved keypoints from `data/keypoints/train/`, applies random augmentations (horizontal flip, speed change) to create two views per sample, and renders them to `data/augmented_renders/` for visual inspection.

## Training

### 1. Pre-training (optional, BERT-style masked pose modeling)

Pre-trains the encoder by masking 15% of frame tokens and learning to reconstruct them (MSE loss). This gives the encoder an understanding of temporal pose dynamics before it sees any labels.

```bash
# With absolute positional encoding
python -m models.pretrain --epochs 50 --save_dir checkpoints/pretrained

# With RoPE (Rotary Position Embeddings)
python -m models.pretrain --epochs 50 --use_rope --save_dir checkpoints/pretrained_rope
```

### 2. Fine-tuning / Training

Trains the transformer encoder with supervised contrastive loss + cross-entropy loss. Each sample is augmented twice to produce two views for contrastive learning.

```bash
python train.py [OPTIONS]
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--weight_decay` | 1e-4 | AdamW weight decay |
| `--temperature` | 0.07 | SupCon temperature |
| `--ce_weight` | 0.1 | Weight of CE loss (lambda) |
| `--warmup_epochs` | 10 | LR warmup epochs |
| `--use_rope` | off | Use RoPE instead of absolute positional encoding |
| `--pretrained_path` | None | Path to pre-trained encoder checkpoint |
| `--save_dir` | checkpoints | Directory for saving model checkpoints |

### Experiment Configurations

```bash
# Baseline: SupCon + CE
python train.py --save_dir checkpoints/supcon_ce

# Pre-trained + SupCon + CE
python -m models.pretrain --epochs 50 --save_dir checkpoints/pretrained
python train.py --pretrained_path checkpoints/pretrained/pretrained_encoder.pt --save_dir checkpoints/pretrained_supcon_ce

# RoPE + SupCon + CE
python train.py --use_rope --save_dir checkpoints/rope_supcon_ce

# Pre-trained + RoPE + SupCon + CE
python -m models.pretrain --epochs 50 --use_rope --save_dir checkpoints/pretrained_rope
python train.py --use_rope --pretrained_path checkpoints/pretrained_rope/pretrained_encoder.pt --save_dir checkpoints/pretrained_rope_supcon_ce
```
