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
# By sample path
python -m tools.visualize_augmentations --sample data/keypoints/train/hello/12345

# By label (picks first sample)
python -m tools.visualize_augmentations --label book
```

Produces 3 videos in `data/augmented_visualization/{label}/{sample_id}/`:
- `original.mp4` -- unmodified skeleton
- `view_1.mp4` -- augmented view with overlay listing applied augmentations and parameters
- `view_2.mp4` -- independently augmented view with its own overlay

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
| `--resume` | off | Resume training from `last_checkpoint.pt` in save_dir |
| `--patience` | 20 | Early stopping: stop after N epochs without val improvement (0 to disable) |
| `--seed` | 42 | Random seed for reproducibility |
| `--save_dir` | checkpoints | Directory for saving model checkpoints |

**Resumable training:** A `last_checkpoint.pt` is saved every epoch with full training state (model, optimizer, scheduler, epoch, early stopping counter). Use `--resume` to pick up where training left off after interruptions or crashes.

**Early stopping:** Training stops if val top-1 accuracy doesn't improve for `--patience` consecutive epochs (default 20). Saves GPU time when the model has plateaued.

**Label consistency:** The training label-to-index mapping is saved as `label_to_idx.json` alongside the model. This ensures val/test evaluation uses the same class indices, even if the val/test sets have fewer classes than training.

## Experiments

The experiments are designed to isolate the contribution of the two core ideas of this project: **Pose-Triplet tokenization** and **Supervised Contrastive Loss (SupCon)**.

### Experiment Design

**Core 2x2 grid** -- isolates each contribution independently and combined:

| # | Experiment | Tokenization | Loss | Tests |
|---|---|---|---|---|
| 1 | Flat + CE | Flat | CE | Naive baseline |
| 2 | Flat + SupCon + CE | Flat | SupCon + CE | SupCon contribution alone |
| 3 | Triplet + CE | Pose-Triplet | CE | Triplet contribution alone |
| 4 | Triplet + SupCon + CE | Pose-Triplet | SupCon + CE | **Proposed method** |

**Enhancements** -- further improvements on the proposed method:

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

`run.py` orchestrates all 7 experiments automatically. It detects available GPUs and parallelizes across them. Progress is printed to the console (every 10th epoch, new best models). All training jobs include `--resume` so re-running after interruption picks up where each experiment left off.

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

# Run multiple experiments per GPU (when GPU utilization is low)
python run.py --jobs_per_gpu 2

# Run in tmux (survives SSH disconnects, auto-stops RunPod when done)
python run.py --tmux training
python run.py --tmux training --jobs_per_gpu 2
```

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 100 | Training epochs per experiment |
| `--pretrain_epochs` | 50 | Pre-training epochs (experiments 6, 7) |
| `--dry_run` | off | Print commands without executing |
| `--only` | all | Run specific experiments by number (e.g. `--only 1 4 5`) |
| `--jobs_per_gpu` | 1 | Concurrent experiments per GPU |
| `--tmux` | off | Run inside a named tmux session |

**Multi-GPU behavior:** With N GPUs and `--jobs_per_gpu M`, up to N*M experiments run concurrently. Each experiment is pinned to a GPU via `CUDA_VISIBLE_DEVICES`.

**tmux + RunPod behavior:** When `--tmux` is set, the script launches in a detached tmux session that survives SSH disconnects. After training completes:
- **Success:** syncs `experiments/trained_models/` to Google Drive via rclone, then prompts to stop the pod
- **Crash:** skips sync (output may be corrupted), prompts to stop the pod
- **User interrupt (Ctrl+C):** does nothing, pod keeps running

If no response to the stop prompt within 2 minutes, the pod auto-stops to prevent charges.

**Output structure:**

```
experiments/trained_models/
├── 1_flat_ce/
│   ├── best_model.pt           # best checkpoint by val top-1 accuracy
│   ├── last_checkpoint.pt      # resumable training state (every epoch)
│   ├── label_to_idx.json       # class index mapping (for eval)
│   └── train.log               # full training output
├── ...
├── 6_triplet_pt_supcon_ce/
│   ├── pretrained_encoder.pt   # pre-trained weights
│   ├── best_model.pt
│   ├── last_checkpoint.pt
│   ├── label_to_idx.json
│   └── train.log
└── 7_triplet_rope_pt_supcon_ce/
    └── ...
```

## Evaluation

### Evaluating a Single Model

`eval.py` loads the label mapping from `label_to_idx.json` (saved during training) to ensure consistent class indices. Falls back to the training directory if the file is missing.

```bash
python eval.py --checkpoint experiments/trained_models/4_triplet_supcon_ce/best_model.pt
```

Architecture flags must match what was used during training:

```bash
# Flat model
python eval.py --checkpoint experiments/trained_models/1_flat_ce/best_model.pt --no-use_triplet

# RoPE model
python eval.py --checkpoint experiments/trained_models/5_triplet_rope_supcon_ce/best_model.pt --use_rope
```

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to `best_model.pt` |
| `--use_triplet` / `--no-use_triplet` | on | Must match training architecture |
| `--use_rope` | off | Must match training architecture |
| `--batch_size` | 32 | Batch size for evaluation |

**Metrics computed:**
- **Top-1 and Top-5 accuracy** on the held-out test set
- **Per-class accuracy** breakdown
- **Intra/Inter-class distance ratio** -- measures embedding cluster quality (lower = tighter clusters, wider separation)
- **t-SNE visualization** -- 2D projection of learned embeddings colored by class

### Evaluating All Models with run_eval.py

`run_eval.py` evaluates all trained models and produces comparison reports. Architecture flags and label mappings are automatically inferred from each experiment folder.

```bash
# Evaluate all models in experiments/trained_models/
python run_eval.py

# Evaluate specific checkpoints
python run_eval.py --models path/to/model1/best_model.pt path/to/model2/best_model.pt

# Preview what would be evaluated
python run_eval.py --dry_run

# Run in tmux with pod auto-stop
python run_eval.py --tmux eval_session
```

| Flag | Default | Description |
|---|---|---|
| `--models` | auto-discover | Explicit list of checkpoint paths |
| `--batch_size` | 32 | Batch size |
| `--dry_run` | off | List models without evaluating |
| `--tmux` | off | Run in tmux session with pod auto-stop |

After evaluation completes successfully, results are synced to Google Drive via rclone before prompting to stop the pod.

**Per-model outputs** (in `experiments/evaluation/{model_name}/`):

| File | Format | Contents |
|---|---|---|
| `eval_results.json` | JSON | All metrics + per-class accuracy |
| `predictions.csv` | CSV | Per-sample: true label, predicted label, correct, confidence, top-5 |
| `tsne.png` | PNG | t-SNE embedding visualization |

**Combined outputs** (in `experiments/evaluation/`):

| File | Format | Contents |
|---|---|---|
| `eval_summary.csv` | CSV | One row per model -- top-1, top-5, distance ratio (for graphing) |
| `eval_report.md` | Markdown | Comparison tables, key deltas, per-class breakdown of best model |

## Live Demo

Real-time ASL sign recognition from webcam with skeleton overlay. Supports loading multiple models and comparing their predictions side by side with inference timing.

```bash
# Skeleton visualization only (no model needed)
python demo.py

# Single model
python demo.py --checkpoint experiments/trained_models/4_triplet_supcon_ce/best_model.pt

# Compare all trained models side by side
python demo.py --checkpoints experiments/trained_models/*/best_model.pt

# Compare specific models
python demo.py --checkpoints experiments/trained_models/1_flat_ce/best_model.pt \
                              experiments/trained_models/4_triplet_supcon_ce/best_model.pt

# Smoke test: preview the UI with fake models (no trained models needed)
python demo.py --smoke_test        # 7 fake models
python demo.py --smoke_test 3      # 3 fake models
```

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | None | Path to a single model checkpoint |
| `--checkpoints` | None | Paths to multiple checkpoints (compared side by side) |
| `--camera` | 0 | Webcam index |
| `--smoke_test` | off | Run with N fake models to test the UI (default: 7) |

If no checkpoint is provided, or a checkpoint path doesn't exist, the demo still runs with skeleton overlay and the recording state machine. Architecture flags and label mappings are auto-detected from `label_to_idx.json` in each model's folder.

**How it works:**

The demo uses a state machine for frame accumulation:

- **IDLE** -- no hands visible, waiting. Shows "Show your hands to start".
- **RECORDING** -- hands detected, accumulating frames. Shows frame count + pulsing red dot. Tolerates up to 5 consecutive frames without hands (MediaPipe detection dropout). Stops at 60 frames or when hands disappear for >5 frames.
- **SHOWING** -- predictions displayed in the lower-right panel with confidence and inference time. Predictions persist on screen until replaced by new ones or cleared with `C`. Frames continue accumulating during this state for seamless back-to-back recognition.

Sequences shorter than 30 frames (~1 second) are discarded. Each prediction set is also logged to the console with timestamps for review after closing. Close with `Q`, the window X button, or `C` to clear predictions and buffer.

## Project Structure

```
supcon-for-asl-reperesentation/
├── train.py                    # Training loop (train + val, resumable, early stopping)
├── eval.py                     # Single-model evaluation on test set
├── demo.py                     # Real-time webcam demo (multi-model comparison)
├── run.py                      # Orchestrate all training experiments
├── run_eval.py                 # Orchestrate all evaluations + comparison reports
├── requirements.txt
├── README.md
├── models/
│   ├── encoder.py              # SignLanguageEncoder (flat/triplet, abs pos/RoPE)
│   ├── losses.py               # SupConLoss, TotalLoss
│   └── pretrain.py             # BERT-style masked pose pre-training
├── utils/
│   ├── augmentation_utils.py   # 7 augmentations with optional tracking
│   ├── data_utils.py           # Temporal resampling utilities
│   └── keypoint_utils.py       # MediaPipe extraction, normalization, visualization
├── tools/
│   ├── wlasl100_to_keypoints.py
│   ├── sample_video_to_keypoints.py
│   └── visualize_augmentations.py  # Augmentation visualization with overlay text
├── data/                       # Dataset files (not committed)
└── experiments/                # Experiment outputs (not committed)
    ├── trained_models/         # One folder per experiment (models, logs, mappings)
    ├── evaluation/             # Eval outputs (per-model + combined reports)
    └── class_distribution.csv  # Per-class sample counts
```
