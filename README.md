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

All tool scripts are run from the project root using `python -m`:

### `python -m tools.video_to_keypoints`

Extracts MediaPipe Holistic keypoints from videos in `data/videos/`, normalizes them (crops to upper body), saves as `.npy` files to `data/keypoints/`, and renders skeleton visualizations to `data/keypoint_renders/`.

### `python -m tools.visualize_augmentations`

Loads saved keypoints from `data/keypoints/`, applies random augmentations (horizontal flip, speed change) to create two views per sample, and renders them to `data/augmented_renders/` for visual inspection.

## Training

```bash
python train.py
```

Loads normalized keypoints, applies augmentations to produce two views per sample (for supervised contrastive learning), and trains the transformer encoder with SupCon + cross-entropy loss.
