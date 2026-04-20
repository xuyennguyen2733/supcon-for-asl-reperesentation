"""Generate confusion matrices from predictions.csv for each evaluated model.

By default, scans experiments/evaluation/ and writes a confusion_matrix.png
into each child folder that contains a predictions.csv.

Usage:
    python -m tools.confusion_matrix
    python -m tools.confusion_matrix --eval_dir experiments/evaluation
    python -m tools.confusion_matrix --only 1_flat_ce 4_triplet_supcon_ce
"""
import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def load_predictions(csv_path):
    """Read predictions.csv and return (true_names, pred_names, all_labels)."""
    true_names = []
    pred_names = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_names.append(row['true_label_name'])
            pred_names.append(row['predicted_label_name'])
    all_labels = sorted(set(true_names) | set(pred_names))
    return true_names, pred_names, all_labels


def build_confusion_matrix(true_names, pred_names, all_labels):
    idx = {label: i for i, label in enumerate(all_labels)}
    n = len(all_labels)
    cm = np.zeros((n, n), dtype=np.int32)
    for t, p in zip(true_names, pred_names):
        cm[idx[t], idx[p]] += 1
    return cm


def plot_confusion_matrix(cm, all_labels, title, save_path, normalize=True):
    n = len(all_labels)
    data = cm.astype(np.float32)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums > 0)

    # Scale figure size so labels don't overlap too much
    size = max(8, min(40, 0.22 * n))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(data, cmap='Blues', vmin=0.0, vmax=1.0 if normalize else data.max())

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    # Only draw every-Nth label if there are many classes, to keep the plot readable
    tick_step = max(1, n // 50)
    xtick_labels = [label if i % tick_step == 0 else '' for i, label in enumerate(all_labels)]
    ytick_labels = [label if i % tick_step == 0 else '' for i, label in enumerate(all_labels)]
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize=6)
    ax.set_yticklabels(ytick_labels, fontsize=6)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_for_folder(folder, normalize=True):
    csv_path = os.path.join(folder, 'predictions.csv')
    if not os.path.isfile(csv_path):
        return None

    true_names, pred_names, all_labels = load_predictions(csv_path)
    cm = build_confusion_matrix(true_names, pred_names, all_labels)

    tag = os.path.basename(folder)
    suffix = 'normalized' if normalize else 'raw'
    title = f'{tag} — confusion matrix ({suffix})'
    save_path = os.path.join(folder, f'confusion_matrix.png')
    plot_confusion_matrix(cm, all_labels, title, save_path, normalize=normalize)
    return save_path


def main():
    parser = argparse.ArgumentParser(description='Generate confusion matrices from predictions.csv files')
    parser.add_argument('--eval_dir', type=str, default=os.path.join('experiments', 'evaluation'),
                        help='Directory containing one sub-folder per evaluated model')
    parser.add_argument('--only', type=str, nargs='+', default=None,
                        help='Only process these sub-folder names')
    parser.add_argument('--no_normalize', action='store_true',
                        help='Use raw counts instead of row-normalized values')
    args = parser.parse_args()

    if not os.path.isdir(args.eval_dir):
        print(f"Directory not found: {args.eval_dir}")
        return

    children = sorted(
        name for name in os.listdir(args.eval_dir)
        if os.path.isdir(os.path.join(args.eval_dir, name))
    )
    if args.only:
        selected = set(args.only)
        children = [c for c in children if c in selected]

    if not children:
        print(f"No evaluation sub-folders found in {args.eval_dir}")
        return

    print(f"Generating confusion matrices for {len(children)} model(s)...")
    for child in children:
        folder = os.path.join(args.eval_dir, child)
        out = generate_for_folder(folder, normalize=not args.no_normalize)
        if out:
            print(f"  [{child}] -> {out}")
        else:
            print(f"  [{child}] skipped (no predictions.csv)")


if __name__ == '__main__':
    main()
