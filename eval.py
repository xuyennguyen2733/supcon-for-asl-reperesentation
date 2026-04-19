"""Evaluate trained models on the held-out test set.

Metrics:
    - Top-1 and Top-5 classification accuracy
    - Per-class accuracy
    - Intra/Inter-class distance ratio (embedding quality)
    - t-SNE visualization of learned embeddings

Usage:
    python eval.py --checkpoint experiments/trained_models/4_triplet_supcon_ce/best_model.pt
    python eval.py --checkpoint experiments/trained_models/4_triplet_supcon_ce/best_model.pt --use_rope
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import DataLoader

from train import ASLKeypointDataset, collate_eval
from models.encoder import SignLanguageEncoder


@torch.no_grad()
def collect_predictions_and_embeddings(model, dataloader, device):
    """Run model on all data, collecting predictions, labels, and embeddings."""
    model.eval()
    all_logits = []
    all_labels = []
    all_embeddings = []

    for tokens, mask, labels in dataloader:
        tokens, mask = tokens.to(device), mask.to(device)

        # Get both projections (embeddings) and logits
        projections, logits = model(tokens, mask)

        all_logits.append(logits.cpu())
        all_labels.append(labels)
        all_embeddings.append(projections.cpu())

    return {
        'logits': torch.cat(all_logits),        # (N, num_classes)
        'labels': torch.cat(all_labels),         # (N,)
        'embeddings': torch.cat(all_embeddings), # (N, proj_dim) L2-normalized
    }


def compute_accuracy(logits, labels):
    """Top-1 and Top-5 accuracy."""
    top1 = (logits.argmax(dim=1) == labels).float().mean().item()

    k = min(5, logits.size(1))
    top5_preds = logits.topk(k, dim=1).indices
    top5 = sum(labels[i] in top5_preds[i] for i in range(len(labels))) / len(labels)

    return {'top1': top1, 'top5': top5}


def compute_per_class_accuracy(logits, labels, label_names):
    """Accuracy broken down by class."""
    preds = logits.argmax(dim=1)
    per_class = {}

    for idx, name in enumerate(label_names):
        mask = labels == idx
        if mask.sum() == 0:
            continue
        correct = (preds[mask] == idx).float().mean().item()
        per_class[name] = {'accuracy': correct, 'count': mask.sum().item()}

    return per_class


def compute_distance_ratio(embeddings, labels):
    """Intra/Inter-class cosine distance ratio.

    Intra-class distance: mean pairwise cosine distance between embeddings
    of the same class.
    Inter-class distance: mean pairwise cosine distance between embeddings
    of different classes.

    A lower ratio means tighter same-class clusters with wider separation —
    exactly what SupCon should produce.

    Cosine distance = 1 - cosine_similarity (since embeddings are L2-normalized,
    cosine_similarity = dot product).
    """
    # Cosine similarity matrix (embeddings are already L2-normalized)
    sim = embeddings @ embeddings.T  # (N, N)
    dist = 1.0 - sim                 # cosine distance

    # Masks
    N = len(labels)
    labels_row = labels.unsqueeze(0).expand(N, N)
    labels_col = labels.unsqueeze(1).expand(N, N)
    same_mask = (labels_row == labels_col)
    diag_mask = torch.eye(N, dtype=torch.bool)

    # Intra-class: same label, exclude self-pairs
    intra_mask = same_mask & ~diag_mask
    intra_dist = dist[intra_mask].mean().item() if intra_mask.sum() > 0 else 0.0

    # Inter-class: different labels
    inter_mask = ~same_mask
    inter_dist = dist[inter_mask].mean().item() if inter_mask.sum() > 0 else 1.0

    ratio = intra_dist / inter_dist if inter_dist > 0 else float('inf')

    return {
        'intra_class_distance': intra_dist,
        'inter_class_distance': inter_dist,
        'ratio': ratio,
    }


def generate_tsne(embeddings, labels, label_names, save_path):
    """Generate and save a t-SNE visualization of the embedding space."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Skipping t-SNE (install scikit-learn and matplotlib: "
              "pip install scikit-learn matplotlib)")
        return

    print("  Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    coords = tsne.fit_transform(embeddings.numpy())

    fig, ax = plt.subplots(figsize=(14, 10))

    unique_labels = sorted(set(labels.numpy()))
    cmap = plt.cm.get_cmap('tab20', len(unique_labels))

    for i, label_idx in enumerate(unique_labels):
        mask = labels.numpy() == label_idx
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cmap(i)], s=15, alpha=0.7, label=label_names[label_idx])

    ax.set_title('t-SNE of Learned Embeddings (Test Set)')
    ax.set_xticks([])
    ax.set_yticks([])

    # Legend outside plot if many classes
    if len(unique_labels) > 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                  fontsize=5, ncol=2, markerscale=2)
    else:
        ax.legend(fontsize=6, ncol=2, markerscale=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved t-SNE plot to {save_path}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build label mapping from training set so test indices match
    train_dir = os.path.join('data', 'keypoints', 'train')
    train_label_names = sorted(os.listdir(train_dir))
    train_label_to_idx = {label: idx for idx, label in enumerate(train_label_names)}

    test_dir = os.path.join('data', 'keypoints', 'test')
    test_dataset = ASLKeypointDataset(test_dir, augment=False, label_to_idx=train_label_to_idx)
    print(f"Test set: {len(test_dataset)} samples, {len(test_dataset.labels)} test classes "
          f"(trained on {len(train_label_names)})")

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_eval, num_workers=0)

    # Load checkpoint first to get num_classes from saved weights
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    trained_classes = checkpoint['model_state_dict']['classification_head.bias'].shape[0]

    model = SignLanguageEncoder(num_classes=trained_classes, use_rope=args.use_rope,
                                use_triplet=args.use_triplet).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint: {args.checkpoint} "
          f"(epoch {checkpoint['epoch']}, val top1 {checkpoint['val_top1']:.4f}, "
          f"trained on {trained_classes} classes, testing on {num_classes})")

    # Collect all predictions and embeddings
    data = collect_predictions_and_embeddings(model, test_loader, device)

    # Output directory (same folder as checkpoint)
    out_dir = os.path.dirname(args.checkpoint)

    # 1. Top-1 / Top-5 accuracy
    acc = compute_accuracy(data['logits'], data['labels'])
    print(f"\nTop-1 Accuracy: {acc['top1']:.4f}")
    print(f"Top-5 Accuracy: {acc['top5']:.4f}")

    # 2. Per-class accuracy
    per_class = compute_per_class_accuracy(data['logits'], data['labels'], train_label_names)
    sorted_classes = sorted(per_class.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nPer-class accuracy ({num_classes} classes):")
    print(f"  Worst 5:")
    for name, stats in sorted_classes[:5]:
        print(f"    {name:25s} {stats['accuracy']:.4f} ({stats['count']} samples)")
    print(f"  Best 5:")
    for name, stats in sorted_classes[-5:]:
        print(f"    {name:25s} {stats['accuracy']:.4f} ({stats['count']} samples)")

    # 3. Intra/Inter-class distance ratio
    dist = compute_distance_ratio(data['embeddings'], data['labels'])
    print(f"\nEmbedding quality:")
    print(f"  Intra-class distance: {dist['intra_class_distance']:.4f}")
    print(f"  Inter-class distance: {dist['inter_class_distance']:.4f}")
    print(f"  Ratio (lower=better): {dist['ratio']:.4f}")

    # 4. t-SNE visualization
    tsne_path = os.path.join(out_dir, 'tsne.png')
    generate_tsne(data['embeddings'], data['labels'], train_label_names, tsne_path)

    # Save all metrics to JSON
    results = {
        'checkpoint': args.checkpoint,
        'epoch': checkpoint['epoch'],
        'val_top1': checkpoint['val_top1'],
        'test_top1': acc['top1'],
        'test_top5': acc['top5'],
        'intra_class_distance': dist['intra_class_distance'],
        'inter_class_distance': dist['inter_class_distance'],
        'distance_ratio': dist['ratio'],
        'per_class': per_class,
    }
    results_path = os.path.join(out_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained ASL model on the test set')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to best_model.pt checkpoint')
    parser.add_argument('--use_triplet', action=argparse.BooleanOptionalAction, default=True,
                        help='Must match the architecture used during training')
    parser.add_argument('--use_rope', action='store_true',
                        help='Must match the architecture used during training')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    main(args)
