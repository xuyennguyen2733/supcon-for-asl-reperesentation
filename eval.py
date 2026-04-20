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
    labels_np = labels.numpy()

    # Count samples per class, pick top 10 for colored display
    from collections import Counter
    label_counts = Counter(labels_np.tolist())
    top_classes = [idx for idx, _ in sorted(label_counts.items(), key=lambda x: -x[1])[:10]]

    # 10 distinct colors that are far from dark grey
    highlight_colors = [
        '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
        '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#dcbeff',
    ]
    color_map = {idx: highlight_colors[i] for i, idx in enumerate(top_classes)}
    grey = '#555555'

    # Draw grey (other) points first so highlighted ones are on top
    other_mask = np.array([l not in color_map for l in labels_np])
    if other_mask.any():
        ax.scatter(coords[other_mask, 0], coords[other_mask, 1],
                   c=grey, s=10, alpha=0.3, label='other classes')

    for label_idx in top_classes:
        mask = labels_np == label_idx
        name = label_names[label_idx] if label_idx < len(label_names) else str(label_idx)
        count = label_counts[label_idx]
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color_map[label_idx], s=25, alpha=0.8,
                   label=f'{name} ({count})')

    ax.set_title('t-SNE of Learned Embeddings (Test Set)')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=8, markerscale=2, loc='best')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved t-SNE plot to {save_path}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load label mapping — try saved file next to checkpoint, fall back to train dir
    label_map_path = os.path.join(os.path.dirname(args.checkpoint), 'label_to_idx.json')
    if os.path.isfile(label_map_path):
        with open(label_map_path) as f:
            train_label_to_idx = json.load(f)
        train_label_names = sorted(train_label_to_idx, key=train_label_to_idx.get)
        print(f"Loaded label mapping from {label_map_path}")
    else:
        train_dir = os.path.join('data', 'keypoints', 'train')
        train_label_names = sorted(os.listdir(train_dir))
        train_label_to_idx = {label: idx for idx, label in enumerate(train_label_names)}
        print(f"Label mapping from training directory ({len(train_label_names)} classes)")

    test_dir = os.path.join('data', 'keypoints', 'test')
    test_dataset = ASLKeypointDataset(test_dir, augment=False, label_to_idx=train_label_to_idx)
    print(f"Test set: {len(test_dataset)} samples, {len(test_dataset.labels)} test classes "
          f"(trained on {len(train_label_names)})")

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_eval, num_workers=0)

    # Load checkpoint and infer architecture from saved weights
    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location=device)
    state = checkpoint['model_state_dict']
    trained_classes = state['classification_head.bias'].shape[0]

    layer_indices = set()
    for key in state:
        if key.startswith('transformer.'):
            for p in key.split('.')[1:]:
                if p.isdigit():
                    layer_indices.add(int(p))
                    break
    num_layers = max(layer_indices) + 1 if layer_indices else 2

    ffn_key = [k for k in state if 'ffn.0.weight' in k or 'linear1.weight' in k]
    dim_feedforward = state[ffn_key[0]].shape[0] if ffn_key else 384

    model = SignLanguageEncoder(num_classes=trained_classes, use_rope=args.use_rope,
                                use_triplet=args.use_triplet, num_layers=num_layers,
                                dim_feedforward=dim_feedforward).to(device)
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
