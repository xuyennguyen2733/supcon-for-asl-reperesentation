import os
import argparse
import math
import numpy as np
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader

from utils.augmentation_utils import random_augment
from models.encoder import SignLanguageEncoder
from models.losses import TotalLoss

train_dir = os.path.join('data', 'keypoints', 'train')
test_dir = os.path.join('data', 'keypoints', 'test')
val_dir = os.path.join('data', 'keypoints', 'val')


class ASLKeypointDataset(Dataset):
    def __init__(self, keypoints_dir, augment=True, target_per_class=50, label_to_idx=None):
        """
        Args:
            keypoints_dir: path to data/keypoints/{split}
            augment: if True, returns two augmented views per sample
            target_per_class: desired effective samples per class per epoch.
                When augment=True, each base sample is repeated enough times
                so every class reaches this target. A label with 4 raw samples
                gets repeated ~13x, a label with 12 gets repeated ~5x.
                All augmentations are random and on-the-fly — nothing is saved.
                Set to 0 to disable oversampling (use raw counts).
            label_to_idx: explicit label->index mapping. If None, built from this
                directory. Pass the training set's mapping to val/test sets to
                ensure consistent indices.
        """
        self.augment = augment
        self.labels = sorted(os.listdir(keypoints_dir))
        if label_to_idx is not None:
            self.label_to_idx = label_to_idx
        else:
            self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}

        # Collect base samples per label
        base_samples = []
        for label in self.labels:
            label_dir = os.path.join(keypoints_dir, label)
            for sample_id in sorted(os.listdir(label_dir)):
                sample_dir = os.path.join(label_dir, sample_id)
                if os.path.isdir(sample_dir):
                    base_samples.append((sample_dir, self.label_to_idx[label]))

        # Build oversampled index: repeat under-represented labels
        if augment and target_per_class > 0:
            label_counts = Counter(label_idx for _, label_idx in base_samples)
            # Group samples by label
            label_to_samples = {}
            for sample_dir, label_idx in base_samples:
                label_to_samples.setdefault(label_idx, []).append((sample_dir, label_idx))

            self.samples = []
            for label_idx, samples in label_to_samples.items():
                n = len(samples)
                repeats = math.ceil(target_per_class / n)
                self.samples.extend(samples * repeats)

            raw_total = len(base_samples)
            print(f"  Oversampling: {raw_total} raw -> {len(self.samples)} effective "
                  f"(target {target_per_class}/class, {len(label_counts)} classes)")
        else:
            self.samples = base_samples

    def __len__(self):
        return len(self.samples)

    def _load(self, sample_dir):
        pose = np.load(os.path.join(sample_dir, 'pose.npy'))
        left_hand = np.load(os.path.join(sample_dir, 'left_hand.npy'))
        right_hand = np.load(os.path.join(sample_dir, 'right_hand.npy'))
        return pose, left_hand, right_hand

    def _to_tokens(self, pose, left_hand, right_hand):
        T = pose.shape[0]
        body_flat = pose.reshape(T, -1)
        left_flat = left_hand.reshape(T, -1)
        right_flat = right_hand.reshape(T, -1)

        left_padded = np.pad(left_flat, ((0, 0), (0, 69 - 63)))
        right_padded = np.pad(right_flat, ((0, 0), (0, 69 - 63)))
        tokens = np.stack([body_flat, left_padded, right_padded], axis=1)
        return torch.tensor(tokens, dtype=torch.float32)

    def __getitem__(self, idx):
        sample_dir, label = self.samples[idx]
        pose, left_hand, right_hand = self._load(sample_dir)
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.augment:
            view1 = self._to_tokens(*random_augment(pose, left_hand, right_hand))
            view2 = self._to_tokens(*random_augment(pose, left_hand, right_hand))
            return view1, view2, label_tensor
        else:
            return self._to_tokens(pose, left_hand, right_hand), label_tensor


def _pad_sequences(sequences):
    """Pad a list of (T, 3, 69) tensors to (B, max_T, 3, 69) with a boolean mask."""
    max_T = max(s.shape[0] for s in sequences)
    B = len(sequences)
    padded = torch.zeros(B, max_T, sequences[0].shape[1], sequences[0].shape[2])
    mask = torch.ones(B, max_T, dtype=torch.bool)  # True = padded (ignored by transformer)

    for i, s in enumerate(sequences):
        T = s.shape[0]
        padded[i, :T] = s
        mask[i, :T] = False

    return padded, mask


def collate_augmented(batch):
    """Collate variable-length augmented samples: returns (v1, mask1, v2, mask2, labels)."""
    views1, views2, labels = zip(*batch)
    tokens1, mask1 = _pad_sequences(views1)
    tokens2, mask2 = _pad_sequences(views2)
    return tokens1, mask1, tokens2, mask2, torch.stack(labels)


def collate_eval(batch):
    """Collate variable-length eval samples: returns (tokens, mask, labels)."""
    views, labels = zip(*batch)
    tokens, mask = _pad_sequences(views)
    return tokens, mask, torch.stack(labels)


def train_one_epoch_supcon(model, dataloader, criterion, optimizer, device):
    """Training step with SupCon + CE (two augmented views)."""
    model.train()
    total_loss_sum = 0.0
    supcon_loss_sum = 0.0
    ce_loss_sum = 0.0
    correct = 0
    total = 0

    for tokens1, mask1, tokens2, mask2, labels in dataloader:
        tokens1, mask1 = tokens1.to(device), mask1.to(device)
        tokens2, mask2 = tokens2.to(device), mask2.to(device)
        labels = labels.to(device)

        proj1, logits1 = model(tokens1, mask1)
        proj2, logits2 = model(tokens2, mask2)

        loss, supcon_loss, ce_loss = criterion(proj1, proj2, logits1, logits2, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss_sum += loss.item() * labels.size(0)
        supcon_loss_sum += supcon_loss.item() * labels.size(0)
        ce_loss_sum += ce_loss.item() * labels.size(0)

        avg_logits = (logits1 + logits2) / 2
        correct += (avg_logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    n = total
    return {
        'loss': total_loss_sum / n,
        'supcon_loss': supcon_loss_sum / n,
        'ce_loss': ce_loss_sum / n,
        'acc': correct / n,
    }


def train_one_epoch_ce(model, dataloader, criterion, optimizer, device):
    """Training step with CE only (single augmented view, no contrastive loss)."""
    model.train()
    ce_loss_sum = 0.0
    correct = 0
    total = 0

    for tokens, mask, labels in dataloader:
        tokens, mask = tokens.to(device), mask.to(device)
        labels = labels.to(device)

        _, logits = model(tokens, mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        ce_loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    n = total
    return {
        'loss': ce_loss_sum / n,
        'supcon_loss': 0.0,
        'ce_loss': ce_loss_sum / n,
        'acc': correct / n,
    }


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for tokens, mask, labels in dataloader:
        tokens, mask = tokens.to(device), mask.to(device)
        labels = labels.to(device)

        _, logits = model(tokens, mask)

        correct_top1 += (logits.argmax(dim=1) == labels).sum().item()
        top5_preds = logits.topk(min(5, logits.size(1)), dim=1).indices
        correct_top5 += sum(labels[i] in top5_preds[i] for i in range(labels.size(0)))
        total += labels.size(0)

    return {
        'top1': correct_top1 / total,
        'top5': correct_top5 / total,
    }


def main(args):
    # Reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_supcon = not args.ce_only

    print(f"Using device: {device}")
    print(f"Tokenization: {'Pose-Triplet' if args.use_triplet else 'Flat'}")
    print(f"Loss: {'SupCon + CE' if use_supcon else 'CE only'}")
    if args.use_rope:
        print(f"Positional encoding: RoPE")
    if args.pretrained_path:
        print(f"Pre-trained: {args.pretrained_path}")

    # Datasets — only train and val; test is reserved for eval.py
    # Val must use the same label_to_idx as train for consistent class indices
    train_dataset = ASLKeypointDataset(train_dir, augment=True, target_per_class=args.target_per_class)
    val_dataset = ASLKeypointDataset(val_dir, augment=False, label_to_idx=train_dataset.label_to_idx)

    num_classes = len(train_dataset.label_to_idx)
    print(f"Classes: {num_classes}, Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    if use_supcon:
        train_collate = collate_augmented
    else:
        # CE-only: dataset still returns (view1, view2, label) but we only use view1
        train_collate = lambda batch: collate_eval([(v1, l) for v1, _, l in batch])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=train_collate, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_eval, num_workers=args.num_workers)

    # Model
    model = SignLanguageEncoder(num_classes=num_classes, use_rope=args.use_rope,
                                use_triplet=args.use_triplet).to(device)

    # Load pre-trained encoder weights if provided
    if args.pretrained_path:
        checkpoint = torch.load(args.pretrained_path, weights_only=False)
        model.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
        print(f"Loaded pre-trained encoder from {args.pretrained_path} "
              f"(epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.6f})")

    if use_supcon:
        criterion = TotalLoss(temperature=args.temperature, ce_weight=args.ce_weight)
        train_fn = train_one_epoch_supcon
    else:
        criterion = torch.nn.CrossEntropyLoss()
        train_fn = train_one_epoch_ce

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Linear warmup then cosine decay
    warmup_epochs = args.warmup_epochs
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Checkpointing
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_top1 = 0.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_fn(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"lr {lr:.6f} | "
              f"train loss {train_metrics['loss']:.4f} "
              f"(supcon {train_metrics['supcon_loss']:.4f}, ce {train_metrics['ce_loss']:.4f}) | "
              f"train acc {train_metrics['acc']:.4f} | "
              f"val top1 {val_metrics['top1']:.4f}, top5 {val_metrics['top5']:.4f}")

        # Save best model
        if val_metrics['top1'] > best_val_top1:
            best_val_top1 = val_metrics['top1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_top1': best_val_top1,
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"  -> Saved new best model (val top1: {best_val_top1:.4f})")

    print(f"\nTraining complete. Best val top1: {best_val_top1:.4f}")
    print(f"Run eval.py to evaluate on the test set.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ASL SupCon model')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--ce_weight', type=float, default=0.1)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--target_per_class', type=int, default=50,
                        help='Oversample each class to this many samples per epoch (0 to disable)')
    parser.add_argument('--ce_only', action='store_true', help='CE-only baseline (no contrastive loss)')
    parser.add_argument('--use_triplet', action=argparse.BooleanOptionalAction, default=True,
                        help='Use Pose-Triplet tokenization (default: True, --no-use_triplet for flat)')
    parser.add_argument('--use_rope', action='store_true', help='Use RoPE instead of absolute positional encoding')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pre-trained encoder checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    main(args)
