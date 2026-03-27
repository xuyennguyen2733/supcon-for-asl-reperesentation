import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.data_utils import form_pose_triplet_units
from utils.augmentation_utils import random_augment
from models.encoder import SignLanguageEncoder

train_dir = os.path.join('data', 'keypoints', 'train')
test_dir = os.path.join('data', 'keypoints', 'test')
val_dir = os.path.join('data', 'keypoints', 'val')


class ASLKeypointDataset(Dataset):
    def __init__(self, keypoints_dir, augment=True):
        self.augment = augment
        self.labels = sorted(os.listdir(keypoints_dir))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.samples = []

        for label in self.labels:
            label_dir = os.path.join(keypoints_dir, label)
            for sample_id in sorted(os.listdir(label_dir)):
                sample_dir = os.path.join(label_dir, sample_id)
                if os.path.isdir(sample_dir):
                    self.samples.append((sample_dir, self.label_to_idx[label]))

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


if __name__ == '__main__':
    dataset = ASLKeypointDataset(train_dir, augment=True)
    print(f"Dataset size: {len(dataset)}")
    print(f"Labels: {dataset.labels}")

    view1, view2, label = dataset[0]
    print(f"View 1 shape: {view1.shape}")
    print(f"View 2 shape: {view2.shape}")
    print(f"Sample label: {label}")

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_augmented)
    for batch_v1, mask1, batch_v2, mask2, batch_labels in dataloader:
        print(f"    Batch view 1: {batch_v1.shape}, mask: {mask1.shape}")
        print(f"    Batch view 2: {batch_v2.shape}, mask: {mask2.shape}")
        print(f"    Batch labels: {batch_labels}")
        break
