import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.data_utils import form_pose_triplet_units
from utils.augmentation_utils import random_augment
from models.encoder import SignLanguageEncoder

KEYPOINTS_DIR = os.path.join('data', 'keypoints')


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


if __name__ == '__main__':
    dataset = ASLKeypointDataset(KEYPOINTS_DIR, augment=True)
    print(f"Dataset size: {len(dataset)}")
    print(f"Labels: {dataset.labels}")

    view1, view2, label = dataset[0]
    print(f"View 1 shape: {view1.shape}")
    print(f"View 2 shape: {view2.shape}")
    print(f"Sample label: {label}")

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for batch_v1, batch_v2, batch_labels in dataloader:
        print(f"    Batch view 1 shape: {batch_v1.shape}")
        print(f"    Batch view 2 shape: {batch_v2.shape}")
        print(f"    Batch labels: {batch_labels}")
        break
