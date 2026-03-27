import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.data_utils import form_pose_triplet_units, temporal_resample
from models.encoder import SignLanguageEncoder

KEYPOINTS_DIR = os.path.join('data', 'keypoints')
TARGET_FRAMES = 32


class ASLKeypointDataset(Dataset):
    def __init__(self, keypoints_dir, target_frames=TARGET_FRAMES):
        self.target_frames = target_frames
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

    def __getitem__(self, idx):
        sample_dir, label = self.samples[idx]
        pose = np.load(os.path.join(sample_dir, 'pose.npy'))
        left_hand = np.load(os.path.join(sample_dir, 'left_hand.npy'))
        right_hand = np.load(os.path.join(sample_dir, 'right_hand.npy'))

        pose = temporal_resample(pose, self.target_frames)
        left_hand = temporal_resample(left_hand, self.target_frames)
        right_hand = temporal_resample(right_hand, self.target_frames)
        
        T = pose.shape[0]
        body_flattened = pose.reshape(T, -1)         # (T, 99)
        left_flattened = left_hand.reshape(T, -1)    # (T, 63)
        right_flattened = right_hand.reshape(T, -1)  # (T, 63)

        # (T, 3, max_keypoints*3) — pad to uniform dim so they can stack
        # body=69, left=63, right=63 -> pad left/right to 69
        left_padded = np.pad(left_flattened, ((0, 0), (0, 69 - 63)))
        right_padded = np.pad(right_flattened, ((0, 0), (0, 69 - 63)))
        tokens = np.stack([body_flattened, left_padded, right_padded], axis=1)

        return torch.tensor(tokens, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


if __name__ == '__main__':
    dataset = ASLKeypointDataset(KEYPOINTS_DIR)
    print(f"Dataset size: {len(dataset)}")
    print(f"Labels: {dataset.labels}")

    tokens, label = dataset[0]
    print(f"Sample tokens shape: {tokens.shape}")
    print(f"Sample label: {label}")

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for batch_tokens, batch_labels in dataloader:
        print(f"    Batch tokens shape: {batch_tokens.shape}")
        print(f"    Batch labels: {batch_labels}")
        break
