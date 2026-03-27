import os
from pathlib import Path

import numpy as np

from utils.keypoint_utils import reconstruct_video_from_keypoints
from utils.augmentation_utils import random_augment

keypoints_dir = Path("data", "keypoints")
output_dir = Path("data", "augmented_renders")

for label_dir in sorted(keypoints_dir.iterdir()):
    if not label_dir.is_dir():
        continue
    label = label_dir.name

    for sample_dir in sorted(label_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        sample_name = sample_dir.name

        pose = np.load(str(sample_dir / "pose.npy"))
        left_hand = np.load(str(sample_dir / "left_hand.npy"))
        right_hand = np.load(str(sample_dir / "right_hand.npy"))

        for view_idx in range(1, 3):
            aug_pose, aug_left, aug_right = random_augment(pose, left_hand, right_hand)
            augmented = [aug_pose, aug_left, aug_right]
            filename = f"view_{view_idx}.mp4"
            save_dir = str(output_dir / label / sample_name)
            reconstruct_video_from_keypoints(augmented, save_dir=save_dir, filename=filename)
            print(f"Saved {save_dir}/{filename}")
