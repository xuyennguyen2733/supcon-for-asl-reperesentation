"""Visualize augmentations on a specific keypoint sample.

Produces 3 videos: original, view 1, view 2 — with augmentation details
overlaid on the augmented views.

Usage:
    python -m tools.visualize_augmentations --sample data/keypoints/train/hello/12345
    python -m tools.visualize_augmentations --sample data/keypoints/train/book/67890
    python -m tools.visualize_augmentations --label hello   # picks first sample of that label
"""

import os
import argparse
from pathlib import Path

import cv2
import numpy as np

from utils.keypoint_utils import reconstruct_video_from_keypoints, draw_landmarks_from_coordinates
from utils.augmentation_utils import random_augment

import mediapipe as mp
mp_holistic = mp.solutions.holistic


def reconstruct_video_with_text(keypoints, save_dir, filename, text_lines=None,
                                 fps=30.0, height=720, width=1280):
    """Reconstruct skeleton video with optional text overlay in the lower-left corner."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    poses, left_hands, right_hands = keypoints

    num_pose_kps = len(poses[0]) if len(poses) > 0 else 33
    pose_connections = [(a, b) for a, b in mp_holistic.POSE_CONNECTIONS
                        if a < num_pose_kps and b < num_pose_kps]

    for pose, left_hand, right_hand in zip(poses, left_hands, right_hands):
        try:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)

            pose_2d = pose[:, :2] * [width, height]
            left_2d = left_hand[:, :2] * [width, height]
            right_2d = right_hand[:, :2] * [width, height]

            canvas = draw_landmarks_from_coordinates(canvas, pose_2d, pose_connections)
            canvas = draw_landmarks_from_coordinates(canvas, left_2d, mp_holistic.HAND_CONNECTIONS,
                                                      dot_color=(255, 100, 0))
            canvas = draw_landmarks_from_coordinates(canvas, right_2d, mp_holistic.HAND_CONNECTIONS,
                                                      dot_color=(0, 100, 255))

            # Draw text overlay in lower-left corner
            if text_lines:
                line_h = 22
                y_start = height - 15 - (len(text_lines) - 1) * line_h
                for i, line in enumerate(text_lines):
                    y = y_start + i * line_h
                    # Shadow for readability
                    cv2.putText(canvas, line, (12, y + 1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(canvas, line, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)

            out.write(canvas)
        except Exception as e:
            print(f"Error reconstructing frame: {e}")
            break

    out.release()


def find_sample(sample_path=None, label=None):
    """Resolve a sample directory from args."""
    if sample_path:
        if os.path.isdir(sample_path):
            return sample_path
        raise FileNotFoundError(f"Sample directory not found: {sample_path}")

    if label:
        # Search train, then val, then test
        for split in ['train', 'val', 'test']:
            label_dir = os.path.join('data', 'keypoints', split, label)
            if os.path.isdir(label_dir):
                samples = sorted([d for d in os.listdir(label_dir)
                                   if os.path.isdir(os.path.join(label_dir, d))])
                if samples:
                    return os.path.join(label_dir, samples[0])
        raise FileNotFoundError(f"No samples found for label '{label}'")

    raise ValueError("Provide --sample or --label")


def main():
    parser = argparse.ArgumentParser(description='Visualize augmentations on a keypoint sample')
    parser.add_argument('--sample', type=str, default=None,
                        help='Path to a keypoint sample directory (e.g. data/keypoints/train/hello/12345)')
    parser.add_argument('--label', type=str, default=None,
                        help='Label name — picks the first sample of that label')
    args = parser.parse_args()

    sample_dir = find_sample(args.sample, args.label)

    # Extract label and sample name from path
    # Expected: .../keypoints/{split}/{label}/{sample_id}
    parts = Path(sample_dir).parts
    label = parts[-2]
    sample_name = parts[-1]

    print(f"Sample: {sample_dir}")
    print(f"Label:  {label}")

    # Load keypoints
    pose = np.load(os.path.join(sample_dir, 'pose.npy'))
    left_hand = np.load(os.path.join(sample_dir, 'left_hand.npy'))
    right_hand = np.load(os.path.join(sample_dir, 'right_hand.npy'))

    print(f"Original: {pose.shape[0]} frames, pose {pose.shape}, "
          f"left {left_hand.shape}, right {right_hand.shape}")

    output_dir = os.path.join('data', 'augmented_visualization', label, sample_name)

    # 1. Original video
    print(f"\nRendering original...")
    reconstruct_video_with_text(
        [pose, left_hand, right_hand], output_dir, 'original.mp4',
        text_lines=[f'Original ({pose.shape[0]} frames)'])
    print(f"  Saved: {output_dir}/original.mp4")

    # 2. View 1
    aug_p1, aug_l1, aug_r1, applied1 = random_augment(
        pose, left_hand, right_hand, track=True)
    print(f"\nView 1 augmentations:")
    for desc in applied1:
        print(f"  - {desc}")
    text1 = ['View 1:'] + [f'  {d}' for d in applied1] + [f'{aug_p1.shape[0]} frames']
    reconstruct_video_with_text(
        [aug_p1, aug_l1, aug_r1], output_dir, 'view_1.mp4',
        text_lines=text1)
    print(f"  Saved: {output_dir}/view_1.mp4")

    # 3. View 2
    aug_p2, aug_l2, aug_r2, applied2 = random_augment(
        pose, left_hand, right_hand, track=True)
    print(f"\nView 2 augmentations:")
    for desc in applied2:
        print(f"  - {desc}")
    text2 = ['View 2:'] + [f'  {d}' for d in applied2] + [f'{aug_p2.shape[0]} frames']
    reconstruct_video_with_text(
        [aug_p2, aug_l2, aug_r2], output_dir, 'view_2.mp4',
        text_lines=text2)
    print(f"  Saved: {output_dir}/view_2.mp4")

    print(f"\nAll videos saved to: {output_dir}/")


if __name__ == '__main__':
    main()
