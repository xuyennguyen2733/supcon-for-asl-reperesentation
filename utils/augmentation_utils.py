import numpy as np


def random_augment(pose, left_hand, right_hand, rng=None, p=0.5):
    """Apply each augmentation independently with probability p.

    With 7 augmentations at p=0.5, there are 2^7 = 128 possible combinations,
    so two views of the same sample will almost always differ meaningfully.

    Args:
        pose: (T, K, 3)
        left_hand: (T, K, 3)
        right_hand: (T, K, 3)
        rng: np.random.Generator (optional, for reproducibility)
        p: probability of applying each augmentation

    Returns:
        Augmented (pose, left_hand, right_hand).
    """
    if rng is None:
        rng = np.random.default_rng()

    augmentations = [
        flip_horizontal,
        random_speed_change,
        joint_noise,
        spatial_scale,
        temporal_crop,
        joint_dropout,
        random_rotation,
    ]

    for aug in augmentations:
        if rng.random() < p:
            pose, left_hand, right_hand = aug(pose, left_hand, right_hand, rng)

    return pose, left_hand, right_hand


def flip_horizontal(pose, left_hand, right_hand, rng=None):
    """Flip x-coordinates and swap left/right hands."""
    def flip_x(arr):
        flipped = arr.copy()
        flipped[:, :, 0] = 1 - flipped[:, :, 0]
        return flipped

    return flip_x(pose), flip_x(right_hand), flip_x(left_hand)


def random_speed_change(pose, left_hand, right_hand, rng=None, fps=25):
    """Simulate speed variation by resampling to a different number of frames.

    speed_factor > 1: fewer output frames (motion plays faster, up to 4x)
    speed_factor < 1: more output frames (motion plays slower, down to 0.25x)
    Output is clamped to [1s, 4s] duration at the given fps.
    """
    if rng is None:
        rng = np.random.default_rng()

    T = pose.shape[0]
    speed_factor = rng.uniform(0.25, 4.0)
    out_T = int(T / speed_factor)

    # Clamp to [1s, 4s] at the given fps
    min_frames = fps * 1    # 1 second
    max_frames = fps * 4    # 4 seconds
    out_T = max(min_frames, min(out_T, max_frames))

    # Sample out_T evenly-spaced points from the original T frames
    source_indices = np.linspace(0, T - 1, out_T)
    lower = np.floor(source_indices).astype(int)
    upper = np.minimum(lower + 1, T - 1)
    alpha = (source_indices - lower)[:, None, None]

    def resample(arr):
        return arr[lower] * (1 - alpha) + arr[upper] * alpha

    return resample(pose), resample(left_hand), resample(right_hand)


def joint_noise(pose, left_hand, right_hand, rng=None, std=0.005):
    """Add small Gaussian noise to keypoint coordinates.

    Simulates MediaPipe detection jitter and natural signing imprecision.
    std=0.005 is ~0.5% of the normalized [0,1] coordinate range.
    """
    if rng is None:
        rng = np.random.default_rng()

    def add_noise(arr):
        return arr + rng.normal(0, std, size=arr.shape).astype(arr.dtype)

    return add_noise(pose), add_noise(left_hand), add_noise(right_hand)


def spatial_scale(pose, left_hand, right_hand, rng=None, scale_range=(0.8, 1.2)):
    """Randomly scale the skeleton around its centroid.

    Simulates signers at different distances from the camera.
    Scaling is uniform across all body parts to preserve proportions.
    """
    if rng is None:
        rng = np.random.default_rng()

    scale = rng.uniform(*scale_range)

    def scale_around_center(arr):
        # Compute centroid across all keypoints and frames
        centroid = arr.mean(axis=(0, 1), keepdims=True)  # (1, 1, 3)
        return centroid + (arr - centroid) * scale

    return scale_around_center(pose), scale_around_center(left_hand), scale_around_center(right_hand)


def temporal_crop(pose, left_hand, right_hand, rng=None, crop_ratio=(0.0, 0.15)):
    """Randomly trim the start and/or end of the sequence.

    Simulates videos that don't start/end exactly at sign boundaries.
    Independently crops up to crop_ratio[1] from each end.
    Always keeps at least 50% of the original frames.
    """
    if rng is None:
        rng = np.random.default_rng()

    T = pose.shape[0]
    start_ratio = rng.uniform(*crop_ratio)
    end_ratio = rng.uniform(*crop_ratio)

    # Ensure we keep at least half the frames
    if start_ratio + end_ratio > 0.5:
        total = start_ratio + end_ratio
        start_ratio = start_ratio / total * 0.5
        end_ratio = end_ratio / total * 0.5

    start = int(T * start_ratio)
    end = T - int(T * end_ratio)
    end = max(end, start + 1)  # keep at least 1 frame

    return pose[start:end], left_hand[start:end], right_hand[start:end]


def joint_dropout(pose, left_hand, right_hand, rng=None, drop_prob=0.05):
    """Randomly zero out entire keypoints across all frames.

    Simulates MediaPipe detection failures (especially common for hands)
    and forces the model to not over-rely on any single joint.
    Each keypoint is independently dropped with probability drop_prob.
    """
    if rng is None:
        rng = np.random.default_rng()

    def drop_joints(arr):
        # arr: (T, K, 3) — drop entire keypoints (same joints across all frames)
        K = arr.shape[1]
        joint_mask = rng.random(K) > drop_prob  # True = keep
        result = arr.copy()
        result[:, ~joint_mask, :] = 0.0
        return result

    return drop_joints(pose), drop_joints(left_hand), drop_joints(right_hand)


def random_rotation(pose, left_hand, right_hand, rng=None, max_angle_deg=15):
    """Randomly rotate keypoints around the Z-axis (in the XY plane).

    Simulates tilted cameras or signers not facing the camera straight on.
    Rotation is applied around the centroid of the pose keypoints.
    """
    if rng is None:
        rng = np.random.default_rng()

    angle = rng.uniform(-max_angle_deg, max_angle_deg) * np.pi / 180
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    # Rotation center: mean of pose across all frames
    center_x = pose[:, :, 0].mean()
    center_y = pose[:, :, 1].mean()

    def rotate(arr):
        result = arr.copy()
        x = result[:, :, 0] - center_x
        y = result[:, :, 1] - center_y
        result[:, :, 0] = x * cos_a - y * sin_a + center_x
        result[:, :, 1] = x * sin_a + y * cos_a + center_y
        return result

    return rotate(pose), rotate(left_hand), rotate(right_hand)
