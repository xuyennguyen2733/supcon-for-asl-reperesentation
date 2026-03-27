import numpy as np


def random_augment(pose, left_hand, right_hand, rng=None, p=0.5):
    """Apply each augmentation independently with probability p.

    Each augmentation is applied (or not) via an independent coin flip,
    so a sample may get zero, one, or multiple augmentations stacked.

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
        random_speed_change
    ]

    for aug in augmentations:
        if rng.random() < p:
            pose, left_hand, right_hand = aug(pose, left_hand, right_hand, rng)

    return pose, left_hand, right_hand


def flip_horizontal(pose, left_hand, right_hand, rng=None):
    """Flip y-coordinates and swap left/right hands."""
    def flip_y(arr):
        flipped = arr.copy()
        flipped[:, :, 0] = 1 - flipped[:, :, 0]
        return flipped

    return flip_y(pose), flip_y(right_hand), flip_y(left_hand)


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
