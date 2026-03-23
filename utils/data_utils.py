import numpy as np

def temporal_resample(keypoints, target_frames=30):
    """Linearly interpolate keypoints to a fixed number of frames.

    Args:
        keypoints: np.ndarray of shape (T, K, 3)
        target_frames: desired number of frames

    Returns:
        np.ndarray of shape (target_frames, K, 3)
    """
    T = keypoints.shape[0]
    if T == target_frames:
        return keypoints
    source_indices = np.linspace(0, T - 1, target_frames)
    lower = np.floor(source_indices).astype(int)
    upper = np.minimum(lower + 1, T - 1)
    alpha = (source_indices - lower)[:, None, None]
    return keypoints[lower] * (1 - alpha) + keypoints[upper] * alpha


def form_pose_triplet_units(pose, left_hand, right_hand):
    """Form pose triplet tokens from keypoint arrays.

    Each input has shape (T, K, 3) where T is frames, K is keypoints, 3 is x/y/z.
    Per frame, each body part's keypoints are flattened into a 1D vector,
    producing three tokens per frame (body, left hand, right hand).

    Returns:
        List of [body_tokens (T, 69), left_tokens (T, 63), right_tokens (T, 63)]
    """
    T = pose.shape[0]
    body_tokens = pose.reshape(T, -1)         # (T, 99)
    left_tokens = left_hand.reshape(T, -1)    # (T, 63)
    right_tokens = right_hand.reshape(T, -1)  # (T, 63)
    return [body_tokens, left_tokens, right_tokens]
