import numpy as np


def _is_detected(arr, threshold=1e-6):
    """Per-frame detection mask: True if the body part was detected in that frame.

    Undetected body parts are stored as all-zero (or near-zero) arrays.
    Uses a small threshold to catch values that are essentially zero
    but not exactly 0.0 due to floating point.
    Returns: (T,) boolean array.
    """
    return np.any(np.abs(arr) > threshold, axis=(1, 2))


def random_augment(pose, left_hand, right_hand, rng=None, p=0.5, track=False):
    """Apply each augmentation independently with probability p.

    With 7 augmentations at p=0.5, there are 2^7 = 128 possible combinations,
    so two views of the same sample will almost always differ meaningfully.

    Args:
        pose: (T, K, 3)
        left_hand: (T, K, 3)
        right_hand: (T, K, 3)
        rng: np.random.Generator (optional, for reproducibility)
        p: probability of applying each augmentation
        track: if True, also return a list of descriptions of applied augmentations

    Returns:
        If track=False: (pose, left_hand, right_hand)
        If track=True:  (pose, left_hand, right_hand, applied_list)
    """
    if rng is None:
        rng = np.random.default_rng()

    augmentations = [
        flip_horizontal,
        random_rotation,
        spatial_scale,
        joint_dropout,
        joint_noise,
        random_speed_change,
        temporal_crop,
    ]

    applied = []
    for aug in augmentations:
        if rng.random() < p:
            result = aug(pose, left_hand, right_hand, rng, track=track)
            if track:
                pose, left_hand, right_hand, desc = result
                applied.append(desc)
            else:
                pose, left_hand, right_hand = result

    if track:
        if not applied:
            applied.append("No augmentation")
        return pose, left_hand, right_hand, applied
    return pose, left_hand, right_hand


def flip_horizontal(pose, left_hand, right_hand, rng=None, track=False):
    """Flip x-coordinates, swap left/right hands, and swap bilateral pose landmarks.

    MediaPipe pose landmarks are anatomically labeled (person's left/right).
    After mirroring x, we must swap the bilateral pairs so that
    index 11 (left shoulder) becomes index 12 (right shoulder), etc.

    Upper body bilateral pairs (indices 0-22):
        11 <-> 12  (shoulders)
        13 <-> 14  (elbows)
        15 <-> 16  (wrists)
        17 <-> 18  (pinky)
        19 <-> 20  (index finger)
        21 <-> 22  (thumb)

    Preserves zero frames (undetected body parts).
    """
    # Bilateral pose landmark pairs to swap
    POSE_SWAP_PAIRS = [(11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22)]

    def flip_x(arr):
        flipped = arr.copy()
        detected = _is_detected(arr)
        flipped[detected, :, 0] = 1 - flipped[detected, :, 0]
        return flipped

    flipped_pose = flip_x(pose)
    # Swap bilateral pose landmarks
    for left_idx, right_idx in POSE_SWAP_PAIRS:
        if left_idx < flipped_pose.shape[1] and right_idx < flipped_pose.shape[1]:
            flipped_pose[:, [left_idx, right_idx]] = flipped_pose[:, [right_idx, left_idx]]

    # Swap left/right hand arrays AND flip their x
    result = (flipped_pose, flip_x(right_hand), flip_x(left_hand))
    if track:
        return (*result, "Flip horizontal (swap L/R)")
    return result


def random_speed_change(pose, left_hand, right_hand, rng=None, fps=25, track=False):
    """Simulate speed variation by resampling to a different number of frames.

    speed_factor > 1: fewer output frames (motion plays faster, up to 4x)
    speed_factor < 1: more output frames (motion plays slower, down to 0.25x)
    Output is clamped to [1s, 4s] duration at the given fps.

    Interpolation is detection-aware: only interpolates between frames where
    the body part is detected in BOTH neighbors. If one neighbor is undetected,
    uses the detected neighbor's value. If both are undetected, output is zero.
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
        detected = _is_detected(arr)  # (T,)
        lo_detected = detected[lower]  # (out_T,)
        hi_detected = detected[upper]  # (out_T,)

        # Default: linear interpolation
        result = arr[lower] * (1 - alpha) + arr[upper] * alpha

        # Where only lower is detected, use lower (no blend toward zero)
        only_lo = lo_detected & ~hi_detected
        if only_lo.any():
            result[only_lo] = arr[lower[only_lo]]

        # Where only upper is detected, use upper
        only_hi = ~lo_detected & hi_detected
        if only_hi.any():
            result[only_hi] = arr[upper[only_hi]]

        # Where neither is detected, zero
        neither = ~lo_detected & ~hi_detected
        if neither.any():
            result[neither] = 0.0

        return result

    result = (resample(pose), resample(left_hand), resample(right_hand))
    if track:
        direction = 'slower' if out_T > T else 'faster'
        ratio = max(out_T, T) / max(min(out_T, T), 1)
        desc = f"Speed: {speed_factor:.2f}x ({T}->{out_T} frames, {direction} {ratio:.1f}x)"
        return (*result, desc)
    return result


def joint_noise(pose, left_hand, right_hand, rng=None, track=False):
    """Add Gaussian noise to keypoint coordinates.

    Uses higher noise for body (larger joints, less precise movements)
    and lower noise for hands (finger positions are semantically important in ASL).
    Noise std is randomized each call. Clamped to [-3*std, +3*std].
    Only applied to detected frames.
    """
    if rng is None:
        rng = np.random.default_rng()

    body_std = rng.uniform(0.001, 0.005)
    hand_std = rng.uniform(0.0005, 0.002)

    def add_noise(arr, std):
        result = arr.copy()
        detected = _is_detected(arr)
        noise = rng.normal(0, std, size=arr[detected].shape).astype(arr.dtype)
        noise = np.clip(noise, -3 * std, 3 * std)
        result[detected] += noise
        return result

    result = (add_noise(pose, body_std),
              add_noise(left_hand, hand_std),
              add_noise(right_hand, hand_std))
    if track:
        desc = f"Joint noise (body std={body_std:.4f}, hand std={hand_std:.4f})"
        return (*result, desc)
    return result


def spatial_scale(pose, left_hand, right_hand, rng=None, track=False):
    """Randomly scale the skeleton around the POSE centroid.

    Scale factor is randomized between 0.8 and 1.2 each call.
    All body parts are scaled around the same center point (the pose centroid
    computed from detected frames only). Undetected frames (zeros) are preserved.
    """
    if rng is None:
        rng = np.random.default_rng()

    scale = rng.uniform(0.8, 1.2)

    # Compute centroid from detected pose frames only
    pose_detected = _is_detected(pose)
    if not pose_detected.any():
        if track:
            return pose, left_hand, right_hand, f"Spatial scale: {scale:.2f}x (no pose detected)"
        return pose, left_hand, right_hand

    centroid = pose[pose_detected].mean(axis=(0, 1), keepdims=True)

    def scale_part(arr, original):
        result = arr.copy()
        detected = _is_detected(original)
        result[detected] = centroid + (arr[detected] - centroid) * scale
        return result

    result = (scale_part(pose, pose),
              scale_part(left_hand, left_hand),
              scale_part(right_hand, right_hand))
    if track:
        return (*result, f"Spatial scale: {scale:.2f}x")
    return result


def temporal_crop(pose, left_hand, right_hand, rng=None, track=False):
    """Randomly trim the start and/or end of the sequence.

    Crop ratios are randomized between 0-15% from each end.
    Always keeps at least 50% of the original frames.
    """
    if rng is None:
        rng = np.random.default_rng()

    T = pose.shape[0]
    start_ratio = rng.uniform(0.0, 0.15)
    end_ratio = rng.uniform(0.0, 0.15)

    # Ensure we keep at least half the frames
    if start_ratio + end_ratio > 0.5:
        total = start_ratio + end_ratio
        start_ratio = start_ratio / total * 0.5
        end_ratio = end_ratio / total * 0.5

    start = int(T * start_ratio)
    end = T - int(T * end_ratio)
    end = max(end, start + 1)

    result = (pose[start:end], left_hand[start:end], right_hand[start:end])
    if track:
        out_T = end - start
        desc = f"Temporal crop: {T}->{out_T} frames (start={start_ratio:.0%}, end={end_ratio:.0%})"
        return (*result, desc)
    return result


def joint_dropout(pose, left_hand, right_hand, rng=None, track=False):
    """Randomly zero out entire keypoints across all frames.

    Drop probability is randomized between 2-10% each call.
    Simulates MediaPipe detection failures.
    """
    if rng is None:
        rng = np.random.default_rng()

    drop_prob = rng.uniform(0.02, 0.10)

    dropped_counts = {}

    def drop_joints(arr, part_name):
        K = arr.shape[1]
        joint_mask = rng.random(K) > drop_prob  # True = keep
        result = arr.copy()
        n_dropped = (~joint_mask).sum()
        dropped_counts[part_name] = int(n_dropped)
        result[:, ~joint_mask, :] = 0.0
        return result

    result = (drop_joints(pose, 'body'),
              drop_joints(left_hand, 'left'),
              drop_joints(right_hand, 'right'))
    if track:
        parts = [f"{k}:{v}" for k, v in dropped_counts.items() if v > 0]
        detail = ', '.join(parts) if parts else 'none'
        desc = f"Joint dropout: {drop_prob:.0%} ({detail})"
        return (*result, desc)
    return result


def random_rotation(pose, left_hand, right_hand, rng=None, track=False):
    """Randomly rotate keypoints around the Z-axis (in the XY plane).

    Angle is randomized between -15 and +15 degrees each call.
    Rotation is applied around the pose centroid (computed from detected frames).
    Undetected frames (zeros) are preserved.
    """
    if rng is None:
        rng = np.random.default_rng()

    angle_deg = rng.uniform(-15, 15)
    angle = angle_deg * np.pi / 180
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    # Rotation center: mean of detected pose frames
    pose_detected = _is_detected(pose)
    if not pose_detected.any():
        if track:
            return pose, left_hand, right_hand, f"Rotation: {angle_deg:+.1f} deg (no pose detected)"
        return pose, left_hand, right_hand

    center_x = pose[pose_detected, :, 0].mean()
    center_y = pose[pose_detected, :, 1].mean()

    def rotate(arr, original):
        result = arr.copy()
        detected = _is_detected(original)
        x = result[detected, :, 0] - center_x
        y = result[detected, :, 1] - center_y
        result[detected, :, 0] = x * cos_a - y * sin_a + center_x
        result[detected, :, 1] = x * sin_a + y * cos_a + center_y
        return result

    result = (rotate(pose, pose),
              rotate(left_hand, left_hand),
              rotate(right_hand, right_hand))
    if track:
        return (*result, f"Rotation: {angle_deg:+.1f} deg")
    return result
