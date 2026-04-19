"""Real-time ASL sign recognition from webcam.

Captures video, extracts MediaPipe keypoints, overlays skeleton,
and uses a state machine to accumulate frames and run inference.
Supports loading multiple models and comparing predictions side by side.

States:
    IDLE        — no hands visible, waiting
    RECORDING   — hands detected, accumulating frames
    SHOWING     — predictions displayed, holding for a moment before reset

Usage:
    python demo.py                                          # skeleton only
    python demo.py --checkpoint path/to/best_model.pt       # single model
    python demo.py --checkpoints experiments/trained_models/*/best_model.pt  # all models
    python demo.py --smoke_test                            # fake 7 models to test UI
    python demo.py --smoke_test 3                          # fake 3 models
"""

import os
import time
import glob
import argparse
import numpy as np
import cv2
import torch
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# State machine
STATE_IDLE = 'IDLE'
STATE_RECORDING = 'RECORDING'
STATE_SHOWING = 'SHOWING'

# Thresholds
MIN_FRAMES = 30          # minimum frames to run inference (~ 1s at 30fps)
MAX_FRAMES = 60          # cut off and infer at this many frames
HANDS_GONE_TOLERANCE = 5 # consecutive no-hand frames before stopping
SHOW_DURATION = 3.0      # seconds to hold predictions on screen

# Map experiment folder names to architecture flags
EXPERIMENT_CONFIG = {
    '1_flat_ce':                  {'use_triplet': False, 'use_rope': False},
    '2_flat_supcon_ce':           {'use_triplet': False, 'use_rope': False},
    '3_triplet_ce':               {'use_triplet': True,  'use_rope': False},
    '4_triplet_supcon_ce':        {'use_triplet': True,  'use_rope': False},
    '5_triplet_rope_supcon_ce':   {'use_triplet': True,  'use_rope': True},
    '6_triplet_pt_supcon_ce':     {'use_triplet': True,  'use_rope': False},
    '7_triplet_rope_pt_supcon_ce':{'use_triplet': True,  'use_rope': True},
}


def infer_config(checkpoint_path):
    """Infer architecture flags from the checkpoint's parent folder name."""
    folder = os.path.basename(os.path.dirname(checkpoint_path))
    if folder in EXPERIMENT_CONFIG:
        return EXPERIMENT_CONFIG[folder]
    use_rope = 'rope' in folder.lower()
    use_triplet = 'flat' not in folder.lower()
    return {'use_triplet': use_triplet, 'use_rope': use_rope}


def short_name(checkpoint_path):
    """Get a short display name from the checkpoint path."""
    folder = os.path.basename(os.path.dirname(checkpoint_path))
    # Strip leading number prefix like "4_"
    if len(folder) > 2 and folder[1] == '_':
        return folder[2:]
    return folder


def has_hands(results):
    """Check if MediaPipe detected at least one hand."""
    return (results.left_hand_landmarks is not None or
            results.right_hand_landmarks is not None)


def extract_frame_keypoints(results):
    """Extract pose, left_hand, right_hand arrays from a single MediaPipe result."""
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        pose = pose[:23]
    else:
        pose = np.zeros((23, 3))

    if results.left_hand_landmarks:
        left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
    else:
        left_hand = np.zeros((21, 3))

    if results.right_hand_landmarks:
        right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
    else:
        right_hand = np.zeros((21, 3))

    return pose, left_hand, right_hand


def keypoints_to_tokens(poses, left_hands, right_hands):
    """Convert lists of keypoint arrays to model-ready tensor (1, T, 3, 69)."""
    pose_arr = np.array(poses)
    left_arr = np.array(left_hands)
    right_arr = np.array(right_hands)

    T = pose_arr.shape[0]
    body_flat = pose_arr.reshape(T, -1)
    left_flat = left_arr.reshape(T, -1)
    right_flat = right_arr.reshape(T, -1)

    left_padded = np.pad(left_flat, ((0, 0), (0, 69 - 63)))
    right_padded = np.pad(right_flat, ((0, 0), (0, 69 - 63)))

    tokens = np.stack([body_flat, left_padded, right_padded], axis=1)
    return torch.tensor(tokens, dtype=torch.float32).unsqueeze(0)


def draw_skeleton(frame, results):
    """Draw MediaPipe skeleton on the frame."""
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1))

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 200, 100), thickness=1))

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 100, 255), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(100, 200, 255), thickness=1))

    return frame


def draw_hud(frame, state, predictions, buffer_len, frame_count, model_status=None):
    """Draw status, predictions, and recording indicator on the frame.

    Layout:
        Top-center:    small recording status + frame count
        Top-right:     pulsing red dot
        Bottom-left:   model status + controls (1/4 width)
        Bottom-right:  predictions panel (3/4 width, with gap between)
    """
    h, w = frame.shape[:2]
    is_recording = state == STATE_RECORDING or (state == STATE_SHOWING and buffer_len > 0)

    # --- Top-center: recording status (small text, no background) ---
    if is_recording:
        text = f'Recording... ({buffer_len} frames)'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, text, (text_x, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 180, 255), 1)
    elif state == STATE_IDLE:
        text = 'Show your hands to start'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, text, (text_x, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # --- Top-right corner: pulsing red dot ---
    if is_recording and frame_count % 20 < 14:
        cv2.circle(frame, (w - 20, 20), 8, (0, 0, 255), -1)

    # --- Bottom-left: model status + controls (no background) ---
    if model_status:
        cv2.putText(frame, model_status, (10, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 180, 255), 1)
    cv2.putText(frame, 'Q: quit | C: clear', (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (150, 150, 150), 1)

    # --- Bottom-right: predictions panel (anchored to right edge) ---
    if predictions:
        line_h = 22
        panel_w = min(380, int(w * 0.55))
        panel_h = 26 + len(predictions) * line_h
        panel_x = w - panel_w - 3
        panel_y = h - panel_h - 3

        overlay_br = frame.copy()
        cv2.rectangle(overlay_br, (panel_x, panel_y),
                      (w - 3, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay_br, 0.7, frame, 0.3, 0, frame)

        # Header
        cv2.putText(frame, 'Predictions', (panel_x + 8, panel_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        for i, (name, label, conf, ms) in enumerate(predictions):
            y = panel_y + 32 + i * line_h
            display_name = name[:20]
            cv2.putText(frame, display_name, (panel_x + 8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (150, 180, 255), 1)
            cv2.putText(frame, label, (panel_x + 155, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            info = f'{conf:.0%} ({ms:.0f}ms)'
            cv2.putText(frame, info, (panel_x + 280, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 180, 180), 1)

    return frame


def run_all_models(models, tokens, device):
    """Run inference on all models sequentially, return list of (name, label, confidence, time_ms)."""
    results = []
    for entry in models:
        name, model_or_fake, label_names = entry[0], entry[1], entry[2]

        if model_or_fake == 'fake':
            # Smoke test: simulate inference with random results and a small delay
            delay_ms = np.random.uniform(2, 8)
            time.sleep(delay_ms / 1000)
            label = np.random.choice(label_names)
            conf = np.random.uniform(0.15, 0.95)
            results.append((name, label, conf, delay_ms))
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                _, logits = model_or_fake(tokens)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = probs.max(dim=1)
            results.append((name, label_names[pred_idx.item()], conf.item(), elapsed_ms))
    return results


def load_models(checkpoint_paths, device):
    """Load models from checkpoint paths, auto-detecting architecture from folder names.

    Returns list of (name, model, label_names) tuples.
    """
    from models.encoder import SignLanguageEncoder

    train_dir = os.path.join('data', 'keypoints', 'train')
    label_names = sorted(os.listdir(train_dir))
    num_classes = len(label_names)

    models = []
    for cp in checkpoint_paths:
        if not os.path.isfile(cp):
            print(f"  Warning: {cp} not found, skipping.")
            continue

        config = infer_config(cp)
        name = short_name(cp)

        model = SignLanguageEncoder(
            num_classes=num_classes,
            use_triplet=config['use_triplet'],
            use_rope=config['use_rope'],
        ).to(device)

        checkpoint = torch.load(cp, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        models.append((name, model, label_names))
        print(f"  Loaded: {name} (triplet={config['use_triplet']}, rope={config['use_rope']})")

    return models


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models (real or fake)
    models = []
    model_status = None

    if args.smoke_test is not None:
        # Smoke test mode: create fake models with experiment names
        n = args.smoke_test
        fake_names = list(EXPERIMENT_CONFIG.keys())[:n]
        # Pad with generic names if n > len(EXPERIMENT_CONFIG)
        while len(fake_names) < n:
            fake_names.append(f'model_{len(fake_names) + 1}')

        # Use real label names if available, otherwise generate fake ones
        train_dir = os.path.join('data', 'keypoints', 'train')
        if os.path.isdir(train_dir):
            label_names = sorted(os.listdir(train_dir))
        else:
            label_names = [f'sign_{i}' for i in range(20)]

        for name in fake_names:
            display = name[2:] if len(name) > 2 and name[1] == '_' else name
            models.append((display, 'fake', label_names))

        model_status = f'SMOKE TEST ({n} fake models)'
        print(f"Smoke test mode: {n} fake model(s) with simulated predictions.")
    else:
        # Collect real checkpoint paths
        checkpoint_paths = []
        if args.checkpoints:
            checkpoint_paths = args.checkpoints
        elif args.checkpoint:
            checkpoint_paths = [args.checkpoint]

        if checkpoint_paths:
            print(f"Loading {len(checkpoint_paths)} model(s)...")
            models = load_models(checkpoint_paths, device)
            if not models:
                model_status = 'No valid models found'
                print("Warning: no valid checkpoints loaded.")
        else:
            model_status = 'No model loaded (skeleton only)'
            print("No checkpoint provided — running skeleton visualization only.")

    if models and args.smoke_test is None:
        print(f"\n{len(models)} model(s) ready. Predictions will run sequentially on each.")
    elif not models:
        print("Running demo without model. Provide --checkpoint or --checkpoints to enable predictions.")

    # State machine
    state = STATE_IDLE
    poses_buffer = []
    left_hands_buffer = []
    right_hands_buffer = []
    no_hands_streak = 0
    current_predictions = None  # list of (name, label, conf, ms)
    show_start_time = 0.0

    # Start webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: cannot open camera {args.camera}")
        return

    cv2.namedWindow('ASL Sign Recognition', cv2.WINDOW_NORMAL)

    print(f"\nCamera opened. Min frames: {MIN_FRAMES}, max: {MAX_FRAMES}, "
          f"hand dropout tolerance: {HANDS_GONE_TOLERANCE}")
    print("Press Q to quit, C to clear.")

    frame_count = 0

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_count += 1

            # MediaPipe processing
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)

            hands_visible = has_hands(results)
            pose, left_hand, right_hand = extract_frame_keypoints(results)

            # ---- State machine ----

            if state == STATE_IDLE:
                if hands_visible:
                    state = STATE_RECORDING
                    poses_buffer = [pose]
                    left_hands_buffer = [left_hand]
                    right_hands_buffer = [right_hand]
                    no_hands_streak = 0

            elif state == STATE_RECORDING:
                poses_buffer.append(pose)
                left_hands_buffer.append(left_hand)
                right_hands_buffer.append(right_hand)

                if hands_visible:
                    no_hands_streak = 0
                else:
                    no_hands_streak += 1

                should_stop = False
                if no_hands_streak > HANDS_GONE_TOLERANCE:
                    trim = min(no_hands_streak, len(poses_buffer))
                    poses_buffer = poses_buffer[:-trim]
                    left_hands_buffer = left_hands_buffer[:-trim]
                    right_hands_buffer = right_hands_buffer[:-trim]
                    should_stop = True
                elif len(poses_buffer) >= MAX_FRAMES:
                    should_stop = True

                if should_stop:
                    if len(poses_buffer) >= MIN_FRAMES and models:
                        tokens = keypoints_to_tokens(
                            poses_buffer, left_hands_buffer, right_hands_buffer
                        ).to(device)
                        current_predictions = run_all_models(models, tokens, device)
                        state = STATE_SHOWING
                        show_start_time = time.time()

                        # Log to console
                        ts = time.strftime('%H:%M:%S')
                        print(f"\n[{ts}] Predictions ({len(poses_buffer)} frames):")
                        for name, label, conf, ms in current_predictions:
                            print(f"  {name:25s} {label:15s} {conf:.0%} ({ms:.0f}ms)")
                    else:
                        state = STATE_IDLE

                    poses_buffer = []
                    left_hands_buffer = []
                    right_hands_buffer = []
                    no_hands_streak = 0

            elif state == STATE_SHOWING:
                # Keep accumulating frames while showing results
                if hands_visible:
                    poses_buffer.append(pose)
                    left_hands_buffer.append(left_hand)
                    right_hands_buffer.append(right_hand)
                    no_hands_streak = 0
                else:
                    no_hands_streak += 1

                if time.time() - show_start_time > SHOW_DURATION:
                    if poses_buffer:
                        state = STATE_RECORDING
                    else:
                        state = STATE_IDLE
                    # Keep current_predictions on screen until replaced or C is pressed

            # ---- Drawing ----

            frame = draw_skeleton(frame, results)
            frame = draw_hud(frame, state, current_predictions,
                             len(poses_buffer), frame_count, model_status)

            cv2.imshow('ASL Sign Recognition', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty('ASL Sign Recognition', cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord('c'):
                state = STATE_IDLE
                poses_buffer = []
                left_hands_buffer = []
                right_hands_buffer = []
                no_hands_streak = 0
                current_predictions = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time ASL sign recognition from webcam')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a single model checkpoint')
    parser.add_argument('--checkpoints', type=str, nargs='+', default=None,
                        help='Paths to multiple model checkpoints (compared side by side)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--smoke_test', type=int, nargs='?', const=7, default=None,
                        help='Run with fake models to test UI (default: 7 models)')
    args = parser.parse_args()
    main(args)
