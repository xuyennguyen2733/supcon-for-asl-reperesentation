import os

import mediapipe as mp
import cv2
import numpy as np

mp_holistic = mp.solutions.holistic

def extract_keypoints(cap, holistic):
    poses = []
    left_hands = []
    right_hands = []
    while cap.isOpened():
      ret, image = cap.read()
      if ret:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = holistic.process(image)

        if results.pose_landmarks:
            pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark])
        else:
            pose = np.zeros((33,3))

        if results.left_hand_landmarks:
            left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
        else:
            left_hand = np.zeros((21,3))

        if results.right_hand_landmarks:
            right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
        else:
            right_hand = np.zeros((21,3))

        poses.append(pose)
        left_hands.append(left_hand)
        right_hands.append(right_hand)

      else:
          return [poses, left_hands, right_hands]

def reconstruct_video_from_keypoints(keypoints, save_dir: str, filename: str, fps: float = 30.0, height: int = 720, width: int = 1280):
  os.makedirs(save_dir, exist_ok=True)
  save_path = os.path.join(save_dir, filename)

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
  poses, left_hands, right_hands = keypoints

  # Filter pose connections to only include indices within the pose array
  num_pose_kps = len(poses[0]) if len(poses) > 0 else 33
  pose_connections = [(a, b) for a, b in mp_holistic.POSE_CONNECTIONS if a < num_pose_kps and b < num_pose_kps]

  for pose, left_hand, right_hand in zip(poses, left_hands, right_hands):
    try:
      canvas = np.zeros((height, width, 3), dtype=np.uint8)

      pose = pose[:, :2] * [width, height]
      left_hand = left_hand[:, :2] * [width, height]
      right_hand = right_hand[:, :2] * [width, height]

      canvas = draw_landmarks_from_coordinates(canvas, pose, pose_connections)
      canvas = draw_landmarks_from_coordinates(canvas, left_hand, mp_holistic.HAND_CONNECTIONS)
      canvas = draw_landmarks_from_coordinates(canvas, right_hand, mp_holistic.HAND_CONNECTIONS)
      out.write(canvas)
    except Exception as e:
      print(f"Error reconstructing frame: {e}")
      break

  out.release()

def draw_landmarks_from_coordinates(image, keypoints, connections, dot_color=(0, 255, 0), line_color=(255, 255, 255), dot_radius=4, line_thickness=2):
    # Identify which keypoints are valid (non-zero)
    valid = [not (kp[0] == 0 and kp[1] == 0) for kp in keypoints]

    # Draw connections first so dots appear on top (skip if either end is dropped)
    for start_idx, end_idx in connections:
        if not valid[start_idx] or not valid[end_idx]:
            continue
        start = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
        end = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
        cv2.line(image, start, end, line_color, line_thickness)

    # Draw dots (skip dropped joints)
    for i, kp in enumerate(keypoints):
        if not valid[i]:
            continue
        pos = (int(kp[0]), int(kp[1]))
        cv2.circle(image, pos, dot_radius, dot_color, -1)

    return image

def normalize_keypoints(keypoints):
    """Normalize extracted keypoints. Crops pose to upper body only (23 keypoints).

    Args:
        keypoints: [poses, left_hands, right_hands] where each is a list of arrays.

    Returns:
        [poses, left_hands, right_hands] with pose cropped to indices 0-22.
    """
    poses, left_hands, right_hands = keypoints
    poses = [pose[:23] for pose in poses]
    return [poses, left_hands, right_hands]


def save_keypoints(keypoints, save_dir: str, filename: str):
    folder_name = os.path.splitext(filename)[0]
    folder_path = os.path.join(save_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    pose, left_hand, right_hand = keypoints
    np.save(os.path.join(folder_path, 'pose.npy'), pose)
    np.save(os.path.join(folder_path, 'left_hand.npy'), left_hand)
    np.save(os.path.join(folder_path, 'right_hand.npy'), right_hand)