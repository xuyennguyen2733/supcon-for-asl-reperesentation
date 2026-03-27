import os
from pathlib import Path

import cv2
import mediapipe as mp

from utils.keypoint_utils import extract_keypoints, normalize_keypoints, reconstruct_video_from_keypoints, save_keypoints

data_dir = "data/samples/"
source_dir = os.path.join(data_dir, "videos/")
save_dir = os.path.join(data_dir, "keypoints/")
reconstructed_dir = os.path.join(data_dir, "keypoint_renders/")

labels = {}

mp_holistic = mp.solutions.holistic
p = Path(source_dir)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
  for child in p.iterdir():
    if child.is_dir():
      labels[child.name] = []
      for file in child.iterdir():
        if file.is_file() and file.suffix == ".mp4":
          try:
            cap = cv2.VideoCapture(str(file))
            results = extract_keypoints(cap, holistic)
            results = normalize_keypoints(results)
            save_keypoints(results, save_dir=str(Path(save_dir) / child.name), filename=file.name)
            reconstruct_video_from_keypoints(results, save_dir=str(Path(reconstructed_dir) / child.name), filename=file.name)
          except Exception as e:
            print(f"Error processing {file}: {e}")
          finally:
            cap.release()
