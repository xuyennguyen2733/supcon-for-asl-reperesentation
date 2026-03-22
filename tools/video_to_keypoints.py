import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


from utils.keypoint_utils import draw_styled_landmarks, extract_keypoints, reconstruct_video_from_keypoints

data_dir = "data/"
source_dir = os.path.join(data_dir, "videos/")  
save_dir = os.path.join(data_dir, "keypoints/")
reconstructed_dir = os.path.join(data_dir, "keypoint_renders/")

labels = {}

mp_holistic = mp.solutions.holistic # Holistic model
# Read labels from data directory
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
            #labels[child.name].append(results)
            reconstruct_video_from_keypoints(results, save_dir=str(Path(reconstructed_dir) /  child.name ), filename=file.name, pose_connections=mp_holistic.POSE_CONNECTIONS, left_hand_connections=mp_holistic.HAND_CONNECTIONS, right_hand_connections=mp_holistic.HAND_CONNECTIONS)
          except Exception as e:
            print(f"Error processing {file}: {e}")
          finally:  
            cap.release()
            cv2.destroyAllWindows()

#print("keypoints: \n", labels, "\n\n", labels['hello'][0])

