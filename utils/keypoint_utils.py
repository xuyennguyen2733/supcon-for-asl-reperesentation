import os

import mediapipe as mp
import cv2
import numpy as np

#Build Keypoints using MP Holistic
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def extract_keypoints(cap, holistic):
    frame_keypoints = None
    while cap.isOpened():
      ret, image = cap.read()
      if ret:
        if not frame_keypoints:
          frame_keypoints = []
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

        frame_keypoints.append([pose, left_hand, right_hand])

        image.flags.writeable = True 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      else:
          return frame_keypoints

def reconstruct_video_from_keypoints(keypoints: np.ndarray, save_dir: str, filename: str, pose_connections, left_hand_connections, right_hand_connections,fps: float = 30.0, height: int =720, width: int =1280):
  os.makedirs(save_dir, exist_ok=True)
  save_path = os.path.join(save_dir, filename)

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

  for frame_keypoints in keypoints:
    try:
      canvas = np.zeros((height, width, 3), dtype=np.uint8)
      pose, left_hand, right_hand = frame_keypoints

      pose = pose[:, :2] * [width, height]
      left_hand = left_hand[:, :2] * [width, height]
      right_hand = right_hand[:, :2] * [width, height]

      canvas = draw_landmarks_from_coordinates(canvas, pose, pose_connections)
      canvas = draw_landmarks_from_coordinates(canvas, left_hand, left_hand_connections)
      canvas = draw_landmarks_from_coordinates(canvas, right_hand, right_hand_connections)
      out.write(canvas)
    except Exception as e:
      print(f"Error reconstructing frame: {e}")
      break

  out.release()
     
   

def draw_landmarks(image, results, mp_holistic):
    mp_drawing.draw_landmarks(
      image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Draw face connections
    mp_drawing.draw_landmarks(
      image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(
      image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(
      image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
    
def draw_landmarks_from_coordinates(image, keypoints, connections, dot_color=(0, 255, 0), line_color=(255, 255, 255), dot_radius=4, line_thickness=2):
    # Draw connections first so dots appear on top
    for start_idx, end_idx in connections:
        start = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
        end = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
        cv2.line(image, start, end, line_color, line_thickness)

    # Draw dots
    for kp in keypoints:
        pos = (int(kp[0]), int(kp[1]))
        cv2.circle(image, pos, dot_radius, dot_color, -1)

    return image

def draw_styled_landmarks(image, results, mp_holistic, mp_drawing):
    # Draw face connections
    mp_drawing.draw_landmarks(
      image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
      mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
      mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
    
