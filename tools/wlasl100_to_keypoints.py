import argparse
import json
import os
from pathlib import Path

import cv2
import mediapipe as mp

from utils.keypoint_utils import extract_keypoints, normalize_keypoints, reconstruct_video_from_keypoints, save_keypoints

parser = argparse.ArgumentParser()
parser.add_argument("--num_labels", type=int, default=100, help="Number of labels to process (default: all 100)")
args = parser.parse_args()

WLASL_DIR = Path("data", "videos_wlasl100")
VIDEOS_DIR = WLASL_DIR / "videos"
NSLT_JSON = WLASL_DIR / "nslt_100.json"
WLASL_JSON = WLASL_DIR / "WLASL_v0.3.json"

KEYPOINTS_DIR = Path("data", "keypoints")
RENDERS_DIR = Path("data", "keypoint_renders")

# Build video_id -> gloss mapping from WLASL_v0.3.json
with open(WLASL_JSON) as f:
    wlasl_data = json.load(f)

vid_to_gloss = {}
for entry in wlasl_data:
    for inst in entry["instances"]:
        vid_to_gloss[inst["video_id"]] = entry["gloss"]

# Load nslt_100.json for the 100-class subset and splits
with open(NSLT_JSON) as f:
    nslt_data = json.load(f)

# Collect glosses that have available videos, pick first N
available_glosses = []
for entry in wlasl_data:
    gloss = entry["gloss"]
    has_video = any(
        (VIDEOS_DIR / f"{inst['video_id']}.mp4").exists()
        for inst in entry["instances"]
        if inst["video_id"] in nslt_data
    )
    if has_video and gloss not in available_glosses:
        available_glosses.append(gloss)

selected_glosses = set(available_glosses[:args.num_labels])
print(f"Processing {len(selected_glosses)} labels: {sorted(selected_glosses)}\n")

mp_holistic = mp.solutions.holistic

total = sum(1 for vid_id in nslt_data if vid_to_gloss.get(vid_id) in selected_glosses)
processed = 0
skipped = 0

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for vid_id, info in nslt_data.items():
        gloss = vid_to_gloss.get(vid_id)
        if gloss not in selected_glosses:
            continue

        video_path = VIDEOS_DIR / f"{vid_id}.mp4"
        if not video_path.exists():
            skipped += 1
            continue

        split = info["subset"]  # train, val, or test
        label_dir = gloss  # folder named after the sign label

        try:
            cap = cv2.VideoCapture(str(video_path))
            results = extract_keypoints(cap, holistic)
            results = normalize_keypoints(results)

            kp_save_dir = str(KEYPOINTS_DIR / split / label_dir)
            render_save_dir = str(RENDERS_DIR / split / label_dir)
            filename = f"{vid_id}.mp4"

            save_keypoints(results, save_dir=kp_save_dir, filename=filename)
            reconstruct_video_from_keypoints(results, save_dir=render_save_dir, filename=filename)

            processed += 1
            print(f"[{processed}/{total}] {gloss}/{vid_id} ({split})")
        except Exception as e:
            print(f"Error processing {vid_id} ({gloss}): {e}")
            skipped += 1
        finally:
            cap.release()

print(f"\nDone. Processed: {processed}, Skipped: {skipped}")
