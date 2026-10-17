"""
Data acquisition module (capture.py).
Captures labelled hand landmarks from webcam into data/raw_landmarks.csv.
- Displays live webcam feed with skeleton.
- Operator selects gesture label using keys 1-7.
- Press SPACE to record one observation.
- Exactly 67 columns (4 metadata + 63 coordinates).
- When no hand is detected, records row with hand_detected = 0 and NaNs for all coordinates and confidence.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import cv2
import pandas as pd
import numpy as np

from . import load_config, get_raw_columns
from .hand_tracker import HandTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GESTURE_KEYS = {
    ord('1'): "index_finger",
    ord('2'): "two_fingers",
    ord('3'): "point_left",
    ord('4'): "point_right",
    ord('5'): "open_palm",
    ord('6'): "fist",
    ord('7'): "thumbs_up",
    ord('8'): "thumbs_down"
}

def record_observation(csv_path: Path, gesture_label: str, hand_detected: bool,
                       landmarks: Optional[List[Dict[str, float]]], confidence: Optional[float]) -> Dict[str, Any]:
    """Append one record to raw_landmarks.csv ensuring 67 columns schema."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cols = get_raw_columns()

    timestamp = round(time.time(), 4)
    row = {
        "timestamp": timestamp,
        "gesture_label": gesture_label,
        "hand_detected": 1 if hand_detected else 0,
        "confidence": confidence if (hand_detected and confidence is not None) else np.nan
    }

    if hand_detected and landmarks and len(landmarks) == 21:
        for i, lm in enumerate(landmarks):
            row[f"landmark_{i}_x"] = float(lm['x'])
            row[f"landmark_{i}_y"] = float(lm['y'])
            row[f"landmark_{i}_z"] = float(lm['z'])
    else:
        for i in range(21):
            row[f"landmark_{i}_x"] = np.nan
            row[f"landmark_{i}_y"] = np.nan
            row[f"landmark_{i}_z"] = np.nan

    df_row = pd.DataFrame([row], columns=cols)
    
    # Check if file exists to write header or append
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    df_row.to_csv(csv_path, mode='a', header=not file_exists, index=False)
    return row

def run_capture(config_path: str = "config/config.yaml"):
    config = load_config(config_path)

    cam_cfg = config.get("camera", {})
    device_idx = cam_cfg.get("device_index", 0)
    width = cam_cfg.get("width", 640)
    height = cam_cfg.get("height", 480)
    mirror = cam_cfg.get("mirror", True)

    mp_cfg = config.get("mediapipe", {})
    raw_csv = Path(config.get("data", {}).get("raw_dataset_path", "data/raw_landmarks.csv"))

    tracker = HandTracker(
        model_path=mp_cfg.get("model_path", "models/hand_landmarker.task"),
        num_hands=1,
        min_hand_detection_confidence=mp_cfg.get("min_hand_detection_confidence", 0.5),
        min_hand_presence_confidence=mp_cfg.get("min_hand_presence_confidence", 0.5),
        min_tracking_confidence=mp_cfg.get("min_tracking_confidence", 0.5)
    )

    cap = cv2.VideoCapture(device_idx)
    if not cap.isOpened():
        logger.error(f"Cannot open webcam with device index {device_idx}. Exiting capture.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    active_label = "index_finger"
    last_status = "Ready to record"
    record_count = 0

    if raw_csv.exists():
        try:
            existing = pd.read_csv(raw_csv)
            record_count = len(existing)
            logger.info(f"Loaded existing dataset with {record_count} observations.")
        except Exception:
            pass

    logger.info("GestureTune Data Capture started. Keys: 1-7 (select label), SPACE (record), 'q' (exit).")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            if mirror:
                frame = cv2.flip(frame, 1)

            hand_detected, landmarks, conf = tracker.process_frame(frame)

            if hand_detected and landmarks:
                frame = tracker.draw_landmarks_on_image(frame, landmarks)

            # Draw HUD
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            cv2.putText(frame, f"Active Label (1-7): [{active_label}]", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            cv2.putText(frame, f"Hand: {'Detected' if hand_detected else 'NOT DETECTED'} | Conf: {f'{conf:.2f}' if conf else 'NaN'}",
                        (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if hand_detected else (0, 0, 255), 2)
            cv2.putText(frame, f"Total Records: {record_count} | Status: {last_status}", (15, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

            cv2.putText(frame, "Keys: 1:Idx 2:Two 3:Left 4:Right 5:Palm 6:Fist 7:Thumb | SPACE:Record | 'q':Quit",
                        (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 180), 1)

            cv2.imshow("GestureTune - Data Acquisition", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in GESTURE_KEYS:
                active_label = GESTURE_KEYS[key]
                last_status = f"Selected '{active_label}'"
            elif key == ord(' '):
                # Record observation
                rec = record_observation(raw_csv, active_label, hand_detected, landmarks, conf)
                record_count += 1
                if hand_detected:
                    last_status = f"Recorded '{active_label}' (Hand Detected)"
                else:
                    last_status = f"Recorded '{active_label}' (No Hand - Structural NaN)"
            elif key == ord('q') or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info(f"Capture finished. Total records in {raw_csv}: {record_count}")

if __name__ == "__main__":
    run_capture()
