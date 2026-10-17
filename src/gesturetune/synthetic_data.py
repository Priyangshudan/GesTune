"""
Synthetic data generator utility for tests and demo benchmarks.
Generates realistic 21-landmark coordinates according to the geometric constraints
of the 7 gestures, including intentional missing-hand frames, duplicates, and controlled outliers.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from src.gesturetune import get_raw_columns

def generate_hand_skeleton_base() -> np.ndarray:
    """Generate default neutral hand landmarks [21, 3] centered at wrist (0, 0, 0)."""
    coords = np.zeros((21, 3))
    # 0: Wrist
    coords[0] = [0.5, 0.8, 0.0]
    # Thumb: 1, 2, 3, 4
    coords[1] = [0.46, 0.75, -0.01]
    coords[2] = [0.43, 0.70, -0.02]
    coords[3] = [0.40, 0.65, -0.03]
    coords[4] = [0.38, 0.60, -0.04]
    # Index: 5, 6, 7, 8
    coords[5] = [0.46, 0.62, 0.0]
    coords[6] = [0.45, 0.54, -0.01]
    coords[7] = [0.44, 0.48, -0.02]
    coords[8] = [0.43, 0.42, -0.03]
    # Middle: 9, 10, 11, 12
    coords[9] = [0.50, 0.60, 0.0]
    coords[10] = [0.50, 0.52, -0.01]
    coords[11] = [0.50, 0.45, -0.02]
    coords[12] = [0.50, 0.39, -0.03]
    # Ring: 13, 14, 15, 16
    coords[13] = [0.54, 0.62, 0.0]
    coords[14] = [0.55, 0.54, -0.01]
    coords[15] = [0.55, 0.48, -0.02]
    coords[16] = [0.55, 0.42, -0.03]
    # Pinky: 17, 18, 19, 20
    coords[17] = [0.58, 0.65, 0.0]
    coords[18] = [0.60, 0.58, -0.01]
    coords[19] = [0.61, 0.53, -0.02]
    coords[20] = [0.62, 0.48, -0.03]
    return coords

def fold_finger(coords: np.ndarray, tip_idx: int, dip_idx: int, pip_idx: int, mcp_idx: int):
    """Fold a finger towards palm."""
    mcp = coords[mcp_idx]
    coords[pip_idx] = mcp + np.array([0.0, 0.04, 0.02])
    coords[dip_idx] = mcp + np.array([0.0, 0.06, 0.03])
    coords[tip_idx] = mcp + np.array([0.0, 0.08, 0.04])

def fold_thumb(coords: np.ndarray):
    """Fold thumb inward across palm."""
    coords[3] = [0.46, 0.68, 0.02]
    coords[4] = [0.48, 0.67, 0.03]

def extend_thumb_up(coords: np.ndarray):
    """Extend thumb vertically upward."""
    coords[1] = [0.47, 0.72, 0.0]
    coords[2] = [0.45, 0.65, 0.0]
    coords[3] = [0.45, 0.57, 0.0]
    coords[4] = [0.45, 0.50, 0.0]

def extend_thumb_down(coords: np.ndarray):
    """Extend thumb vertically downward."""
    coords[1] = [0.47, 0.83, 0.0]
    coords[2] = [0.45, 0.88, 0.0]
    coords[3] = [0.45, 0.93, 0.0]
    coords[4] = [0.45, 0.98, 0.0]

def point_horizontal(coords: np.ndarray, direction: str = "right"):
    """Displace index tip horizontally."""
    dx = 0.16 if direction == "right" else -0.16
    coords[6] = coords[5] + np.array([dx * 0.33, -0.02, 0.0])
    coords[7] = coords[5] + np.array([dx * 0.66, -0.03, 0.0])
    coords[8] = coords[5] + np.array([dx, -0.04, 0.0])

def generate_gesture_skeleton(gesture_label: str) -> np.ndarray:
    """Generate exact landmark coordinates for a target gesture."""
    coords = generate_hand_skeleton_base()
    
    if gesture_label == "open_palm":
        # All extended (default base has all extended)
        pass
    elif gesture_label == "fist":
        fold_thumb(coords)
        fold_finger(coords, 8, 7, 6, 5)
        fold_finger(coords, 12, 11, 10, 9)
        fold_finger(coords, 16, 15, 14, 13)
        fold_finger(coords, 20, 19, 18, 17)
    elif gesture_label == "thumbs_up":
        extend_thumb_up(coords)
        fold_finger(coords, 8, 7, 6, 5)
        fold_finger(coords, 12, 11, 10, 9)
        fold_finger(coords, 16, 15, 14, 13)
        fold_finger(coords, 20, 19, 18, 17)
    elif gesture_label == "thumbs_down":
        extend_thumb_down(coords)
        fold_finger(coords, 8, 7, 6, 5)
        fold_finger(coords, 12, 11, 10, 9)
        fold_finger(coords, 16, 15, 14, 13)
        fold_finger(coords, 20, 19, 18, 17)
    elif gesture_label == "index_finger":
        fold_thumb(coords)
        fold_finger(coords, 12, 11, 10, 9)
        fold_finger(coords, 16, 15, 14, 13)
        fold_finger(coords, 20, 19, 18, 17)
    elif gesture_label == "two_fingers":
        fold_thumb(coords)
        fold_finger(coords, 16, 15, 14, 13)
        fold_finger(coords, 20, 19, 18, 17)
    elif gesture_label == "point_right":
        fold_thumb(coords)
        point_horizontal(coords, direction="right")
        fold_finger(coords, 12, 11, 10, 9)
        fold_finger(coords, 16, 15, 14, 13)
        fold_finger(coords, 20, 19, 18, 17)
    elif gesture_label == "point_left":
        fold_thumb(coords)
        point_horizontal(coords, direction="left")
        fold_finger(coords, 12, 11, 10, 9)
        fold_finger(coords, 16, 15, 14, 13)
        fold_finger(coords, 20, 19, 18, 17)
    return coords

def generate_synthetic_dataset(n_samples_per_gesture: int = 30, no_hand_count: int = 15,
                               seed: int = 42) -> pd.DataFrame:
    """Generate a realistic dataset adhering strictly to the 67-column schema."""
    np.random.seed(seed)
    cols = get_raw_columns()
    rows = []
    
    gestures = ["index_finger", "two_fingers", "point_left", "point_right", "open_palm", "fist", "thumbs_up"]
    cur_time = 1700000000.0

    # 1. Valid gesture samples
    for g in gestures:
        base_skeleton = generate_gesture_skeleton(g)
        for i in range(n_samples_per_gesture):
            cur_time += np.random.uniform(0.1, 0.4)
            # Add small biological / tracking jitter
            noise = np.random.normal(0, 0.005, size=base_skeleton.shape)
            skeleton = np.clip(base_skeleton + noise, 0.01, 0.99)
            conf = float(np.random.uniform(0.85, 0.99))

            row = {
                "timestamp": round(cur_time, 4),
                "gesture_label": g,
                "hand_detected": 1,
                "confidence": conf
            }
            for lm_idx in range(21):
                row[f"landmark_{lm_idx}_x"] = skeleton[lm_idx, 0]
                row[f"landmark_{lm_idx}_y"] = skeleton[lm_idx, 1]
                row[f"landmark_{lm_idx}_z"] = skeleton[lm_idx, 2]
            rows.append(row)

    # 2. Add controlled outliers (e.g. erratic jump / speed)
    for _ in range(5):
        cur_time += 0.05
        base_skeleton = generate_gesture_skeleton("open_palm")
        outlier_skeleton = base_skeleton + np.random.normal(0, 0.08, size=base_skeleton.shape)
        row = {
            "timestamp": round(cur_time, 4),
            "gesture_label": "open_palm",
            "hand_detected": 1,
            "confidence": 0.52
        }
        for lm_idx in range(21):
            row[f"landmark_{lm_idx}_x"] = float(outlier_skeleton[lm_idx, 0])
            row[f"landmark_{lm_idx}_y"] = float(outlier_skeleton[lm_idx, 1])
            row[f"landmark_{lm_idx}_z"] = float(outlier_skeleton[lm_idx, 2])
        rows.append(row)

    # 3. Add deliberate missing-hand observations (structural missingness)
    for _ in range(no_hand_count):
        cur_time += np.random.uniform(0.2, 0.8)
        chosen_g = np.random.choice(gestures)
        row = {
            "timestamp": round(cur_time, 4),
            "gesture_label": chosen_g,
            "hand_detected": 0,
            "confidence": np.nan
        }
        for lm_idx in range(21):
            row[f"landmark_{lm_idx}_x"] = np.nan
            row[f"landmark_{lm_idx}_y"] = np.nan
            row[f"landmark_{lm_idx}_z"] = np.nan
        rows.append(row)

    # 4. Add exact duplicate rows (3 duplicates)
    for _ in range(3):
        dup_row = rows[np.random.randint(0, len(rows))].copy()
        rows.append(dup_row)

    df = pd.DataFrame(rows, columns=cols)
    return df

