"""
Invariant hand landmark feature extraction module.
Extracts 64 translation-, scale-, and position-invariant geometric features from 21 3D landmarks:
1. Wrist-centered coordinates normalized by robust palm scale (s = ||p9 - p0||) [63 features: 21 * 3]
2. 5 Finger extension ratios (tip-to-wrist distance / PIP-to-wrist distance)
3. 5 Joint angle cosines (wrist - MCP - PIP - TIP articulation)
4. Key pairwise fingertip distances relative to palm scale (thumb-index, index-middle, etc.)
5. Thumb vertical and horizontal deflection relative to palm orientation
6. Index horizontal pointing component relative to MCP base

Used identically across:
- Gesture registration wizard
- Model training & inference
- Fallback & testing
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np

# Landmark Indices:
# 0: Wrist
# 1-4: Thumb (CMC, MCP, IP, TIP)
# 5-8: Index (MCP, PIP, DIP, TIP)
# 9-12: Middle (MCP, PIP, DIP, TIP)
# 13-16: Ring (MCP, PIP, DIP, TIP)
# 17-20: Pinky (MCP, PIP, DIP, TIP)

FINGER_JOINTS = {
    "thumb": (0, 1, 2, 4),
    "index": (0, 5, 6, 8),
    "middle": (0, 9, 10, 12),
    "ring": (0, 13, 14, 16),
    "pinky": (0, 17, 18, 20),
}

def landmarks_to_array(landmarks: Union[List[Dict[str, float]], np.ndarray]) -> np.ndarray:
    """Convert list of 21 landmark dicts or array to a float numpy array [21, 3]."""
    if isinstance(landmarks, np.ndarray):
        if landmarks.shape == (21, 3):
            return landmarks.astype(np.float32)
        elif landmarks.ndim == 1 and landmarks.shape[0] == 63:
            return landmarks.reshape(21, 3).astype(np.float32)
        else:
            raise ValueError(f"Invalid landmark array shape: {landmarks.shape}")

    if not isinstance(landmarks, (list, tuple)) or len(landmarks) < 21:
        raise ValueError(f"Expected at least 21 landmarks, got {len(landmarks) if hasattr(landmarks, '__len__') else type(landmarks)}")

    arr = np.zeros((21, 3), dtype=np.float32)
    for i in range(21):
        lm = landmarks[i]
        if isinstance(lm, dict):
            arr[i, 0] = float(lm.get("x", 0.0))
            arr[i, 1] = float(lm.get("y", 0.0))
            arr[i, 2] = float(lm.get("z", 0.0))
        elif isinstance(lm, (list, tuple, np.ndarray)):
            arr[i, :3] = lm[:3]
        else:
            raise TypeError(f"Invalid landmark element at index {i}: {type(lm)}")
    return arr

def compute_angle_cosine(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute cosine of angle formed by p1 - p2 - p3 (vertex at p2)."""
    v1 = p1 - p2
    v2 = p3 - p2
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cosine = np.dot(v1, v2) / (n1 * n2)
    return float(np.clip(cosine, -1.0, 1.0))

class LandmarkFeatureExtractor:
    """Extracts invariant normalized geometric features from 21 MediaPipe hand landmarks."""

    FEATURE_DIM = 63 + 5 + 5 + 6 + 4  # 83 invariant features

    @classmethod
    def get_feature_dim(cls) -> int:
        return cls.FEATURE_DIM

    def extract_features(self, landmarks: Union[List[Dict[str, float]], np.ndarray],
                         is_mirrored_preview: bool = True) -> np.ndarray:
        """
        Extract normalized invariant feature vector from 21 landmarks.
        Returns 1D numpy array of length 83.
        Handles zero-length, nan, degenerate cases without crashing.
        """
        try:
            pts = landmarks_to_array(landmarks)
        except Exception:
            return np.zeros(self.FEATURE_DIM, dtype=np.float32)

        if np.isnan(pts).any() or np.isinf(pts).any():
            return np.zeros(self.FEATURE_DIM, dtype=np.float32)

        wrist = pts[0].copy()
        middle_mcp = pts[9].copy()
        palm_scale = float(np.linalg.norm(middle_mcp - wrist))

        # Degenerate palm size protection
        if palm_scale < 1e-4:
            palm_scale = 1.0

        features = []

        # 1. Wrist-relative, scale-normalized 3D coordinates (21 * 3 = 63 features)
        norm_coords = (pts - wrist) / palm_scale
        features.extend(norm_coords.flatten())

        # 2. Finger extension ratios (tip-to-wrist distance / pip-to-wrist distance) (5 features)
        for finger, (w_idx, mcp_idx, pip_idx, tip_idx) in FINGER_JOINTS.items():
            dist_tip = np.linalg.norm(pts[tip_idx] - pts[w_idx])
            dist_pip = np.linalg.norm(pts[pip_idx] - pts[w_idx])
            ratio = float(dist_tip / (dist_pip + 1e-4))
            features.append(ratio)

        # 3. Finger joint angles (cosine of angle at PIP joint: MCP - PIP - TIP) (5 features)
        for finger, (w_idx, mcp_idx, pip_idx, tip_idx) in FINGER_JOINTS.items():
            cos_angle = compute_angle_cosine(pts[mcp_idx], pts[pip_idx], pts[tip_idx])
            features.append(cos_angle)

        # 4. Pairwise normalized fingertip distances (6 features)
        # Pairs: (4, 8), (8, 12), (12, 16), (16, 20), (4, 12), (4, 20)
        pairs = [(4, 8), (8, 12), (12, 16), (16, 20), (4, 12), (4, 20)]
        for idx1, idx2 in pairs:
            dist = float(np.linalg.norm(pts[idx1] - pts[idx2]) / palm_scale)
            features.append(dist)

        # 5. Orientation and pointing signals (4 features)
        # Index tip displacement relative to index MCP in x and y (normalized)
        index_tip = pts[8]
        index_mcp = pts[5]
        dx_index = float((index_tip[0] - index_mcp[0]) / palm_scale)
        dy_index = float((index_tip[1] - index_mcp[1]) / palm_scale)

        # In mirrored preview: pointing user-right corresponds to dx > 0
        features.append(dx_index)
        features.append(dy_index)

        # Thumb tip displacement relative to thumb MCP in x and y (normalized)
        thumb_tip = pts[4]
        thumb_mcp = pts[2]
        dx_thumb = float((thumb_tip[0] - thumb_mcp[0]) / palm_scale)
        dy_thumb = float((thumb_tip[1] - thumb_mcp[1]) / palm_scale)
        features.append(dx_thumb)
        features.append(dy_thumb)

        feat_arr = np.array(features, dtype=np.float32)
        # Safety check against any nan/inf
        feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)
        return feat_arr

