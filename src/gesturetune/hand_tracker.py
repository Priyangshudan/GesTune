"""
Hand tracking wrapper supporting MediaPipe Tasks API (with model file).
Provides 21 3D landmarks (x, y, z) and handedness confidence score.
"""

from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
import cv2

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False

class HandTracker:
    def __init__(self, model_path: str = "models/hand_landmarker.task",
                 num_hands: int = 1,
                 min_hand_detection_confidence: float = 0.5,
                 min_hand_presence_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.model_path = Path(model_path)
        self.num_hands = num_hands
        self.min_detection_confidence = min_hand_detection_confidence
        self.min_presence_confidence = min_hand_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.detector = None
        self._init_detector()

    def _init_detector(self):
        if not MEDIAPIPE_AVAILABLE:
            return
        if not self.model_path.exists():
            return

        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[bool, Optional[List[Dict[str, float]]], Optional[float]]:
        """
        Process a BGR image frame.
        Returns:
            (hand_detected: bool, landmarks: List[Dict[str, float]], confidence: float)
            If no hand is detected, returns (False, None, None).
        """
        if self.detector is None:
            return False, None, None

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection_result = self.detector.detect(mp_image)

        if not detection_result.hand_landmarks or len(detection_result.hand_landmarks) == 0:
            return False, None, None

        raw_landmarks = detection_result.hand_landmarks[0]
        landmarks_list = []
        for lm in raw_landmarks:
            landmarks_list.append({
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z)
            })

        # Confidence from handedness score
        confidence = 1.0
        if detection_result.handedness and len(detection_result.handedness) > 0:
            first_handedness = detection_result.handedness[0]
            if len(first_handedness) > 0:
                confidence = float(first_handedness[0].score)

        return True, landmarks_list, confidence

    @staticmethod
    def draw_landmarks_on_image(rgb_image: np.ndarray, landmarks: List[Dict[str, float]]) -> np.ndarray:
        """Draw hand skeleton and landmarks on an image."""
        if not landmarks:
            return rgb_image

        annotated = rgb_image.copy()
        h, w, _ = annotated.shape

        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
            (9, 13), (13, 14), (14, 15), (15, 16), # Ring
            (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
            (0, 17)                                # Palm base
        ]

        # Draw lines
        for start_idx, end_idx in HAND_CONNECTIONS:
            p1 = (int(landmarks[start_idx]['x'] * w), int(landmarks[start_idx]['y'] * h))
            p2 = (int(landmarks[end_idx]['x'] * w), int(landmarks[end_idx]['y'] * h))
            cv2.line(annotated, p1, p2, (0, 255, 0), 2)

        # Draw landmarks
        for i, lm in enumerate(landmarks):
            center = (int(lm['x'] * w), int(lm['y'] * h))
            # Color fingertips differently
            color = (0, 0, 255) if i in (4, 8, 12, 16, 20) else (255, 0, 0)
            radius = 5 if i in (4, 8, 12, 16, 20) else 4
            cv2.circle(annotated, center, radius, color, -1)

        return annotated

