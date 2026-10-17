"""
Geometric rule-based recognition for the 7 static gestures:
1. index_finger (play/pause toggle)
2. two_fingers (mute/unmute toggle)
3. point_left (previous track)
4. point_right (next track)
5. open_palm (volume up)
6. fist (volume down)
7. thumbs_up (like current track)

Operates on the 21 MediaPipe hand landmarks.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np

# Landmark indices reference:
# 0: Wrist
# 1-4: Thumb (CMC, MCP, IP, TIP)
# 5-8: Index (MCP, PIP, DIP, TIP)
# 9-12: Middle (MCP, PIP, DIP, TIP)
# 13-16: Ring (MCP, PIP, DIP, TIP)
# 17-20: Pinky (MCP, PIP, DIP, TIP)

class GestureRecognizer:
    def __init__(self, horizontal_point_threshold: float = 0.08,
                 thumb_up_y_threshold: float = 0.06):
        self.horizontal_point_threshold = horizontal_point_threshold
        self.thumb_up_y_threshold = thumb_up_y_threshold

    def is_finger_extended(self, landmarks: List[Dict[str, float]], tip_idx: int, pip_idx: int, mcp_idx: int, wrist_idx: int = 0) -> bool:
        """
        Check if a regular finger (index, middle, ring, pinky) is extended.
        In image coords, y increases downwards.
        Extended means tip is significantly further from wrist and above/further than pip/mcp.
        """
        wrist = np.array([landmarks[wrist_idx]['x'], landmarks[wrist_idx]['y']])
        pip = np.array([landmarks[pip_idx]['x'], landmarks[pip_idx]['y']])
        tip = np.array([landmarks[tip_idx]['x'], landmarks[tip_idx]['y']])
        
        dist_tip_wrist = np.linalg.norm(tip - wrist)
        dist_pip_wrist = np.linalg.norm(pip - wrist)
        
        # Tip must be further from wrist than PIP
        return bool(dist_tip_wrist > dist_pip_wrist * 1.15)

    def is_thumb_extended(self, landmarks: List[Dict[str, float]], wrist_idx: int = 0) -> bool:
        """Check if thumb is extended away from palm."""
        wrist = np.array([landmarks[wrist_idx]['x'], landmarks[wrist_idx]['y']])
        thumb_tip = np.array([landmarks[4]['x'], landmarks[4]['y']])
        thumb_mcp = np.array([landmarks[2]['x'], landmarks[2]['y']])
        index_mcp = np.array([landmarks[5]['x'], landmarks[5]['y']])
        
        dist_tip_wrist = np.linalg.norm(thumb_tip - wrist)
        dist_mcp_wrist = np.linalg.norm(thumb_mcp - wrist)
        dist_tip_index_mcp = np.linalg.norm(thumb_tip - index_mcp)
        
        return bool(dist_tip_wrist > dist_mcp_wrist * 1.1 and dist_tip_index_mcp > 0.08)

    def is_thumb_pointing_up(self, landmarks: List[Dict[str, float]]) -> bool:
        """Check if thumb is extended vertically upward (lower y in camera coords)."""
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        wrist = landmarks[0]
        
        # Tip must be above IP and MCP, and substantially above wrist
        is_above = (thumb_tip['y'] < thumb_ip['y'] < thumb_mcp['y'] and 
                    (wrist['y'] - thumb_tip['y']) > self.thumb_up_y_threshold)
        # Horizontal deflection should not be extreme
        is_vertical = abs(thumb_tip['x'] - thumb_mcp['x']) < 0.12
        return bool(is_above and is_vertical)

    def is_thumb_pointing_down(self, landmarks: List[Dict[str, float]]) -> bool:
        """Check if thumb is extended vertically downward (higher y in camera coords)."""
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        wrist = landmarks[0]
        
        # Tip must be below IP and MCP, and substantially below wrist/MCP
        is_below = (thumb_tip['y'] > thumb_ip['y'] > thumb_mcp['y'] and 
                    (thumb_tip['y'] - wrist['y']) > self.thumb_up_y_threshold)
        # Horizontal deflection should not be extreme
        is_vertical = abs(thumb_tip['x'] - thumb_mcp['x']) < 0.12
        return bool(is_below and is_vertical)

    def recognize(self, landmarks: List[Dict[str, float]], is_mirrored_preview: bool = True) -> Optional[str]:
        """
        Recognize one of the gestures from 21 landmarks.
        landmarks: list of 21 dicts with 'x', 'y', 'z' in [0, 1].
        is_mirrored_preview: When true, camera feed is horizontally mirrored for intuitive user interaction.
        In mirrored preview, pointing user-right corresponds to x increasing, pointing user-left corresponds to x decreasing.
        """
        if not landmarks or len(landmarks) < 21:
            return None

        # Determine extension state of all 5 fingers
        thumb_ext = self.is_thumb_extended(landmarks)
        index_ext = self.is_finger_extended(landmarks, tip_idx=8, pip_idx=6, mcp_idx=5)
        middle_ext = self.is_finger_extended(landmarks, tip_idx=12, pip_idx=10, mcp_idx=9)
        ring_ext = self.is_finger_extended(landmarks, tip_idx=16, pip_idx=14, mcp_idx=13)
        pinky_ext = self.is_finger_extended(landmarks, tip_idx=20, pip_idx=18, mcp_idx=17)

        extended_count = sum([thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext])
        four_fingers_ext = sum([index_ext, middle_ext, ring_ext, pinky_ext])

        # Priority 1: Open Palm (All 5 fingers extended, or all 4 fingers extended with open hand spread)
        if extended_count >= 5 or (four_fingers_ext == 4 and thumb_ext):
            return "open_palm"

        # Priority 2: Fist (All 5 fingers folded)
        if extended_count == 0 or (four_fingers_ext == 0 and not thumb_ext):
            return "fist"

        # Priority 3: Thumbs Up / Down (Thumb extended, all 4 other fingers folded)
        if thumb_ext and four_fingers_ext == 0:
            if self.is_thumb_pointing_up(landmarks):
                return "thumbs_up"
            if self.is_thumb_pointing_down(landmarks):
                return "thumbs_down"

        # Priority 4: Two Fingers (Index and Middle extended, Ring and Pinky folded)
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            return "two_fingers"

        # Priority 5: Pointing gestures (Only Index finger extended, Middle, Ring, Pinky folded)
        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            # Check horizontal displacement of index tip (8) relative to index base/MCP (5)
            # In mirrored preview:
            # If user points right, on mirrored screen the tip is to the right of the base (x_tip > x_base).
            # If user points left, on mirrored screen the tip is to the left of the base (x_tip < x_base).
            tip = landmarks[8]
            mcp = landmarks[5]
            dx = tip['x'] - mcp['x']
            dy = tip['y'] - mcp['y']

            # If horizontal component dominates over vertical
            if abs(dx) > self.horizontal_point_threshold and abs(dx) > abs(dy) * 0.8:
                if dx > 0:
                    return "point_right"
                else:
                    return "point_left"
            else:
                # Vertical / upright index finger
                return "index_finger"

        return None

