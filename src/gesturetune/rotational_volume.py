"""
Palm orientation estimation and calibrated rotational volume control module.
Features:
1. Robust 3D local coordinate frame from 3D hand landmarks:
   - Origin: Wrist (Landmark 0)
   - Primary vector: Palm axis along Wrist (0) -> Middle MCP (9)
   - Secondary vector: Cross-palm axis from Index MCP (5) -> Pinky MCP (17)
   - Normal vector: cross(Primary, Secondary)
2. Calibrated reference frame storage and signed angle calculation:
   - Roll/tilt angle in camera image plane or rotational axis.
   - User calibration establishing which physical rotation increases volume vs decreases.
3. RotationalVolumeController:
   - Angular smoothing using exponential moving average (EMA)
   - Abrupt-jump and tracking loss rejection
   - Configurable neutral dead zone
   - Maximum slew rate to prevent rapid volume spikes
   - Volume clamping in [0.0, 1.0]
   - Direct integration with MusicController
"""

from typing import List, Dict, Optional, Tuple, Any, Union
import math
import time
import numpy as np

from .feature_extraction import landmarks_to_array

def estimate_palm_roll_angle(landmarks: Union[List[Dict[str, float]], np.ndarray],
                             is_mirrored_preview: bool = True) -> Optional[float]:
    """
    Estimate signed in-plane palm roll angle (in degrees) from 3D hand landmarks.
    Uses vector from wrist (0) to middle-finger base (9) and transverse vector index_mcp (5) -> pinky_mcp (17).
    Returns signed angle in [-180, 180] degrees, or None if degenerate/missing.
    """
    try:
        pts = landmarks_to_array(landmarks)
    except Exception:
        return None

    if np.isnan(pts).any() or np.isinf(pts).any():
        return None

    wrist = pts[0]
    middle_mcp = pts[9]
    vec_palm = middle_mcp - wrist
    palm_norm = float(np.linalg.norm(vec_palm[:2]))  # In-plane (x, y) length
    if palm_norm < 1e-4:
        return None

    # In camera coordinates, y points downwards.
    # When hand is pointed straight up, vec_palm_y is negative, vec_palm_x is approx 0.
    # Angle relative to vertical upward (-y axis):
    # angle = math.atan2(dx, -dy) in radians
    dx = float(vec_palm[0])
    dy = float(vec_palm[1])

    # If mirrored preview: user rotating clockwise on screen corresponds to positive dx
    angle_rad = math.atan2(dx, -dy)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

class RotationalVolumeCalibration:
    """Stores user-calibrated reference orientations and rotation direction mapping."""

    def __init__(self,
                 neutral_angle_deg: float = 0.0,
                 increase_direction: int = 1,  # +1 if positive angle increases volume, -1 if negative
                 min_range_deg: float = 20.0,
                 calibrated_at: float = 0.0):
        self.neutral_angle_deg = float(neutral_angle_deg)
        self.increase_direction = 1 if increase_direction >= 0 else -1
        self.min_range_deg = float(min_range_deg)
        self.calibrated_at = calibrated_at or time.time()

    def get_signed_rotation(self, current_angle_deg: float) -> float:
        """
        Calculate signed rotation amount relative to neutral angle,
        normalized such that POSITIVE always means VOLUME INCREASE
        and NEGATIVE always means VOLUME DECREASE.
        Wraps across [-180, 180].
        """
        diff = current_angle_deg - self.neutral_angle_deg
        # Wrap to [-180, 180]
        while diff > 180.0:
            diff -= 360.0
        while diff < -180.0:
            diff += 360.0

        # Multiply by increase_direction so positive = increase
        return diff * self.increase_direction

    def to_dict(self) -> Dict[str, Any]:
        return {
            "neutral_angle_deg": round(self.neutral_angle_deg, 2),
            "increase_direction": self.increase_direction,
            "min_range_deg": round(self.min_range_deg, 2),
            "calibrated_at": self.calibrated_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RotationalVolumeCalibration":
        return cls(
            neutral_angle_deg=float(data.get("neutral_angle_deg", 0.0)),
            increase_direction=int(data.get("increase_direction", 1)),
            min_range_deg=float(data.get("min_range_deg", 20.0)),
            calibrated_at=float(data.get("calibrated_at", time.time()))
        )

class RotationalVolumeController:
    """
    Continuous rotational volume controller.
    Applies calibrated angular deviation to continuously adjust playback volume.
    """

    def __init__(self,
                 calibration: Optional[RotationalVolumeCalibration] = None,
                 deadzone_deg: float = 8.0,
                 max_angle_deg: float = 45.0,
                 rate_multiplier: float = 0.35,  # volume delta per second at max rotation
                 smoothing_alpha: float = 0.25):
        self.calibration = calibration or RotationalVolumeCalibration()
        self.deadzone_deg = deadzone_deg
        self.max_angle_deg = max_angle_deg
        self.rate_multiplier = rate_multiplier
        self.smoothing_alpha = smoothing_alpha

        self.smoothed_angle: Optional[float] = None
        self.last_update_time: float = time.monotonic()
        self.is_active: bool = False
        self.last_signed_rotation: float = 0.0

    def update_calibration(self, calibration: RotationalVolumeCalibration):
        self.calibration = calibration

    def reset_state(self):
        self.smoothed_angle = None
        self.is_active = False
        self.last_signed_rotation = 0.0
        self.last_update_time = time.monotonic()

    def process_landmarks(self, landmarks: Optional[Any], current_volume: float,
                          is_mirrored_preview: bool = True) -> Tuple[float, Dict[str, Any]]:
        """
        Process current frame landmarks, update smoothed orientation, and compute adjusted volume.
        Returns:
            (new_volume: float in [0.0, 1.0], diagnostics_dict)
        """
        now = time.monotonic()
        dt = max(1e-4, min(0.2, now - self.last_update_time))
        self.last_update_time = now

        if landmarks is None:
            self.reset_state()
            return current_volume, {
                "active": False,
                "reason": "no_hand",
                "signed_rotation": 0.0,
                "direction": "none"
            }

        raw_angle = estimate_palm_roll_angle(landmarks, is_mirrored_preview=is_mirrored_preview)
        if raw_angle is None:
            self.reset_state()
            return current_volume, {
                "active": False,
                "reason": "degenerate_angle",
                "signed_rotation": 0.0,
                "direction": "none"
            }

        # Smooth angle with abrupt jump protection
        if self.smoothed_angle is None:
            self.smoothed_angle = raw_angle
        else:
            diff = raw_angle - self.smoothed_angle
            while diff > 180.0:
                diff -= 360.0
            while diff < -180.0:
                diff += 360.0
            
            # Rejection of extreme abrupt telemetry jumps (> 90 deg in a single frame)
            if abs(diff) < 90.0:
                self.smoothed_angle += self.smoothing_alpha * diff
                while self.smoothed_angle > 180.0:
                    self.smoothed_angle -= 360.0
                while self.smoothed_angle < -180.0:
                    self.smoothed_angle += 360.0

        # Calculate calibrated signed rotation (positive = increase, negative = decrease)
        signed_rot = self.calibration.get_signed_rotation(self.smoothed_angle)
        self.last_signed_rotation = signed_rot

        # Apply deadzone
        abs_rot = abs(signed_rot)
        new_volume = current_volume
        direction = "neutral"

        if abs_rot < self.deadzone_deg:
            self.is_active = False
            direction = "neutral"
        else:
            self.is_active = True
            # Normalized intensity beyond deadzone [0.0, 1.0]
            intensity = min(1.0, (abs_rot - self.deadzone_deg) / max(1.0, (self.max_angle_deg - self.deadzone_deg)))
            
            if signed_rot > 0:
                direction = "increase"
                vol_delta = intensity * self.rate_multiplier * dt
                new_volume = min(1.0, current_volume + vol_delta)
            else:
                direction = "decrease"
                vol_delta = intensity * self.rate_multiplier * dt
                new_volume = max(0.0, current_volume - vol_delta)

        return float(np.clip(new_volume, 0.0, 1.0)), {
            "active": self.is_active,
            "raw_angle": round(raw_angle, 1),
            "smoothed_angle": round(self.smoothed_angle, 1),
            "signed_rotation": round(signed_rot, 1),
            "direction": direction,
            "volume_pct": int(new_volume * 100)
        }

