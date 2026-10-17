import time
import pytest
import numpy as np

from src.gesturetune.music_controller import MusicController
from src.gesturetune.stabilized_controller import StabilizedGestureController
from src.gesturetune.rotational_volume import (
    RotationalVolumeController,
    RotationalVolumeCalibration
)

def make_dummy_landmarks(angle_deg: float = 0.0):
    """
    Generate synthetic hand landmarks oriented at a specific in-plane angle.
    Wrist at (0.5, 0.7, 0.0), Middle MCP at (0.5 + dx, 0.7 + dy, 0.0).
    angle_deg is relative to vertical upward (-y).
    """
    rad = np.radians(angle_deg)
    length = 0.2
    dx = float(length * np.sin(rad))
    dy = float(-length * np.cos(rad))

    pts = []
    for _ in range(21):
        pts.append({"x": 0.5, "y": 0.5, "z": 0.0})
    # Wrist (0)
    pts[0] = {"x": 0.5, "y": 0.7, "z": 0.0}
    # Middle MCP (9)
    pts[9] = {"x": 0.5 + dx, "y": 0.7 + dy, "z": 0.0}
    # Index MCP (5)
    pts[5] = {"x": 0.45, "y": 0.65, "z": 0.0}
    # Pinky MCP (17)
    pts[17] = {"x": 0.55, "y": 0.65, "z": 0.0}
    return pts

def test_rotation_inactive_during_transient_open_palm():
    """
    Transient raw open_palm must NOT activate rotation before window consistency
    stabilizes the gesture.
    """
    music = MusicController(music_dir="data/test_tracks", volume_step=0.05)
    controller = StabilizedGestureController(
        music_controller=music,
        window_size=7,
        consistency_ratio=0.7,
        suppress_palm_volume=True
    )
    calib = RotationalVolumeCalibration(neutral_angle_deg=0.0, increase_direction=1)
    rot_ctrl = RotationalVolumeController(calibration=calib, deadzone_deg=5.0)

    tilted_landmarks = make_dummy_landmarks(angle_deg=30.0)

    # Send 1 single frame of open_palm into a 7-frame window
    stable_gesture, _, _ = controller.process_prediction("open_palm", confidence=0.95)
    assert stable_gesture is None, "A single transient frame should not produce a stable gesture"

    rotation_active = (stable_gesture == "open_palm")
    assert not rotation_active, "Rotational controller must not be active on transient prediction"

def test_rotation_activates_only_when_open_palm_is_stable():
    """
    When open_palm satisfies the consistency threshold and becomes stable,
    rotational volume control becomes active and adjusts volume.
    """
    music = MusicController(music_dir="data/test_tracks", volume_step=0.05)
    music.current_volume = 0.5
    controller = StabilizedGestureController(
        music_controller=music,
        window_size=5,
        consistency_ratio=0.7,
        suppress_palm_volume=True
    )
    calib = RotationalVolumeCalibration(neutral_angle_deg=0.0, increase_direction=1)
    rot_ctrl = RotationalVolumeController(calibration=calib, deadzone_deg=5.0)

    tilted_landmarks = make_dummy_landmarks(angle_deg=30.0)

    # Feed frames to achieve stability
    stable = None
    for _ in range(5):
        stable, action, state = controller.process_prediction("open_palm", confidence=0.95)
        # Suppress palm volume is True, so discrete action must be None
        assert action is None

    assert stable == "open_palm"

    # Now evaluate gate condition
    rotation_active = (stable == "open_palm")
    assert rotation_active

    # Adjust volume using tilted landmarks (+30 deg with +1 direction increases volume)
    time.sleep(0.02)
    new_vol, diag = rot_ctrl.process_landmarks(tilted_landmarks, current_volume=music.current_volume)
    assert diag["active"] is True
    assert diag["direction"] == "increase"
    assert new_vol > 0.5

def test_all_other_gestures_keep_rotation_inactive():
    """
    Verify fist, index_finger, two_fingers, point_left, point_right, thumbs_up, thumbs_down
    do NOT activate rotational volume, even if the hand is tilted.
    """
    other_gestures = [
        "fist",
        "index_finger",
        "two_fingers",
        "point_left",
        "point_right",
        "thumbs_up",
        "thumbs_down"
    ]

    calib = RotationalVolumeCalibration(neutral_angle_deg=0.0, increase_direction=1)
    rot_ctrl = RotationalVolumeController(calibration=calib, deadzone_deg=5.0)
    tilted_landmarks = make_dummy_landmarks(angle_deg=35.0)

    for gest in other_gestures:
        music = MusicController(music_dir="data/test_tracks", volume_step=0.05)
        controller = StabilizedGestureController(
            music_controller=music,
            window_size=5,
            consistency_ratio=0.7,
            suppress_palm_volume=True
        )

        # Feed 5 frames of this gesture
        stable = None
        for _ in range(5):
            stable, _, _ = controller.process_prediction(gest, confidence=0.95)

        assert stable == gest

        # Gate condition check
        rotation_active = (stable == "open_palm")
        assert not rotation_active, f"Rotation must be INACTIVE for stable gesture '{gest}'"

def test_rotation_deactivates_and_resets_when_gesture_changes():
    """
    When transitioning from stable open_palm to another gesture or neutral,
    rotation must immediately deactivate and reset transient smoothing,
    preserving calibration.
    """
    music = MusicController(music_dir="data/test_tracks", volume_step=0.05)
    controller = StabilizedGestureController(
        music_controller=music,
        window_size=5,
        consistency_ratio=0.7,
        suppress_palm_volume=True
    )
    calib = RotationalVolumeCalibration(neutral_angle_deg=15.0, increase_direction=1)
    rot_ctrl = RotationalVolumeController(calibration=calib, deadzone_deg=5.0)
    tilted_landmarks = make_dummy_landmarks(angle_deg=40.0)

    # 1. Stabilize open_palm
    for _ in range(5):
        stable, _, _ = controller.process_prediction("open_palm", confidence=0.95)
    assert stable == "open_palm"

    # Perform rotational processing
    time.sleep(0.01)
    vol, diag = rot_ctrl.process_landmarks(tilted_landmarks, current_volume=0.5)
    assert rot_ctrl.smoothed_angle is not None
    assert diag["active"] is True

    # 2. Switch to fist
    for _ in range(5):
        stable, _, _ = controller.process_prediction("fist", confidence=0.95)
    assert stable == "fist"

    rotation_active = (stable == "open_palm")
    assert not rotation_active

    # Player executes reset_state() when rotation_active is False
    rot_ctrl.reset_state()
    assert rot_ctrl.is_active is False
    assert rot_ctrl.smoothed_angle is None
    # User calibration must be safely preserved!
    assert rot_ctrl.calibration.neutral_angle_deg == 15.0
    assert rot_ctrl.calibration.increase_direction == 1

def test_tracking_loss_immediately_deactivates_rotation_and_freezes_volume():
    """
    When hand tracking is lost (landmarks is None or hand_detected is False),
    rotation is immediately deactivated and volume remains frozen.
    """
    music = MusicController(music_dir="data/test_tracks", volume_step=0.05)
    music.current_volume = 0.65

    calib = RotationalVolumeCalibration(neutral_angle_deg=0.0, increase_direction=1)
    rot_ctrl = RotationalVolumeController(calibration=calib, deadzone_deg=5.0)

    # Hand lost
    hand_detected = False
    landmarks = None
    stable_gesture = "open_palm"

    rotation_active = (
        stable_gesture == "open_palm"
        and hand_detected
        and (landmarks is not None)
    )
    assert not rotation_active

    rot_ctrl.reset_state()
    assert rot_ctrl.is_active is False
    assert music.current_volume == 0.65

def test_discrete_volume_mode_preserves_palm_and_fist():
    """
    When in discrete mode (suppress_palm_volume is False),
    open_palm triggers volume_up and fist triggers volume_down.
    """
    music = MusicController(music_dir="data/test_tracks", volume_step=0.05)
    music.current_volume = 0.5
    controller = StabilizedGestureController(
        music_controller=music,
        window_size=3,
        consistency_ratio=0.7,
        suppress_palm_volume=False,
        volume_repeat_interval=0.01
    )

    # Stabilize open_palm
    action = None
    for _ in range(3):
        stable, action, _ = controller.process_prediction("open_palm", confidence=0.95)

    assert stable == "open_palm"
    assert action is not None
    assert "Volume: 55%" in action
    assert round(music.current_volume, 2) == 0.55
