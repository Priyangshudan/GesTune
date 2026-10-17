import pytest
import numpy as np
import tempfile
import time
from pathlib import Path

from src.gesturetune.rotational_volume import (
    estimate_palm_roll_angle,
    RotationalVolumeCalibration,
    RotationalVolumeController
)
from src.gesturetune.music_controller import MusicController
from src.gesturetune.synthetic_data import generate_gesture_skeleton
from src.gesturetune.personalized_model import PersonalizedGestureModel

def skeleton_to_dict_list(coords: np.ndarray):
    return [{"x": float(pt[0]), "y": float(pt[1]), "z": float(pt[2])} for pt in coords]

def test_orientation_estimation_upright_and_tilted():
    # Base upright palm skeleton
    base_skel = generate_gesture_skeleton("open_palm")
    angle_up = estimate_palm_roll_angle(skeleton_to_dict_list(base_skel))
    assert angle_up is not None
    # Upright palm angle should be close to 0 degrees
    assert abs(angle_up) < 15.0

    # Tilt palm 30 degrees clockwise in image plane
    tilted_skel = base_skel.copy()
    theta = np.radians(30.0)
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    wrist = base_skel[0, :2]
    tilted_skel[:, :2] = wrist + (base_skel[:, :2] - wrist) @ rot_matrix.T

    angle_tilted = estimate_palm_roll_angle(skeleton_to_dict_list(tilted_skel))
    assert angle_tilted is not None
    diff = angle_tilted - angle_up
    assert abs(diff - 30.0) < 5.0

def test_orientation_estimation_translation_invariance():
    base_skel = generate_gesture_skeleton("open_palm")
    angle_base = estimate_palm_roll_angle(skeleton_to_dict_list(base_skel))

    # Translate hand across frame (+0.25 in X, -0.20 in Y)
    trans_skel = base_skel.copy()
    trans_skel[:, 0] += 0.25
    trans_skel[:, 1] -= 0.20

    angle_trans = estimate_palm_roll_angle(skeleton_to_dict_list(trans_skel))
    assert angle_trans is not None
    assert abs(angle_trans - angle_base) < 1e-4

def test_orientation_degenerate_and_missing_landmarks():
    # Empty / none
    assert estimate_palm_roll_angle(None) is None
    # All zeros
    zeros = np.zeros((21, 3))
    assert estimate_palm_roll_angle(zeros) is None
    # NaNs
    nans = np.full((21, 3), np.nan)
    assert estimate_palm_roll_angle(nans) is None

def test_rotational_volume_calibration_serialization():
    calib = RotationalVolumeCalibration(
        neutral_angle_deg=5.2,
        increase_direction=1,
        min_range_deg=25.0
    )
    d = calib.to_dict()
    assert d["neutral_angle_deg"] == 5.2
    assert d["increase_direction"] == 1

    calib_loaded = RotationalVolumeCalibration.from_dict(d)
    assert calib_loaded.neutral_angle_deg == 5.2
    assert calib_loaded.increase_direction == 1

    # Check signed rotation mapping
    # 5.2 + 20 deg = 25.2 deg -> signed_rotation = +20 deg (increase)
    assert abs(calib_loaded.get_signed_rotation(25.2) - 20.0) < 1e-3
    # 5.2 - 20 deg = -14.8 deg -> signed_rotation = -20 deg (decrease)
    assert abs(calib_loaded.get_signed_rotation(-14.8) - (-20.0)) < 1e-3

def test_rotational_volume_deadzone_and_clamping():
    calib = RotationalVolumeCalibration(neutral_angle_deg=0.0, increase_direction=1)
    controller = RotationalVolumeController(
        calibration=calib,
        deadzone_deg=10.0,
        max_angle_deg=40.0,
        rate_multiplier=1.0,
        smoothing_alpha=1.0
    )

    base_skel = generate_gesture_skeleton("open_palm")
    current_vol = 0.50

    # 1. Hand within dead zone (e.g. 5 degrees tilt)
    tilted_small = base_skel.copy()
    theta = np.radians(5.0)
    rot_m = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    tilted_small[:, :2] = base_skel[0, :2] + (base_skel[:, :2] - base_skel[0, :2]) @ rot_m.T

    new_vol, diag = controller.process_landmarks(skeleton_to_dict_list(tilted_small), current_vol)
    assert diag["active"] is False
    assert diag["direction"] == "neutral"
    assert new_vol == current_vol  # Volume unchanged in deadzone

    # 2. Hand rotated beyond deadzone (e.g. +35 degrees tilt -> volume increase)
    tilted_large = base_skel.copy()
    theta = np.radians(35.0)
    rot_m = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    tilted_large[:, :2] = base_skel[0, :2] + (base_skel[:, :2] - base_skel[0, :2]) @ rot_m.T

    # Give a synthetic time delta
    time.sleep(0.05)
    new_vol, diag = controller.process_landmarks(skeleton_to_dict_list(tilted_large), current_vol)
    assert diag["active"] is True
    assert diag["direction"] == "increase"
    assert new_vol > current_vol

    # 3. Clamping test: volume clamped to 1.0 max and 0.0 min
    vol_clamped_high, _ = controller.process_landmarks(skeleton_to_dict_list(tilted_large), 1.0)
    assert vol_clamped_high == 1.0

def test_rotational_volume_tracking_loss():
    calib = RotationalVolumeCalibration(neutral_angle_deg=0.0, increase_direction=1)
    controller = RotationalVolumeController(calibration=calib)

    # When hand is lost (None), controller safely deactivates and leaves volume unchanged
    vol, diag = controller.process_landmarks(None, 0.70)
    assert vol == 0.70
    assert diag["active"] is False
    assert diag["reason"] == "no_hand"

def test_like_unlike_state_management():
    with tempfile.TemporaryDirectory() as tmpdir:
        liked_json = Path(tmpdir) / "liked_tracks.json"
        music = MusicController(liked_file=str(liked_json))

        track_name = music.current_track_name
        status = music.get_status()
        assert status["is_liked"] is False

        # Like track
        res_like = music.like_current_track()
        assert "Liked:" in res_like
        assert music.get_status()["is_liked"] is True

        # Like again (harmless idempotent)
        res_like2 = music.like_current_track()
        assert "Already liked:" in res_like2
        assert music.get_status()["is_liked"] is True

        # Unlike track
        res_unlike = music.unlike_current_track()
        assert "Unliked:" in res_unlike
        assert music.get_status()["is_liked"] is False

        # Unlike already-unliked track (harmless)
        res_unlike2 = music.unlike_current_track()
        assert "Not liked:" in res_unlike2
        assert music.get_status()["is_liked"] is False

def test_backward_compatibility_v2_profile():
    with tempfile.TemporaryDirectory() as tmpdir:
        prof_json = Path(tmpdir) / "v2_profile.json"
        # Write legacy v2.0.0 profile with 7 gestures and no rotational calibration
        legacy_data = {
            "schema_version": "2.0.0",
            "created_at": 1700000000.0,
            "vocabulary": ["index_finger", "two_fingers", "point_left", "point_right", "open_palm", "fist", "thumbs_up"],
            "classes": ["index_finger", "two_fingers", "point_left", "point_right", "open_palm", "fist", "thumbs_up"],
            "confidence_threshold": 0.55,
            "model_file": None,
            "metadata": {}
        }
        import json
        with open(prof_json, "w", encoding="utf-8") as f:
            json.dump(legacy_data, f)

        # Loading v2.0.0 profile should succeed without crashing
        model = PersonalizedGestureModel.load_profile(prof_json)
        assert model is not None
        assert len(model.classes_) == 7
        assert model.rotational_calibration is None

