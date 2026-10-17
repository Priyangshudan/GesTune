import pytest
import numpy as np
import time
from src.gesturetune.gesture_recognition import GestureRecognizer
from src.gesturetune.music_controller import MusicController
from src.gesturetune.gesture_controller import GestureController
from src.gesturetune.synthetic_data import generate_gesture_skeleton

def skeleton_to_dict_list(coords: np.ndarray):
    return [{"x": float(pt[0]), "y": float(pt[1]), "z": float(pt[2])} for pt in coords]

def test_all_eight_synthetic_gestures_recognized():
    recognizer = GestureRecognizer()
    gestures = ["index_finger", "two_fingers", "point_left", "point_right", "open_palm", "fist", "thumbs_up", "thumbs_down"]

    for g in gestures:
        skeleton = generate_gesture_skeleton(g)
        lm_dicts = skeleton_to_dict_list(skeleton)
        pred = recognizer.recognize(lm_dicts, is_mirrored_preview=True)
        assert pred == g, f"Expected {g}, but got {pred}"

def test_pointing_direction_mirrored():
    recognizer = GestureRecognizer()

    right_skel = generate_gesture_skeleton("point_right")
    pred_r = recognizer.recognize(skeleton_to_dict_list(right_skel), is_mirrored_preview=True)
    assert pred_r == "point_right"

    left_skel = generate_gesture_skeleton("point_left")
    pred_l = recognizer.recognize(skeleton_to_dict_list(left_skel), is_mirrored_preview=True)
    assert pred_l == "point_left"

def test_debouncing_and_discrete_cooldown():
    music = MusicController()
    controller = GestureController(music, debounce_frames=3, cooldown_seconds=0.5)

    # Frame 1: Index finger
    stable, action = controller.process_gesture("index_finger")
    assert stable is None
    assert action is None

    # Frame 2: Index finger
    stable, action = controller.process_gesture("index_finger")
    assert stable is None
    assert action is None

    # Frame 3: Threshold reached (3 consecutive frames) -> Fired
    stable, action = controller.process_gesture("index_finger")
    assert stable == "index_finger"
    assert action is not None
    assert "Playing" in action or "Resumed" in action

    # Frame 4: Continuous hold -> Should NOT refire discrete command
    stable, action = controller.process_gesture("index_finger")
    assert stable == "index_finger"
    assert action is None

def test_volume_continuous_repeat():
    music = MusicController(default_volume=0.5, volume_step=0.05)
    controller = GestureController(music, debounce_frames=2, volume_repeat_interval=0.05)

    # Reach stability
    controller.process_gesture("open_palm")
    stable, action = controller.process_gesture("open_palm")
    assert stable == "open_palm"
    assert action is not None
    assert music.current_volume == 0.55

    # Immediately process -> interval not passed
    stable, action = controller.process_gesture("open_palm")
    assert action is None

    # Wait repeat interval
    time.sleep(0.06)
    stable, action = controller.process_gesture("open_palm")
    assert action is not None
    assert music.current_volume == 0.60

