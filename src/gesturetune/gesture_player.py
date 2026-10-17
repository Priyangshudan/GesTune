"""
Real-time gesture-controlled music player application.
Connects:
- Webcam feed via OpenCV
- HandTracker (MediaPipe Tasks)
- HybridGestureRecognizer (Personalized 8-gesture profile + Rule fallback)
- StabilizedGestureController (temporal smoothing + release/rearm logic)
- RotationalVolumeController (calibrated continuous rotational volume control)
- MusicController (pygame.mixer)

Interactive Controls:
- 'v': Toggle volume mode between 'Rotational (Calibrated)' and 'Discrete (Palm/Fist)'
- 'r': Launch full recalibration wizard
- 'q' or ESC: Safe shutdown
"""

import sys
import time
import logging
from pathlib import Path
import argparse
import cv2

from . import load_config
from .music_controller import MusicController
from .hand_tracker import HandTracker
from .hybrid_recognizer import HybridGestureRecognizer
from .stabilized_controller import StabilizedGestureController
from .rotational_volume import RotationalVolumeController, RotationalVolumeCalibration
from .wizard import GestureRegistrationWizard
from .modern_ui import ModernPlayerUI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_player(config_path: str = "config/config.yaml",
               recalibrate: bool = False,
               profile_path: str = "data/gesture_profiles/default.json",
               volume_mode: str = "rotational"):
    config = load_config(config_path)

    prof_file = Path(profile_path)
    wizard = GestureRegistrationWizard(config_path=config_path, profile_dir=str(prof_file.parent), profile_name=prof_file.stem)

    # 1. First-run check: if no profile exists or recalibrate requested
    if recalibrate or not wizard.has_valid_profile():
        if recalibrate:
            logger.info("Recalibration requested by user. Starting Registration Wizard...")
        else:
            logger.info("No personalized gesture profile found. Launching First-Run Registration Wizard...")
        
        wizard_success = wizard.run_wizard()
        if not wizard_success and not wizard.has_valid_profile():
            logger.warning("Registration cancelled and no fallback profile exists. Will run in rule-based mode.")

    cam_cfg = config.get("camera", {})
    device_idx = cam_cfg.get("device_index", 0)
    width = cam_cfg.get("width", 640)
    height = cam_cfg.get("height", 480)
    mirror = cam_cfg.get("mirror", True)

    mp_cfg = config.get("mediapipe", {})
    ctrl_cfg = config.get("controls", {})
    player_cfg = config.get("player", {})

    # Initialize components
    music = MusicController(
        music_dir=player_cfg.get("music_dir", "music"),
        liked_file=player_cfg.get("liked_tracks_file", "data/liked_tracks.json"),
        default_volume=ctrl_cfg.get("default_volume", 0.7),
        volume_step=ctrl_cfg.get("volume_step", 0.05)
    )

    tracker = HandTracker(
        model_path=mp_cfg.get("model_path", "models/hand_landmarker.task"),
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    recognizer = HybridGestureRecognizer(profile_path=prof_file, confidence_threshold=0.55)
    
    # Initialize rotational volume controller
    rot_calib_obj = None
    if recognizer.personalized_model and recognizer.personalized_model.rotational_calibration:
        rot_calib_obj = RotationalVolumeCalibration.from_dict(recognizer.personalized_model.rotational_calibration)

    rot_controller = RotationalVolumeController(
        calibration=rot_calib_obj,
        deadzone_deg=10.0,
        max_angle_deg=45.0,
        rate_multiplier=0.40,
        smoothing_alpha=0.30
    )

    controller = StabilizedGestureController(
        music_controller=music,
        window_size=ctrl_cfg.get("debounce_frames", 6),
        consistency_ratio=0.7,
        cooldown_seconds=ctrl_cfg.get("cooldown_seconds", 1.0),
        volume_repeat_interval=ctrl_cfg.get("volume_repeat_interval", 0.45)
    )

    # Initialize Modern UI Renderer
    ui_width = 1280
    ui_height = 720
    ui = ModernPlayerUI(target_width=ui_width, target_height=ui_height)
    window_name = "GestureTune - Intelligent Audio Interface"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, ui_width, ui_height)

    # Open webcam
    cap = cv2.VideoCapture(device_idx)
    if not cap.isOpened():
        logger.error(f"Cannot open webcam with device index {device_idx}. Exiting.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    active_vol_mode = volume_mode.lower()  # 'rotational' or 'discrete'
    if active_vol_mode == "rotational" and rot_calib_obj is None:
        logger.info("Rotational volume not calibrated in profile. Falling back to discrete palm/fist volume control.")
        active_vol_mode = "discrete"

    mode_str = "Personalized (ML)" if recognizer.is_personalized() else "Rule-based Fallback"
    logger.info(f"GestureTune Live Player started [{mode_str}]. Volume Mode: [{active_vol_mode.upper()}].")

    fps_time = time.monotonic()
    fps = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab camera frame. Retrying...")
                time.sleep(0.05)
                continue

            if mirror:
                frame = cv2.flip(frame, 1)

            # Process hand tracking
            hand_detected, landmarks, conf = tracker.process_frame(frame)

            raw_gesture = None
            pred_conf = 0.0
            source_tag = "None"
            rot_diag = {}

            if hand_detected and landmarks:
                # Draw skeleton
                frame = tracker.draw_landmarks_on_image(frame, landmarks)
                
                # Recognize gesture (instantaneous prediction)
                raw_gesture, pred_conf, source_tag = recognizer.recognize(landmarks, is_mirrored_preview=mirror)

            # Keep discrete volume suppression synced with active volume mode
            controller.suppress_palm_volume = (active_vol_mode == "rotational")

            # Run through temporal smoothing & interaction state machine
            stable_gesture, action_msg, machine_state = controller.process_prediction(raw_gesture, confidence=pred_conf)

            # Gate Rotational Volume Control:
            # Active ONLY when open_palm has been recognized as the stable gesture and hand tracking is valid
            rotation_active = (
                active_vol_mode == "rotational"
                and stable_gesture == "open_palm"
                and hand_detected
                and (landmarks is not None)
            )

            if rotation_active:
                cur_vol = music.current_volume
                new_vol, rot_diag = rot_controller.process_landmarks(landmarks, cur_vol, is_mirrored_preview=mirror)
                if rot_diag.get("active", False) and abs(new_vol - cur_vol) > 0.001:
                    music.set_volume_direct(new_vol)
                    controller.last_executed_command = f"Rotational Vol -> {int(new_vol*100)}% ({rot_diag.get('direction')})"
            else:
                # Reset transient smoothing state when not in stable open_palm or hand tracking lost
                rot_controller.reset_state()

            # FPS calculation
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, (now - fps_time)))
            fps_time = now

            # Fetch music status
            status = music.get_status()

            # Render Modern Desktop Player UI
            rendered_display = ui.render(
                camera_frame=frame,
                hand_detected=hand_detected,
                raw_gesture=raw_gesture,
                pred_conf=pred_conf,
                stable_gesture=stable_gesture,
                action_msg=action_msg or controller.last_executed_command,
                machine_state=machine_state,
                active_vol_mode=active_vol_mode,
                rotation_active=rotation_active,
                rot_diag=rot_diag,
                music_status=status,
                mode_str=mode_str,
                fps=fps
            )

            cv2.imshow(window_name, rendered_display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('v'):
                # Toggle volume mode
                if active_vol_mode == "rotational":
                    active_vol_mode = "discrete"
                    controller.suppress_palm_volume = False
                    rot_controller.reset_state()
                    logger.info("Switched Volume Mode to DISCRETE (Palm/Fist).")
                else:
                    active_vol_mode = "rotational"
                    controller.suppress_palm_volume = True
                    logger.info("Switched Volume Mode to ROTATIONAL (Calibrated).")
            elif key == ord('r'):
                # Launch recalibration wizard
                cv2.destroyAllWindows()
                cap.release()
                wizard.run_wizard()
                # Reload player
                return run_player(config_path=config_path, recalibrate=False, profile_path=profile_path, volume_mode=active_vol_mode)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("GestureTune Live Player stopped safely.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GestureTune Music Player.")
    parser.add_argument("--recalibrate", action="store_true", help="Launch gesture recalibration wizard.")
    parser.add_argument("--profile", default="data/gesture_profiles/default.json", help="Path to profile JSON.")
    parser.add_argument("--volume-mode", choices=["rotational", "discrete"], default="rotational", help="Volume adjustment mode.")
    args = parser.parse_args()

    run_player(recalibrate=args.recalibrate, profile_path=args.profile, volume_mode=args.volume_mode)
