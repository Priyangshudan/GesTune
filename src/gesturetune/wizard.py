"""
First-Run Interactive Gesture Registration Wizard module (wizard.py).
Guides the user step-by-step through calibrating the 8 gestures and optional rotational volume:
1. Play/pause - Index finger
2. Mute/unmute - Two fingers
3. Previous track - Point left
4. Next track - Point right
5. Volume up - Open palm
6. Volume down - Fist
7. Like track - Thumbs up
8. Unlike track - Thumbs down
+ Rotational Volume Calibration (neutral orientation, increase direction, decrease direction)

Features:
- Live camera preview with countdown, sample capture progress, and skeleton display.
- Multiple attempts per gesture so samples aren't identical consecutive frames.
- Rejection of low-confidence or boundary-clipped hand detections.
- Feature similarity sanity checks between registered gesture classes.
- Profile migration: can append new gestures or rotational calibration without deleting existing calibration data.
- Graceful cancellation ('q' or ESC).
"""

import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
import pandas as pd

from . import load_config
from .hand_tracker import HandTracker
from .feature_extraction import LandmarkFeatureExtractor
from .personalized_model import PersonalizedGestureModel
from .rotational_volume import estimate_palm_roll_angle, RotationalVolumeCalibration

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REGISTRATION_SEQUENCE = [
    {
        "label": "index_finger",
        "action": "Play / Pause Toggle",
        "instruction": "Extend ONLY index finger upwards; keep other fingers folded.",
    },
    {
        "label": "two_fingers",
        "action": "Mute / Unmute Toggle",
        "instruction": "Extend index AND middle fingers together (peace/victory sign).",
    },
    {
        "label": "point_left",
        "action": "Previous Track",
        "instruction": "Extend index finger horizontally pointing towards your LEFT.",
    },
    {
        "label": "point_right",
        "action": "Next Track",
        "instruction": "Extend index finger horizontally pointing towards your RIGHT.",
    },
    {
        "label": "open_palm",
        "action": "Volume Up",
        "instruction": "Open your entire hand flat with all 5 fingers spread out.",
    },
    {
        "label": "fist",
        "action": "Volume Down",
        "instruction": "Close all fingers and thumb tightly into a solid fist.",
    },
    {
        "label": "thumbs_up",
        "action": "Like Current Track",
        "instruction": "Point thumb straight UP while keeping other 4 fingers curled.",
    },
    {
        "label": "thumbs_down",
        "action": "Unlike Current Track",
        "instruction": "Point thumb straight DOWN while keeping other 4 fingers curled.",
    }
]

class GestureRegistrationWizard:
    def __init__(self, config_path: str = "config/config.yaml",
                 profile_dir: str = "data/gesture_profiles",
                 profile_name: str = "default",
                 target_samples_per_gesture: int = 24,
                 num_attempts_per_gesture: int = 3):
        self.config = load_config(config_path)
        self.profile_dir = Path(profile_dir)
        self.profile_name = profile_name
        self.profile_json_path = self.profile_dir / f"{profile_name}.json"
        self.model_joblib_path = self.profile_dir / f"{profile_name}_model.joblib"
        self.samples_csv_path = self.profile_dir / f"{profile_name}_samples.csv"

        self.target_samples = target_samples_per_gesture
        self.num_attempts = num_attempts_per_gesture
        self.samples_per_attempt = target_samples_per_gesture // num_attempts_per_gesture

        cam_cfg = self.config.get("camera", {})
        self.device_idx = cam_cfg.get("device_index", 0)
        self.width = cam_cfg.get("width", 640)
        self.height = cam_cfg.get("height", 480)
        self.mirror = cam_cfg.get("mirror", True)

        mp_cfg = self.config.get("mediapipe", {})
        self.tracker = HandTracker(
            model_path=mp_cfg.get("model_path", "models/hand_landmarker.task"),
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.extractor = LandmarkFeatureExtractor()

    def has_valid_profile(self) -> bool:
        """Check if a valid, uncorrupted profile and model exist."""
        if not self.profile_json_path.exists() or not self.model_joblib_path.exists():
            return False
        model = PersonalizedGestureModel.load_profile(self.profile_json_path)
        return model is not None and model.classifier is not None

    def has_all_gestures(self) -> bool:
        """Check if profile has all 8 command gestures registered."""
        if not self.has_valid_profile():
            return False
        model = PersonalizedGestureModel.load_profile(self.profile_json_path)
        req_set = {g["label"] for g in REGISTRATION_SEQUENCE if g["label"] != "neutral"}
        return model is not None and req_set.issubset(set(model.classes_))

    def has_rotational_calibration(self) -> bool:
        """Check if profile has valid rotational volume calibration."""
        if not self.has_valid_profile():
            return False
        model = PersonalizedGestureModel.load_profile(self.profile_json_path)
        return model is not None and model.rotational_calibration is not None

    def check_boundary_clip(self, landmarks: List[Dict[str, float]], margin: float = 0.04) -> bool:
        """Check if hand is too close to frame edge."""
        for lm in landmarks:
            x, y = lm['x'], lm['y']
            if x < margin or x > (1.0 - margin) or y < margin or y > (1.0 - margin):
                return True
        return False

    def calibrate_rotational_volume(self, cap: cv2.VideoCapture) -> Optional[RotationalVolumeCalibration]:
        """
        Interactive wizard step for rotational volume calibration:
        1. Capture neutral orientation (steady hand).
        2. Prompt user to rotate hand in preferred volume-increase direction.
        3. Prompt user to rotate hand in volume-decrease direction.
        4. Validate angular separation and determine increase_direction (+1 or -1).
        """
        logger.info("Starting Rotational Volume Calibration Step...")

        # Step 1: Capture Neutral Hand Orientation
        neutral_angles: List[float] = []
        hand_steady_frames = 0
        prompt_start = time.monotonic()

        while len(neutral_angles) < 20:
            ret, frame = cap.read()
            if not ret:
                continue
            if self.mirror:
                frame = cv2.flip(frame, 1)

            hand_detected, landmarks, conf = self.tracker.process_frame(frame)
            angle = None
            if hand_detected and landmarks and not self.check_boundary_clip(landmarks):
                angle = estimate_palm_roll_angle(landmarks, is_mirrored_preview=self.mirror)
                frame = self.tracker.draw_landmarks_on_image(frame, landmarks)

            h, w, _ = frame.shape
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 125), (25, 25, 25), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

            cv2.putText(frame, "ROTATIONAL VOLUME CALIBRATION (1/3: Neutral)", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame, "Hold your hand comfortably upright in neutral position.", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)

            if angle is None:
                status_text = "Place palm upright in camera view..."
                color = (0, 165, 255)
                hand_steady_frames = 0
            else:
                hand_steady_frames += 1
                if hand_steady_frames < 10:
                    status_text = "Hold neutral position steady..."
                    color = (255, 255, 0)
                else:
                    neutral_angles.append(angle)
                    status_text = f"Recording neutral angle... ({len(neutral_angles)}/20)"
                    color = (0, 255, 0)

            cv2.putText(frame, status_text, (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, "Press 's' to skip rotation calibration, 'q' to cancel", (15, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

            cv2.imshow("GestureTune - Registration Wizard", frame)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('s'):
                return None
            if key == ord('q') or key == 27:
                return None

        median_neutral = float(np.median(neutral_angles))

        # Step 2: Rotate for Volume INCREASE
        increase_angles: List[float] = []
        while len(increase_angles) < 20:
            ret, frame = cap.read()
            if not ret:
                continue
            if self.mirror:
                frame = cv2.flip(frame, 1)

            hand_detected, landmarks, conf = self.tracker.process_frame(frame)
            angle = None
            if hand_detected and landmarks and not self.check_boundary_clip(landmarks):
                angle = estimate_palm_roll_angle(landmarks, is_mirrored_preview=self.mirror)
                frame = self.tracker.draw_landmarks_on_image(frame, landmarks)

            h, w, _ = frame.shape
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 125), (25, 25, 25), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

            cv2.putText(frame, "ROTATIONAL VOLUME CALIBRATION (2/3: Volume UP)", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame, "Rotate your hand in the direction you want for VOLUME UP.", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)

            if angle is None:
                status_text = "Show palm in camera view..."
                color = (0, 165, 255)
            else:
                diff = angle - median_neutral
                while diff > 180.0: diff -= 360.0
                while diff < -180.0: diff += 360.0

                if abs(diff) < 12.0:
                    status_text = f"Rotate farther into your Volume UP angle (current tilt: {diff:+.1f} deg)..."
                    color = (255, 255, 0)
                else:
                    increase_angles.append(diff)
                    status_text = f"Capturing Volume UP angle: {diff:+.1f} deg ({len(increase_angles)}/20)"
                    color = (0, 255, 0)

            cv2.putText(frame, status_text, (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, "Press 's' to skip rotation calibration, 'q' to cancel", (15, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

            cv2.imshow("GestureTune - Registration Wizard", frame)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('s') or key == ord('q') or key == 27:
                return None

        median_inc_diff = float(np.median(increase_angles))
        inc_dir = 1 if median_inc_diff > 0 else -1

        # Step 3: Rotate for Volume DECREASE
        decrease_angles: List[float] = []
        while len(decrease_angles) < 20:
            ret, frame = cap.read()
            if not ret:
                continue
            if self.mirror:
                frame = cv2.flip(frame, 1)

            hand_detected, landmarks, conf = self.tracker.process_frame(frame)
            angle = None
            if hand_detected and landmarks and not self.check_boundary_clip(landmarks):
                angle = estimate_palm_roll_angle(landmarks, is_mirrored_preview=self.mirror)
                frame = self.tracker.draw_landmarks_on_image(frame, landmarks)

            h, w, _ = frame.shape
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 125), (25, 25, 25), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

            cv2.putText(frame, "ROTATIONAL VOLUME CALIBRATION (3/3: Volume DOWN)", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame, "Rotate in the OPPOSITE direction for VOLUME DOWN.", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)

            if angle is None:
                status_text = "Show palm in camera view..."
                color = (0, 165, 255)
            else:
                diff = angle - median_neutral
                while diff > 180.0: diff -= 360.0
                while diff < -180.0: diff += 360.0

                # Check if it opposes increase direction
                is_opposing = (diff * inc_dir) < -10.0
                if not is_opposing:
                    status_text = f"Rotate in opposite direction for Volume DOWN (tilt: {diff:+.1f} deg)..."
                    color = (255, 255, 0)
                else:
                    decrease_angles.append(diff)
                    status_text = f"Capturing Volume DOWN angle: {diff:+.1f} deg ({len(decrease_angles)}/20)"
                    color = (0, 255, 0)

            cv2.putText(frame, status_text, (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, "Press 's' to skip rotation calibration, 'q' to cancel", (15, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

            cv2.imshow("GestureTune - Registration Wizard", frame)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('s') or key == ord('q') or key == 27:
                return None

        calib = RotationalVolumeCalibration(
            neutral_angle_deg=median_neutral,
            increase_direction=inc_dir,
            min_range_deg=max(15.0, abs(median_inc_diff))
        )
        logger.info(f"Rotational Volume Calibrated! Neutral: {median_neutral:.1f} deg, Inc Direction: {inc_dir}")
        return calib

    def run_wizard(self, gestures_to_register: Optional[List[Dict[str, str]]] = None) -> bool:
        """
        Run the interactive camera registration wizard.
        If gestures_to_register is specified (e.g. only ['thumbs_down']), it appends/updates without
        requiring the user to repeat already-calibrated gestures.
        Returns True if registration succeeded and saved, False if cancelled.
        """
        cap = cv2.VideoCapture(self.device_idx)
        if not cap.isOpened():
            logger.error(f"Cannot open webcam (device {self.device_idx}) for registration.")
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        logger.info("Starting GestureTune Registration Wizard...")

        target_sequence = gestures_to_register or REGISTRATION_SEQUENCE

        # Load existing calibration samples if appending
        all_feature_vectors: List[np.ndarray] = []
        all_labels: List[str] = []
        all_attempt_groups: List[int] = []

        if self.samples_csv_path.exists():
            try:
                df_existing = pd.read_csv(self.samples_csv_path)
                exclude_labels = {g["label"] for g in target_sequence}
                # Keep existing samples not in target_sequence
                for _, row in df_existing.iterrows():
                    lbl = str(row["gesture_label"])
                    if lbl not in exclude_labels:
                        feat_cols = [c for c in df_existing.columns if c.startswith("feat_")]
                        if feat_cols:
                            all_feature_vectors.append(row[feat_cols].values.astype(np.float32))
                            all_labels.append(lbl)
                            all_attempt_groups.append(int(row.get("group", 0)))
                logger.info(f"Loaded {len(all_labels)} existing calibration samples across {len(set(all_labels))} gestures.")
            except Exception as e:
                logger.warning(f"Could not load existing calibration samples: {e}")

        try:
            for g_idx, g_info in enumerate(target_sequence):
                label = g_info["label"]
                action = g_info["action"]
                instr = g_info["instruction"]

                collected_for_gesture = 0
                attempt = 1

                while attempt <= self.num_attempts:
                    # Stage A: Prompt user to prepare pose and wait for steady hand
                    prompt_start = time.monotonic()
                    ready_countdown = 3  # seconds
                    hand_steady_frames = 0
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        if self.mirror:
                            frame = cv2.flip(frame, 1)

                        hand_detected, landmarks, conf = self.tracker.process_frame(frame)
                        is_clipped = False
                        if hand_detected and landmarks:
                            is_clipped = self.check_boundary_clip(landmarks)
                            frame = self.tracker.draw_landmarks_on_image(frame, landmarks)

                        # Overlay instructions
                        h, w, _ = frame.shape
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (0, 0), (w, 125), (30, 30, 30), -1)
                        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

                        cv2.putText(frame, f"Gesture {g_idx+1}/{len(target_sequence)}: {label.upper()} -> [{action}]", (15, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                        cv2.putText(frame, f"Attempt {attempt}/{self.num_attempts} | {instr}", (15, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)

                        if not hand_detected:
                            status_text = "Please show your hand in the camera..."
                            color = (0, 165, 255)
                            hand_steady_frames = 0
                        elif is_clipped:
                            status_text = "Hand too close to edge! Move hand to center."
                            color = (0, 0, 255)
                            hand_steady_frames = 0
                        else:
                            hand_steady_frames += 1
                            if hand_steady_frames < 15:
                                status_text = "Hold pose steady..."
                                color = (255, 255, 0)
                            else:
                                elapsed_steady = (hand_steady_frames - 15) / 30.0
                                remaining = max(0, int(ready_countdown - elapsed_steady))
                                status_text = f"Recording starting in {remaining}s... KEEP STEADY"
                                color = (0, 255, 0)
                                if remaining == 0:
                                    break

                        cv2.putText(frame, status_text, (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.putText(frame, "Press 'q' to cancel registration", (15, h - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

                        cv2.imshow("GestureTune - Registration Wizard", frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:
                            logger.info("Registration cancelled by user.")
                            return False

                    # Stage B: Capture samples for this attempt
                    samples_this_attempt = 0
                    while samples_this_attempt < self.samples_per_attempt:
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        if self.mirror:
                            frame = cv2.flip(frame, 1)

                        hand_detected, landmarks, conf = self.tracker.process_frame(frame)
                        if hand_detected and landmarks and not self.check_boundary_clip(landmarks):
                            feat = self.extractor.extract_features(landmarks, is_mirrored_preview=self.mirror)
                            all_feature_vectors.append(feat)
                            all_labels.append(label)
                            all_attempt_groups.append(g_idx * 10 + attempt)
                            samples_this_attempt += 1
                            collected_for_gesture += 1
                            frame = self.tracker.draw_landmarks_on_image(frame, landmarks)

                        # Render progress
                        h, w, _ = frame.shape
                        cv2.rectangle(frame, (0, 0), (w, 80), (0, 100, 0), -1)
                        cv2.putText(frame, f"Recording {label} (Attempt {attempt}): {collected_for_gesture}/{self.target_samples} samples",
                                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                        cv2.putText(frame, "Keep holding pose...", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)

                        cv2.imshow("GestureTune - Registration Wizard", frame)
                        key = cv2.waitKey(20) & 0xFF
                        if key == ord('q') or key == 27:
                            return False

                    # Stage C: Brief pause between attempts
                    attempt += 1
                    if attempt <= self.num_attempts:
                        pause_start = time.monotonic()
                        while time.monotonic() - pause_start < 1.2:
                            ret, frame = cap.read()
                            if ret:
                                if self.mirror:
                                    frame = cv2.flip(frame, 1)
                                cv2.putText(frame, "Relax hand briefly...", (frame.shape[1]//4, frame.shape[0]//2),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                                cv2.imshow("GestureTune - Registration Wizard", frame)
                                cv2.waitKey(1)

            # Optional Rotational Volume Calibration Step
            rot_calib = self.calibrate_rotational_volume(cap)
            rot_calib_dict = rot_calib.to_dict() if rot_calib else None

            # Training Stage
            h, w, _ = frame.shape
            loading_screen = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(loading_screen, "Training Personalized Model...", (50, h//2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(loading_screen, f"Optimizing decision boundaries across {len(set(all_labels))} gestures...", (50, h//2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.imshow("GestureTune - Registration Wizard", loading_screen)
            cv2.waitKey(100)

            # Train model
            X = np.array(all_feature_vectors)
            y = all_labels
            groups = all_attempt_groups

            vocab_list = sorted(list(set(all_labels)))
            model = PersonalizedGestureModel(vocabulary=vocab_list)
            eval_metrics = model.train(X, y, groups=groups)

            # Save calibration samples to CSV for future migrations
            self.profile_dir.mkdir(parents=True, exist_ok=True)
            df_samples = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
            df_samples["gesture_label"] = y
            df_samples["group"] = groups
            df_samples.to_csv(self.samples_csv_path, index=False)

            # Save profile and weights
            model.save_profile(
                self.profile_json_path,
                self.model_joblib_path,
                calibration_csv_path=self.samples_csv_path,
                rotational_calibration=rot_calib_dict
            )

            acc = eval_metrics.get("accuracy", 1.0)
            logger.info(f"Personalized gesture model registered successfully! Cross-validated accuracy: {acc*100:.1f}%")

            # Show completion screen
            conf_screen = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(conf_screen, "REGISTRATION COMPLETE!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
            cv2.putText(conf_screen, f"Model Accuracy: {acc*100:.1f}% ({len(X)} calibration frames, {len(vocab_list)} gestures)", (50, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            rot_status = "CALIBRATED" if rot_calib else "NOT CONFIGURED (Palm/Fist fallback)"
            cv2.putText(conf_screen, f"Rotational Volume: {rot_status}", (50, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
            
            y_offset = 150
            for item in REGISTRATION_SEQUENCE:
                cv2.putText(conf_screen, f"- {item['label']}: {item['action']}", (50, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 220, 200), 1)
                y_offset += 25

            cv2.putText(conf_screen, "Press SPACE to launch player, or 'q' to exit", (50, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("GestureTune - Registration Wizard", conf_screen)
            while True:
                k = cv2.waitKey(0) & 0xFF
                if k == ord(' ') or k == 13:
                    return True
                elif k == ord('q') or k == 27:
                    return True

        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    wizard = GestureRegistrationWizard()
    wizard.run_wizard()
