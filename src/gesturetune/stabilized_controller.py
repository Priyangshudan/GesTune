"""
Robust prediction stabilization and interaction state machine.
Features:
1. Rolling window temporal smoothing (deque of recent predictions).
2. Minimum consistent prediction ratio (e.g. 70% of window must agree).
3. Discrete action rearm state machine:
   - Firing a discrete action (play/pause, mute, track skip, like) requires transitioning
     through a RELEASE / NEUTRAL state before the same action can fire again.
   - Prevents rapid repeated toggling even if the user continuously holds the pose for seconds or minutes.
4. Cooldown timer safeguards.
5. Continuous volume hold control (repeat interval).
6. Safe tracking loss recovery.
"""

import time
import math
from collections import deque, Counter
from typing import Optional, Tuple, Dict, Any
from .music_controller import MusicController

class StabilizedGestureController:
    def __init__(self, music_controller: MusicController,
                 window_size: int = 7,
                 consistency_ratio: float = 0.7,
                 cooldown_seconds: float = 1.0,
                 volume_repeat_interval: float = 0.45,
                 suppress_palm_volume: bool = False):
        self.music = music_controller
        self.window_size = window_size
        self.consistency_ratio = consistency_ratio
        self.cooldown_seconds = cooldown_seconds
        self.volume_repeat_interval = volume_repeat_interval
        self.suppress_palm_volume = suppress_palm_volume

        # Prediction buffer for temporal smoothing
        self.prediction_window = deque(maxlen=window_size)

        # State machine
        # States: 'NEUTRAL', 'STABLE_POSE', 'COOLDOWN', 'WAITING_FOR_RELEASE'
        self.state: str = "NEUTRAL"
        self.stable_gesture: Optional[str] = None
        self.active_held_gesture: Optional[str] = None
        self.last_fired_discrete_gesture: Optional[str] = None
        self.last_command_time: float = 0.0
        self.last_volume_time: float = 0.0
        self.last_executed_command: str = "None"

    def process_prediction(self, raw_gesture: Optional[str], confidence: float = 1.0) -> Tuple[Optional[str], Optional[str], str]:
        """
        Process the frame's instantaneous raw prediction through temporal smoothing
        and interaction state machine.
        
        Returns:
            (smoothed_stable_gesture, executed_action_message, current_machine_state)
        """
        now = time.monotonic()

        # Step 1: Update sliding window
        self.prediction_window.append(raw_gesture)

        # Step 2: Determine smoothed candidate if majority threshold is met
        valid_predictions = [p for p in self.prediction_window if p is not None]
        smoothed_candidate = None

        min_required_frames = max(2, int(math.ceil(self.window_size * self.consistency_ratio)))
        if len(self.prediction_window) >= min_required_frames and len(valid_predictions) >= min_required_frames:
            counts = Counter(valid_predictions)
            most_common, freq = counts.most_common(1)[0]
            if freq >= min_required_frames:
                smoothed_candidate = most_common

        self.stable_gesture = smoothed_candidate
        action_msg = None

        # Step 3: State Machine transitions
        if smoothed_candidate is None:
            # Hand is absent or in neutral transition
            if self.state == "WAITING_FOR_RELEASE":
                # Successfully re-armed!
                self.state = "NEUTRAL"
                self.last_fired_discrete_gesture = None
            elif self.state != "COOLDOWN":
                self.state = "NEUTRAL"

            return None, None, self.state

        # Handle Discrete gestures that are held and waiting for release
        if smoothed_candidate == self.last_fired_discrete_gesture:
            self.state = "WAITING_FOR_RELEASE"
            return smoothed_candidate, None, self.state

        # Check cooldown state
        if (now - self.last_command_time) < self.cooldown_seconds:
            self.state = "COOLDOWN"
            return smoothed_candidate, None, self.state

        # Handle Volume gestures (continuous hold allowed with repeat interval)
        if smoothed_candidate in ("open_palm", "fist"):
            self.state = "STABLE_POSE"
            # If suppress_palm_volume is enabled (e.g. rotational volume mode), open_palm does not trigger discrete volume_up
            if smoothed_candidate == "open_palm" and self.suppress_palm_volume:
                return smoothed_candidate, None, self.state

            if (now - self.last_volume_time) >= self.volume_repeat_interval:
                if smoothed_candidate == "open_palm":
                    action_msg = self.music.volume_up()
                else:
                    action_msg = self.music.volume_down()
                self.last_volume_time = now
                self.last_executed_command = f"{smoothed_candidate} -> {action_msg}"
            return smoothed_candidate, action_msg, self.state

        # Ready to fire new discrete action!
        self.state = "STABLE_POSE"
        if smoothed_candidate == "index_finger":
            action_msg = self.music.toggle_play_pause()
        elif smoothed_candidate == "two_fingers":
            action_msg = self.music.toggle_mute()
        elif smoothed_candidate == "point_left":
            action_msg = self.music.prev_track()
        elif smoothed_candidate == "point_right":
            action_msg = self.music.next_track()
        elif smoothed_candidate == "thumbs_up":
            action_msg = self.music.like_current_track()
        elif smoothed_candidate == "thumbs_down":
            action_msg = self.music.unlike_current_track()

        self.last_command_time = now
        self.last_fired_discrete_gesture = smoothed_candidate
        self.state = "WAITING_FOR_RELEASE"
        self.last_executed_command = f"{smoothed_candidate} -> {action_msg}"

        return smoothed_candidate, action_msg, self.state
