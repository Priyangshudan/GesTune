"""
Stateful gesture controller that manages:
- Consecutive-frame debouncing
- Command cooldowns
- Continuous volume adjustments (repeat interval)
- Preventing accidental repeated discrete triggers
- Dispatching commands to the MusicController
"""

import time
from typing import Optional, Tuple, Dict, Any
from .music_controller import MusicController

class GestureController:
    def __init__(self, music_controller: MusicController,
                 debounce_frames: int = 5,
                 cooldown_seconds: float = 1.0,
                 volume_repeat_interval: float = 0.5):
        self.music = music_controller
        self.debounce_frames = debounce_frames
        self.cooldown_seconds = cooldown_seconds
        self.volume_repeat_interval = volume_repeat_interval

        # Tracking state
        self.current_candidate_gesture: Optional[str] = None
        self.consecutive_frames: int = 0
        self.last_executed_command: str = "None"
        self.last_command_time: float = 0.0
        self.last_volume_time: float = 0.0
        self.active_discrete_gesture: Optional[str] = None  # Prevents repeated discrete fires while held

    def process_gesture(self, detected_gesture: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        Process a detected gesture for the current frame.
        Returns:
            (stable_gesture, executed_action_message)
        """
        now = time.time()

        if detected_gesture is None:
            self.current_candidate_gesture = None
            self.consecutive_frames = 0
            self.active_discrete_gesture = None
            return None, None

        if detected_gesture == self.current_candidate_gesture:
            self.consecutive_frames += 1
        else:
            self.current_candidate_gesture = detected_gesture
            self.consecutive_frames = 1
            self.active_discrete_gesture = None

        stable_gesture = None
        action_msg = None

        if self.consecutive_frames >= self.debounce_frames:
            stable_gesture = self.current_candidate_gesture

            # Check if volume gesture (continuous repeating)
            if stable_gesture in ("open_palm", "fist"):
                if (now - self.last_volume_time) >= self.volume_repeat_interval:
                    if stable_gesture == "open_palm":
                        action_msg = self.music.volume_up()
                    else:
                        action_msg = self.music.volume_down()
                    self.last_volume_time = now
                    self.last_executed_command = f"{stable_gesture} -> {action_msg}"

            # Discrete gestures (play/pause, mute, track navigation, like)
            else:
                # Fire only once per continuous hold and respect cooldown
                if self.active_discrete_gesture != stable_gesture:
                    if (now - self.last_command_time) >= self.cooldown_seconds:
                        if stable_gesture == "index_finger":
                            action_msg = self.music.toggle_play_pause()
                        elif stable_gesture == "two_fingers":
                            action_msg = self.music.toggle_mute()
                        elif stable_gesture == "point_left":
                            action_msg = self.music.prev_track()
                        elif stable_gesture == "point_right":
                            action_msg = self.music.next_track()
                        elif stable_gesture == "thumbs_up":
                            action_msg = self.music.like_current_track()
                        elif stable_gesture == "thumbs_down":
                            action_msg = self.music.unlike_current_track()

                        self.last_command_time = now
                        self.active_discrete_gesture = stable_gesture
                        self.last_executed_command = f"{stable_gesture} -> {action_msg}"

        return stable_gesture, action_msg

