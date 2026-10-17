"""
Music player controller supporting local playback, volume, mute, track navigation, and track liking.
Uses pygame.mixer for cross-platform audio playback, with graceful fallback to simulated demo mode
if no audio device or music files are present.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

class MusicController:
    def __init__(self, music_dir: str = "music", liked_file: str = "data/liked_tracks.json",
                 default_volume: float = 0.7, volume_step: float = 0.05):
        self.music_dir = Path(music_dir)
        self.liked_file = Path(liked_file)
        self.default_volume = max(0.0, min(1.0, default_volume))
        self.volume_step = volume_step
        self.current_volume = self.default_volume
        self.is_muted = False
        self.unmuted_volume = self.default_volume
        self.is_playing = False
        self.is_paused = False
        self.playlist: List[Path] = []
        self.current_index = 0
        self.audio_available = False
        self.liked_tracks: List[str] = []

        self._init_audio()
        self._load_playlist()
        self._load_liked_tracks()

    def _init_audio(self):
        try:
            import pygame
            # Initialize mixer if not already initialized
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.set_volume(self.current_volume)
            self.audio_available = True
            logger.info("pygame.mixer initialized successfully.")
        except Exception as e:
            logger.warning(f"Audio device not available or pygame.mixer failed ({e}). Running in simulation mode.")
            self.audio_available = False

    def _load_playlist(self):
        self.playlist = []
        if self.music_dir.exists():
            for ext in (".mp3", ".wav", ".ogg"):
                self.playlist.extend(sorted(self.music_dir.glob(f"*{ext}")))
                self.playlist.extend(sorted(self.music_dir.glob(f"*{ext.upper()}")))
        
        # Remove duplicates preserving order
        seen = set()
        unique_playlist = []
        for p in self.playlist:
            if p.name not in seen:
                seen.add(p.name)
                unique_playlist.append(p)
        self.playlist = unique_playlist

        if not self.playlist:
            logger.info(f"No audio files found in '{self.music_dir}'. Player will operate with virtual demo tracks.")
            # Virtual tracks for demonstration
            self.virtual_playlist = ["Demo Track 1 - Acoustic Groove", "Demo Track 2 - Ambient Waves", "Demo Track 3 - Electronic Pulse"]
        else:
            self.virtual_playlist = []

    def _load_liked_tracks(self):
        if self.liked_file.exists():
            try:
                with open(self.liked_file, "r", encoding="utf-8") as f:
                    self.liked_tracks = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read liked tracks: {e}")
                self.liked_tracks = []
        else:
            self.liked_tracks = []

    def _save_liked_tracks(self):
        self.liked_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.liked_file, "w", encoding="utf-8") as f:
                json.dump(self.liked_tracks, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save liked tracks: {e}")

    @property
    def current_track_name(self) -> str:
        if self.playlist:
            return self.playlist[self.current_index].name
        if self.virtual_playlist:
            return self.virtual_playlist[self.current_index % len(self.virtual_playlist)]
        return "No Tracks Available"

    def play(self) -> str:
        if self.is_paused:
            if self.audio_available:
                import pygame
                pygame.mixer.music.unpause()
            self.is_paused = False
            self.is_playing = True
            return f"Resumed: {self.current_track_name}"
        
        if self.playlist and self.audio_available:
            import pygame
            track = str(self.playlist[self.current_index])
            try:
                pygame.mixer.music.load(track)
                pygame.mixer.music.play()
            except Exception as e:
                logger.error(f"Error playing {track}: {e}")
        self.is_playing = True
        self.is_paused = False
        return f"Playing: {self.current_track_name}"

    def pause(self) -> str:
        if self.is_playing and not self.is_paused:
            if self.audio_available:
                import pygame
                pygame.mixer.music.pause()
            self.is_paused = True
            self.is_playing = False
            return f"Paused: {self.current_track_name}"
        return "Already paused"

    def toggle_play_pause(self) -> str:
        if self.is_playing:
            return self.pause()
        else:
            return self.play()

    def next_track(self) -> str:
        total = len(self.playlist) if self.playlist else len(self.virtual_playlist)
        if total == 0:
            return "No tracks"
        self.current_index = (self.current_index + 1) % total
        self.is_paused = False
        return self.play()

    def prev_track(self) -> str:
        total = len(self.playlist) if self.playlist else len(self.virtual_playlist)
        if total == 0:
            return "No tracks"
        self.current_index = (self.current_index - 1 + total) % total
        self.is_paused = False
        return self.play()

    def volume_up(self) -> str:
        if self.is_muted:
            self.is_muted = False
            self.current_volume = self.unmuted_volume
        self.current_volume = round(min(1.0, self.current_volume + self.volume_step), 2)
        if self.audio_available:
            import pygame
            pygame.mixer.music.set_volume(self.current_volume)
        return f"Volume: {int(self.current_volume * 100)}%"

    def volume_down(self) -> str:
        if self.is_muted:
            self.is_muted = False
            self.current_volume = self.unmuted_volume
        self.current_volume = round(max(0.0, self.current_volume - self.volume_step), 2)
        if self.audio_available:
            import pygame
            pygame.mixer.music.set_volume(self.current_volume)
        return f"Volume: {int(self.current_volume * 100)}%"

    def toggle_mute(self) -> str:
        if self.is_muted:
            self.is_muted = False
            self.current_volume = self.unmuted_volume if self.unmuted_volume > 0 else self.default_volume
            if self.audio_available:
                import pygame
                pygame.mixer.music.set_volume(self.current_volume)
            return f"Unmuted: {int(self.current_volume * 100)}%"
        else:
            self.is_muted = True
            self.unmuted_volume = self.current_volume
            self.current_volume = 0.0
            if self.audio_available:
                import pygame
                pygame.mixer.music.set_volume(0.0)
            return "Muted"

    def like_current_track(self) -> str:
        name = self.current_track_name
        if name not in self.liked_tracks:
            self.liked_tracks.append(name)
            self._save_liked_tracks()
            return f"Liked: {name}"
        else:
            return f"Already liked: {name}"

    def unlike_current_track(self) -> str:
        name = self.current_track_name
        if name in self.liked_tracks:
            self.liked_tracks.remove(name)
            self._save_liked_tracks()
            return f"Unliked: {name}"
        else:
            return f"Not liked: {name}"

    def set_volume_direct(self, target_volume: float) -> str:
        """Set volume directly to a clamped float [0.0, 1.0]. Used by continuous rotational control."""
        self.is_muted = False
        self.current_volume = round(float(np.clip(target_volume, 0.0, 1.0)), 3)
        if self.audio_available:
            import pygame
            pygame.mixer.music.set_volume(self.current_volume)
        return f"Volume: {int(self.current_volume * 100)}%"

    def get_status(self) -> Dict[str, Any]:
        return {
            "track": self.current_track_name,
            "is_playing": self.is_playing,
            "is_paused": self.is_paused,
            "volume": int(self.current_volume * 100),
            "is_muted": self.is_muted,
            "is_liked": self.current_track_name in self.liked_tracks,
            "audio_available": self.audio_available,
            "playlist_size": len(self.playlist) if self.playlist else len(self.virtual_playlist)
        }

