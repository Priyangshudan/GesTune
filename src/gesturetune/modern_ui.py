"""
Futuristic, high-resolution desktop UI layout and renderer for GestureTune.

Visual Concept:
- Deep Midnight Canvas: #090B10 (RGB: 9, 11, 16 / BGR: 16, 11, 9)
- Surface Cards / Panels: #10141E (RGB: 16, 20, 30 / BGR: 30, 20, 16)
- Elevated Sub-Panels: #161C2C (RGB: 22, 28, 44 / BGR: 44, 28, 22)
- Borders / Subtle Grid: #1E293B (RGB: 30, 41, 59 / BGR: 59, 41, 30)
- Electric Cyan (Primary Accent / Telemetry): #00F0FF (RGB: 0, 240, 255)
- Neon Violet / Magenta (Secondary Accent / Liked): #B026FF (RGB: 176, 38, 255)
- Emerald Plasma (Active / Playing): #00FFA3 (RGB: 0, 255, 163)
- Amber Warning: #FFB300 (RGB: 255, 179, 0)
- Crimson Alert: #FF4B4B (RGB: 255, 75, 75)
- Crisp Pure White: #FFFFFF (RGB: 255, 255, 255)
- Cool Slate Gray: #94A3B8 (RGB: 148, 163, 184)
- Dim Slate Gray: #64748B (RGB: 100, 116, 139)

Architecture & Performance:
- 100% pure Pillow RGB anti-aliased vector rendering.
- Exact RGB blending for translucent badges and chips (no color-masking bugs).
- No back-and-forth OpenCV conversions mid-frame.
- Clean bilinear aspect-ratio camera compositing inside a dedicated VISION INPUT monitor.
- Segoe UI TrueType typography with fallback caching.
- Windows Per-Monitor High-DPI awareness activation.
- Fully compatible with existing test suite and gesture player interface.
"""

from typing import Dict, Any, Optional, Tuple, List
import os
import sys
import time
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Set Windows Per-Monitor High-DPI Awareness on Windows OS
if sys.platform.startswith("win"):
    try:
        import ctypes
        # Try Per-Monitor DPI awareness v2 (value 2), fallback to v1 (value 1)
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

# High-Tech Futuristic RGB Palette
RGB_BG = (9, 11, 16)                # Deep midnight
RGB_PANEL = (16, 20, 30)            # Graphite / Navy surface
RGB_PANEL_ALT = (22, 28, 44)        # Elevated card surface
RGB_BORDER = (30, 41, 59)           # Sleek border
RGB_CYAN = (0, 240, 255)            # Electric Cyan
RGB_VIOLET = (176, 38, 255)         # Neon Violet
RGB_GREEN = (0, 255, 163)           # Emerald Plasma
RGB_AMBER = (255, 179, 0)           # Amber alert
RGB_RED = (255, 75, 75)             # Crimson alert
RGB_TEXT = (255, 255, 255)          # Crisp white
RGB_MUTED = (148, 163, 184)         # Cool gray
RGB_DIM = (100, 116, 139)           # Muted slate

def blend_rgb(fg: Tuple[int, int, int], bg: Tuple[int, int, int], alpha: float) -> Tuple[int, int, int]:
    """Calculates alpha-blended RGB color over a background surface."""
    return tuple(int(fg[i] * alpha + bg[i] * (1.0 - alpha)) for i in range(3))

class FontManager:
    """Loads and caches TrueType fonts with fallbacks."""
    def __init__(self):
        self._fonts: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}
        self.font_candidates = [
            ("segoeui.ttf", "segoeuib.ttf"),
            ("arial.ttf", "arialbd.ttf"),
            ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")
        ]
        self.regular_path, self.bold_path = self._find_best_font()

    def _find_best_font(self) -> Tuple[Optional[str], Optional[str]]:
        windir = os.environ.get("WINDIR", "C:\\Windows")
        fonts_dir = Path(windir) / "Fonts"

        for reg, bld in self.font_candidates:
            p_reg = fonts_dir / reg
            p_bld = fonts_dir / bld
            if p_reg.exists():
                return str(p_reg), str(p_bld) if p_bld.exists() else str(p_reg)
        return None, None

    def get_font(self, size: int = 14, bold: bool = False) -> ImageFont.FreeTypeFont:
        key = ("bold" if bold else "regular", size)
        if key in self._fonts:
            return self._fonts[key]

        path = self.bold_path if bold else self.regular_path
        if path:
            try:
                font = ImageFont.truetype(path, size)
                self._fonts[key] = font
                return font
            except Exception:
                pass

        font = ImageFont.load_default()
        self._fonts[key] = font
        return font

_font_manager = FontManager()

class ModernPlayerUI:
    """
    Renders the futuristic, high-resolution desktop music player interface for GestureTune.
    Delivers crisp typography, smooth panels, and high-tech telemetry.
    """

    def __init__(self, target_width: int = 1280, target_height: int = 720):
        self.width = target_width
        self.height = target_height
        self.last_action_timestamp = 0.0
        self.last_action_text = "Ready"

    def update_action(self, action_text: Optional[str]):
        if action_text and action_text != "None":
            self.last_action_text = action_text
            self.last_action_timestamp = time.monotonic()

    def render(self,
               camera_frame: np.ndarray,
               hand_detected: bool,
               raw_gesture: Optional[str],
               pred_conf: float,
               stable_gesture: Optional[str],
               action_msg: Optional[str],
               machine_state: str,
               active_vol_mode: str,
               rotation_active: bool,
               rot_diag: Dict[str, Any],
               music_status: Dict[str, Any],
               mode_str: str,
               fps: float) -> np.ndarray:
        """
        Produce a crisp futuristic frame combining camera feed, now-playing card,
        diagnostics chips, and dynamic reference footer.
        """
        if action_msg:
            self.update_action(action_msg)

        # 1. Create a pure PIL RGB high-resolution canvas
        pil_canvas = Image.new("RGB", (self.width, self.height), RGB_BG)
        draw = ImageDraw.Draw(pil_canvas)

        # Layout boundaries
        pad = 20
        header_h = 58
        footer_h = 92
        status_bar_h = 56

        main_y = header_h + pad
        main_h = self.height - header_h - footer_h - status_bar_h - (pad * 3)

        # 2. Render Header
        self._render_header(draw, pad, header_h, hand_detected, mode_str, fps)

        # 3. Two-Column Main Content
        col_gap = 18
        left_w = int((self.width - (2 * pad) - col_gap) * 0.56)
        right_w = self.width - (2 * pad) - col_gap - left_w

        left_x = pad
        right_x = left_x + left_w + col_gap

        # Render Left Column: Camera Monitor
        self._render_camera_panel(pil_canvas, draw, camera_frame, hand_detected,
                                  (left_x, main_y, left_w, main_h))

        # Render Right Column: Now Playing & Telemetry Card
        self._render_now_playing_panel(draw,
                                       (right_x, main_y, right_w, main_h),
                                       music_status, active_vol_mode,
                                       rotation_active, rot_diag)

        # 4. Status Bar (Below main content)
        status_y = main_y + main_h + 14
        self._render_gesture_status_bar(draw,
                                        (pad, status_y, self.width - 2 * pad, status_bar_h),
                                        raw_gesture, pred_conf,
                                        stable_gesture, machine_state,
                                        active_vol_mode, rotation_active, rot_diag,
                                        hand_detected=hand_detected)

        # 5. Footer (Gesture Reference & Keyboard shortcuts)
        footer_y = status_y + status_bar_h + 12
        self._render_footer(draw,
                            (pad, footer_y, self.width - 2 * pad, footer_h),
                            active_vol_mode)

        # Convert final RGB image to OpenCV BGR numpy array
        final_bgr = cv2.cvtColor(np.array(pil_canvas), cv2.COLOR_RGB2BGR)
        return final_bgr

    def _render_header(self, draw: ImageDraw.Draw, pad: int, h: int,
                       hand_detected: bool, mode_str: str, fps: float):
        """Crisp futuristic header with brand mark, subtitle, camera indicator, and recognition mode."""
        font_brand = _font_manager.get_font(21, bold=True)
        font_sub = _font_manager.get_font(11, bold=True)
        font_pill = _font_manager.get_font(11, bold=True)

        y_center = h // 2

        # Brand Accent Dot + Logo
        dot_r = 5
        draw.ellipse([pad, y_center - dot_r, pad + 2 * dot_r, y_center + dot_r], fill=RGB_CYAN)
        draw.text((pad + 18, y_center - 15), "GestureTune", fill=RGB_TEXT, font=font_brand)
        
        # Futuristic Subtitle Tag
        sub_x = pad + 175
        draw.rounded_rectangle([sub_x, y_center - 11, sub_x + 195, y_center + 11],
                               radius=4, fill=RGB_PANEL_ALT, outline=RGB_BORDER, width=1)
        draw.text((sub_x + 10, y_center - 7), "INTELLIGENT AUDIO INTERFACE", fill=RGB_CYAN, font=font_sub)

        # Right side telemetry pills
        cur_x = self.width - pad

        # 1. Camera Status Pill
        cam_text = "OPTICAL TRACK: ACTIVE" if hand_detected else "OPTICAL TRACK: SEARCHING"
        cam_col = RGB_GREEN if hand_detected else RGB_AMBER
        cam_bg = blend_rgb(cam_col, RGB_BG, 0.15)
        pill_w, pill_h = 175, 26
        cur_x -= pill_w
        pill_y = y_center - pill_h // 2
        draw.rounded_rectangle([cur_x, pill_y, cur_x + pill_w, pill_y + pill_h],
                               radius=6, fill=cam_bg, outline=cam_col, width=1)
        draw.text((cur_x + 12, pill_y + 5), cam_text, fill=cam_col, font=font_pill)

        # 2. Recognition Mode Pill
        mode_text = "MODE: PERSONALIZED" if "Personalized" in mode_str else "MODE: RULE FALLBACK"
        mode_col = RGB_VIOLET if "Personalized" in mode_str else RGB_CYAN
        mode_bg = blend_rgb(mode_col, RGB_BG, 0.15)
        cur_x -= (155 + 10)
        draw.rounded_rectangle([cur_x, pill_y, cur_x + 155, pill_y + pill_h],
                               radius=6, fill=mode_bg, outline=mode_col, width=1)
        draw.text((cur_x + 12, pill_y + 5), mode_text, fill=mode_col, font=font_pill)

        # 3. FPS counter
        fps_text = f"{int(fps)} FPS"
        cur_x -= (65 + 10)
        draw.text((cur_x, pill_y + 5), fps_text, fill=RGB_MUTED, font=font_pill)

        # Bottom hairline separator with subtle cyan gradient feel
        draw.line([pad, h, self.width - pad, h], fill=RGB_BORDER, width=1)

    def _render_camera_panel(self, pil_canvas: Image.Image, draw: ImageDraw.Draw,
                             camera_frame: np.ndarray, hand_detected: bool,
                             rect: Tuple[int, int, int, int]):
        """Renders the webcam feed in the left panel with crisp anti-aliased HUD lines."""
        x, y, w, h = rect

        # Draw outer container card with rounded corners
        draw.rounded_rectangle([x, y, x + w, y + h], radius=10, fill=RGB_PANEL, outline=RGB_BORDER, width=1)

        # Panel title
        font_panel_lbl = _font_manager.get_font(11, bold=True)
        draw.text((x + 18, y + 12), "VISION INPUT", fill=RGB_CYAN, font=font_panel_lbl)

        # Inner display box
        box_pad = 12
        box_top = y + 34
        box_x = x + box_pad
        box_y = box_top
        box_w = w - (2 * box_pad)
        box_h = h - 34 - box_pad

        # Inner monitor background
        draw.rounded_rectangle([box_x, box_y, box_x + box_w, box_y + box_h],
                               radius=6, fill=(6, 8, 12), outline=(24, 32, 48), width=1)

        # Fit camera frame preserving aspect ratio
        cam_h, cam_w = camera_frame.shape[:2]
        if cam_w > 0 and cam_h > 0 and box_w > 0 and box_h > 0:
            scale = min(box_w / cam_w, box_h / cam_h)
            nw = int(cam_w * scale)
            nh = int(cam_h * scale)

            # High-quality bilinear resize for smooth video scaling
            resized_cam = cv2.resize(camera_frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            
            # Convert camera to PIL RGB
            cam_pil = Image.fromarray(cv2.cvtColor(resized_cam, cv2.COLOR_BGR2RGB))

            # Offset
            offset_x = box_x + (box_w - nw) // 2
            offset_y = box_y + (box_h - nh) // 2

            # Paste directly onto canvas
            pil_canvas.paste(cam_pil, (offset_x, offset_y))

            # Draw sci-fi corner brackets over camera view
            cw = 16
            bracket_col = RGB_CYAN if hand_detected else RGB_BORDER
            # Top-left
            draw.line([offset_x, offset_y, offset_x + cw, offset_y], fill=bracket_col, width=2)
            draw.line([offset_x, offset_y, offset_x, offset_y + cw], fill=bracket_col, width=2)
            # Top-right
            draw.line([offset_x + nw - 1, offset_y, offset_x + nw - cw - 1, offset_y], fill=bracket_col, width=2)
            draw.line([offset_x + nw - 1, offset_y, offset_x + nw - 1, offset_y + cw], fill=bracket_col, width=2)
            # Bottom-left
            draw.line([offset_x, offset_y + nh - 1, offset_x + cw, offset_y + nh - 1], fill=bracket_col, width=2)
            draw.line([offset_x, offset_y + nh - 1, offset_x, offset_y + nh - cw - 1], fill=bracket_col, width=2)
            # Bottom-right
            draw.line([offset_x + nw - 1, offset_y + nh - 1, offset_x + nw - cw - 1, offset_y + nh - 1], fill=bracket_col, width=2)
            draw.line([offset_x + nw - 1, offset_y + nh - 1, offset_x + nw - 1, offset_y + nh - cw - 1], fill=bracket_col, width=2)

        # If no hand detected, show posture guidance pill
        if not hand_detected:
            guide_w, guide_h = 320, 30
            gx = box_x + (box_w - guide_w) // 2
            gy = box_y + box_h - guide_h - 18
            guide_bg = blend_rgb(RGB_PANEL_ALT, (6, 8, 12), 0.85)
            draw.rounded_rectangle([gx, gy, gx + guide_w, gy + guide_h], radius=6,
                                   fill=guide_bg, outline=RGB_BORDER, width=1)
            font_guide = _font_manager.get_font(11, bold=False)
            draw.text((gx + 18, gy + 7), "Present hand to camera to activate gestures", fill=RGB_TEXT, font=font_guide)

    def _render_now_playing_panel(self, draw: ImageDraw.Draw,
                                  rect: Tuple[int, int, int, int],
                                  music_status: Dict[str, Any],
                                  active_vol_mode: str,
                                  rotation_active: bool,
                                  rot_diag: Dict[str, Any]):
        """Right panel: Futuristic Now Playing card, volume bar, and rotational dial."""
        x, y, w, h = rect
        draw.rounded_rectangle([x, y, x + w, y + h], radius=10, fill=RGB_PANEL, outline=RGB_BORDER, width=1)

        # Section Tag
        font_tag = _font_manager.get_font(11, bold=True)
        draw.text((x + 20, y + 14), "NOW PLAYING", fill=RGB_CYAN, font=font_tag)

        # Clean track title (strip extension like .mp3)
        raw_track = music_status.get("track", "Unknown Track")
        clean_track = Path(raw_track).stem if raw_track.endswith((".mp3", ".wav", ".ogg")) else raw_track
        if len(clean_track) > 28:
            clean_track = clean_track[:25] + "..."

        font_track = _font_manager.get_font(20, bold=True)
        draw.text((x + 20, y + 34), clean_track, fill=RGB_TEXT, font=font_track)

        # Status badge & Like badge
        is_playing = music_status.get("is_playing", False)
        is_paused = music_status.get("is_paused", False)
        is_liked = music_status.get("is_liked", False)

        status_text = "PLAYING" if is_playing else ("PAUSED" if is_paused else "STOPPED")
        status_col = RGB_GREEN if is_playing else (RGB_AMBER if is_paused else RGB_DIM)

        font_badge = _font_manager.get_font(11, bold=True)
        badge_y = y + 70
        # Play status chip
        status_bg = blend_rgb(status_col, RGB_PANEL, 0.15)
        draw.rounded_rectangle([x + 20, badge_y, x + 20 + 82, badge_y + 22], radius=4,
                               fill=status_bg, outline=status_col, width=1)
        draw.text((x + 30, badge_y + 4), status_text, fill=status_col, font=font_badge)

        # Liked badge
        if is_liked:
            liked_bg = blend_rgb(RGB_VIOLET, RGB_PANEL, 0.15)
            draw.rounded_rectangle([x + 112, badge_y, x + 112 + 76, badge_y + 22], radius=4,
                                   fill=liked_bg, outline=RGB_VIOLET, width=1)
            draw.text((x + 124, badge_y + 4), "LIKED", fill=RGB_VIOLET, font=font_badge)

        # Volume Section Card
        vol_box_y = y + 108
        vol_box_h = 96
        draw.rounded_rectangle([x + 20, vol_box_y, x + w - 20, vol_box_y + vol_box_h],
                               radius=6, fill=RGB_PANEL_ALT, outline=RGB_BORDER, width=1)

        font_card_head = _font_manager.get_font(11, bold=True)
        draw.text((x + 34, vol_box_y + 12), "VOLUME CONTROL", fill=RGB_MUTED, font=font_card_head)

        # Volume mode chip on right side of card
        mode_chip_text = f"MODE: {active_vol_mode.upper()}"
        mode_chip_col = RGB_CYAN if active_vol_mode == "rotational" else RGB_MUTED
        mode_chip_bg = blend_rgb(mode_chip_col, RGB_PANEL_ALT, 0.15)
        draw.rounded_rectangle([x + w - 162, vol_box_y + 10, x + w - 34, vol_box_y + 30],
                               radius=4, fill=mode_chip_bg,
                               outline=mode_chip_col, width=1)
        draw.text((x + w - 152, vol_box_y + 13), mode_chip_text, fill=mode_chip_col, font=font_badge)

        # Volume Bar
        vol_pct = music_status.get("volume", 70)
        is_muted = music_status.get("is_muted", False)
        display_vol = 0 if is_muted else vol_pct

        bar_x = x + 34
        bar_y = vol_box_y + 50
        bar_w = w - 130
        bar_h = 10

        # Bar background track
        draw.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], radius=5, fill=(40, 50, 70))
        # Filled track
        fill_w = int(bar_w * (display_vol / 100.0))
        if fill_w > 0:
            bar_color = RGB_DIM if is_muted else (RGB_CYAN if active_vol_mode == "rotational" else RGB_VIOLET)
            draw.rounded_rectangle([bar_x, bar_y, bar_x + fill_w, bar_y + bar_h], radius=5, fill=bar_color)

        # Volume numerical text
        font_vol = _font_manager.get_font(13, bold=True)
        vol_str = "MUTED" if is_muted else f"{vol_pct}%"
        draw.text((bar_x + bar_w + 14, bar_y - 3), vol_str, fill=RGB_TEXT if not is_muted else RGB_AMBER, font=font_vol)

        # Rotational Telemetry Sub-Card (Dynamic based on volume mode)
        rot_box_y = vol_box_y + vol_box_h + 14
        rot_box_h = h - (rot_box_y - y) - 16

        draw.rounded_rectangle([x + 20, rot_box_y, x + w - 20, rot_box_y + rot_box_h],
                               radius=6, fill=RGB_PANEL_ALT, outline=RGB_BORDER, width=1)

        font_sec = _font_manager.get_font(11, bold=True)
        draw.text((x + 34, rot_box_y + 12), "ROTATIONAL TELEMETRY", fill=RGB_MUTED, font=font_sec)

        font_body = _font_manager.get_font(11, bold=False)
        font_stat = _font_manager.get_font(12, bold=True)

        if active_vol_mode == "rotational":
            rot_status_str = "ACTIVE (OPEN PALM)" if rotation_active else "IDLE (HOLD OPEN PALM)"
            rot_col = RGB_GREEN if rotation_active else RGB_DIM

            draw.text((x + 34, rot_box_y + 36), "Status:", fill=RGB_MUTED, font=font_body)
            draw.text((x + 85, rot_box_y + 36), rot_status_str, fill=rot_col, font=font_stat)

            signed_rot = rot_diag.get("signed_rotation", 0.0) if rotation_active else 0.0
            direction = rot_diag.get("direction", "neutral").upper() if rotation_active else "NEUTRAL"

            draw.text((x + 34, rot_box_y + 60), "Angle / Tilt:", fill=RGB_MUTED, font=font_body)
            draw.text((x + 115, rot_box_y + 60), f"{signed_rot:+.1f} deg", fill=RGB_TEXT, font=font_stat)

            draw.text((x + 34, rot_box_y + 84), "Action:", fill=RGB_MUTED, font=font_body)
            dir_col = RGB_CYAN if direction == "INCREASE" else (RGB_RED if direction == "DECREASE" else RGB_MUTED)
            draw.text((x + 88, rot_box_y + 84), direction, fill=dir_col, font=font_stat)
        else:
            draw.text((x + 34, rot_box_y + 40), "Discrete mode selected.", fill=RGB_MUTED, font=font_body)
            draw.text((x + 34, rot_box_y + 62), "Use Open Palm for Vol+ and Fist for Vol-.", fill=RGB_DIM, font=font_body)
            draw.text((x + 34, rot_box_y + 84), "Press 'V' to enable Rotational Volume control.", fill=RGB_CYAN, font=font_body)

    def _render_gesture_status_bar(self, draw: ImageDraw.Draw,
                                   rect: Tuple[int, int, int, int],
                                   raw_gesture: Optional[str],
                                   pred_conf: float,
                                   stable_gesture: Optional[str],
                                   machine_state: str,
                                   active_vol_mode: str,
                                   rotation_active: bool,
                                   rot_diag: Dict[str, Any],
                                   hand_detected: bool = True):
        """Horizontal status section showing distinct futuristic chips for gesture metrics."""
        x, y, w, h = rect
        draw.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=RGB_PANEL, outline=RGB_BORDER, width=1)

        font_lbl = _font_manager.get_font(10, bold=True)
        font_val = _font_manager.get_font(12, bold=True)

        if not hand_detected:
            raw_text = "NO HAND"
            raw_col = RGB_DIM
        elif raw_gesture:
            raw_text = raw_gesture.upper()
            raw_col = RGB_TEXT
        else:
            raw_text = "UNKNOWN"
            raw_col = RGB_AMBER

        if not hand_detected:
            stable_text = "NONE"
            stable_col = RGB_DIM
        elif stable_gesture:
            stable_text = stable_gesture.upper()
            stable_col = RGB_CYAN
        else:
            stable_text = "UNKNOWN" if hand_detected else "NONE"
            stable_col = RGB_DIM

        conf_str = f"{int(pred_conf * 100)}%" if (hand_detected and raw_gesture) else "--"

        chips = [
            ("PREDICTION", raw_text, raw_col),
            ("STABLE GESTURE", stable_text, stable_col),
            ("CONFIDENCE", conf_str, RGB_GREEN if pred_conf >= 0.7 else (RGB_AMBER if pred_conf >= 0.4 else RGB_DIM)),
            ("MACHINE STATE", machine_state, RGB_TEXT if machine_state == "STABLE_POSE" else (RGB_AMBER if machine_state == "WAITING_FOR_RELEASE" else RGB_MUTED)),
            ("LAST ACTION", self.last_action_text, RGB_VIOLET)
        ]

        chip_gap = 10
        chip_w = (w - 20 - (chip_gap * (len(chips) - 1))) // len(chips)
        chip_y = y + 8
        chip_h = h - 16

        for i, (label, val, col) in enumerate(chips):
            cx = x + 10 + i * (chip_w + chip_gap)
            draw.rounded_rectangle([cx, chip_y, cx + chip_w, chip_y + chip_h], radius=4, fill=RGB_PANEL_ALT, outline=RGB_BORDER, width=1)
            draw.text((cx + 10, chip_y + 5), label, fill=RGB_MUTED, font=font_lbl)
            
            # Truncate gracefully if long
            disp_val = val if len(val) <= 16 else val[:14] + ".."
            draw.text((cx + 10, chip_y + 20), disp_val, fill=col, font=font_val)

    def _render_footer(self, draw: ImageDraw.Draw,
                       rect: Tuple[int, int, int, int],
                       active_vol_mode: str):
        """Futuristic 2-tier reference footer with clean gesture command grid and keyboard shortcuts."""
        x, y, w, h = rect
        draw.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=RGB_PANEL, outline=RGB_BORDER, width=1)

        font_title = _font_manager.get_font(10, bold=True)
        font_key = _font_manager.get_font(10, bold=True)
        font_desc = _font_manager.get_font(10, bold=False)

        # Header title
        draw.text((x + 16, y + 8), "GESTURE CONTROLS MATRIX", fill=RGB_CYAN, font=font_title)

        vol_action = "Rotate hand (Vol +/-)" if active_vol_mode == "rotational" else "Vol + (Palm) / Vol - (Fist)"

        # 8 Commands in 2 balanced rows of 4 columns
        matrix_row1 = [
            ("INDEX FINGER", "Play / Pause"),
            ("TWO FINGERS", "Mute / Unmute"),
            ("POINT LEFT", "Prev Track"),
            ("POINT RIGHT", "Next Track")
        ]
        matrix_row2 = [
            ("THUMBS UP", "Like Track"),
            ("THUMBS DOWN", "Unlike Track"),
            ("OPEN PALM", vol_action),
            ("FIST", "Volume Down (Discrete)")
        ]

        col_w = (w - 32) // 4
        y_r1 = y + 24
        y_r2 = y + 42

        for i, (g_name, g_desc) in enumerate(matrix_row1):
            gx = x + 16 + (i * col_w)
            draw.text((gx, y_r1), g_name + " • ", fill=RGB_TEXT, font=font_key)
            w_nm = int(draw.textlength(g_name + " • ", font=font_key))
            draw.text((gx + w_nm, y_r1), g_desc, fill=RGB_MUTED, font=font_desc)

        for i, (g_name, g_desc) in enumerate(matrix_row2):
            gx = x + 16 + (i * col_w)
            draw.text((gx, y_r2), g_name + " • ", fill=RGB_TEXT, font=font_key)
            w_nm = int(draw.textlength(g_name + " • ", font=font_key))
            draw.text((gx + w_nm, y_r2), g_desc, fill=RGB_MUTED, font=font_desc)

        # Bottom hairline separator for keyboard shortcuts
        draw.line([x + 16, y + 62, x + w - 16, y + 62], fill=RGB_BORDER, width=1)

        shortcuts = [
            ("V", "Toggle Volume Mode (Rotational / Discrete)"),
            ("R", "Recalibrate Gestures"),
            ("Q / ESC", "Quit GestureTune")
        ]

        sx = x + 16
        for k, d in shortcuts:
            k_len = int(draw.textlength(k, font=font_key))
            draw.rounded_rectangle([sx, y + 68, sx + k_len + 12, y + 84],
                                   radius=3, fill=RGB_PANEL_ALT, outline=RGB_BORDER, width=1)
            draw.text((sx + 6, y + 70), k, fill=RGB_CYAN, font=font_key)
            sx += k_len + 18
            draw.text((sx, y + 70), d, fill=RGB_MUTED, font=font_desc)
            sx += int(draw.textlength(d, font=font_desc)) + 24
