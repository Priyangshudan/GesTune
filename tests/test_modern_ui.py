import pytest
import numpy as np

from src.gesturetune.modern_ui import ModernPlayerUI, _font_manager

def test_modern_player_ui_initialization_and_fonts():
    ui = ModernPlayerUI(target_width=1120, target_height=720)
    assert ui.width == 1120
    assert ui.height == 720
    assert _font_manager.get_font(14, bold=False) is not None
    assert _font_manager.get_font(18, bold=True) is not None

def test_modern_player_ui_rendering_dimensions_and_channels():
    ui = ModernPlayerUI(target_width=1120, target_height=720)
    dummy_cam = np.zeros((480, 640, 3), dtype=np.uint8)
    status = {
        "track": "Cosmic_Journey.mp3",
        "is_playing": True,
        "is_paused": False,
        "volume": 75,
        "is_muted": False,
        "is_liked": True
    }
    
    frame = ui.render(
        camera_frame=dummy_cam,
        hand_detected=True,
        raw_gesture="open_palm",
        pred_conf=0.88,
        stable_gesture="open_palm",
        action_msg="Volume: 75%",
        machine_state="STABLE_POSE",
        active_vol_mode="rotational",
        rotation_active=True,
        rot_diag={"signed_rotation": 16.4, "direction": "increase", "active": True},
        music_status=status,
        mode_str="Personalized (ML)",
        fps=30.0
    )
    
    assert frame is not None
    assert frame.shape == (720, 1120, 3)
    assert frame.dtype == np.uint8

def test_modern_player_ui_discrete_volume_mode_rendering():
    ui = ModernPlayerUI(target_width=1000, target_height=650)
    dummy_cam = np.zeros((480, 640, 3), dtype=np.uint8)
    status = {
        "track": "Ambient_Vibes",
        "is_playing": False,
        "is_paused": True,
        "volume": 40,
        "is_muted": True,
        "is_liked": False
    }

    frame = ui.render(
        camera_frame=dummy_cam,
        hand_detected=False,
        raw_gesture=None,
        pred_conf=0.0,
        stable_gesture=None,
        action_msg=None,
        machine_state="NEUTRAL",
        active_vol_mode="discrete",
        rotation_active=False,
        rot_diag={},
        music_status=status,
        mode_str="Rule-based Fallback",
        fps=28.5
    )

    assert frame is not None
    assert frame.shape == (650, 1000, 3)

def test_modern_player_ui_action_persistence():
    ui = ModernPlayerUI(target_width=1120, target_height=720)
    ui.update_action("index_finger -> Playing: Song A")
    assert ui.last_action_text == "index_finger -> Playing: Song A"
    assert ui.last_action_timestamp > 0.0

