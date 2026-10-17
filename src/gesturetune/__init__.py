"""
GestureTune package initialization and configuration loader.
"""

from pathlib import Path
from typing import Any, Dict
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration YAML relative to the project root."""
    full_path = PROJECT_ROOT / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {full_path}")
    with open(full_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def get_raw_columns() -> list:
    """Return the exact 67 columns for raw_landmarks.csv as defined in the case study."""
    cols = ["timestamp", "gesture_label", "hand_detected", "confidence"]
    for i in range(21):
        cols.extend([f"landmark_{i}_x", f"landmark_{i}_y", f"landmark_{i}_z"])
    return cols

