import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.gesturetune import get_raw_columns
from src.gesturetune.capture import record_observation
from src.gesturetune.synthetic_data import generate_synthetic_dataset

def test_raw_schema_column_count():
    cols = get_raw_columns()
    assert len(cols) == 67
    assert cols[0] == "timestamp"
    assert cols[1] == "gesture_label"
    assert cols[2] == "hand_detected"
    assert cols[3] == "confidence"
    assert cols[4] == "landmark_0_x"
    assert cols[-1] == "landmark_20_z"

def test_record_detected_hand():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_raw.csv"
        landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(21)]
        
        record_observation(csv_path, "open_palm", True, landmarks, confidence=0.95)
        
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert df.shape[1] == 67
        assert df.iloc[0]["hand_detected"] == 1
        assert df.iloc[0]["gesture_label"] == "open_palm"
        assert df.iloc[0]["confidence"] == 0.95
        assert not np.isnan(df.iloc[0]["landmark_0_x"])

def test_record_no_hand_structural_nan():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_raw.csv"
        
        record_observation(csv_path, "fist", False, None, confidence=None)
        
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert df.shape[1] == 67
        assert df.iloc[0]["hand_detected"] == 0
        assert df.iloc[0]["gesture_label"] == "fist"
        assert np.isnan(df.iloc[0]["confidence"])
        for i in range(21):
            assert np.isnan(df.iloc[0][f"landmark_{i}_x"])
            assert np.isnan(df.iloc[0][f"landmark_{i}_y"])
            assert np.isnan(df.iloc[0][f"landmark_{i}_z"])

def test_synthetic_dataset_conforms_to_schema():
    df = generate_synthetic_dataset(n_samples_per_gesture=5, no_hand_count=3)
    assert df.shape[1] == 67
    assert set(df.columns) == set(get_raw_columns())
    assert (df["hand_detected"] == 0).sum() == 3

