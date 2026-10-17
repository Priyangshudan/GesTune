import pytest
import numpy as np
import pandas as pd
from src.gesturetune.preprocess import PreprocessingPipeline, FEATURE_NAMES
from src.gesturetune.synthetic_data import generate_synthetic_dataset

def test_preprocessing_audit_and_cleaning():
    df_raw = generate_synthetic_dataset(n_samples_per_gesture=10, no_hand_count=5)
    pipeline = PreprocessingPipeline()

    audit = pipeline.audit_raw_data(df_raw)
    assert audit["total_columns"] == 67
    assert audit["schema_valid_67"] is True
    assert audit["hand_undetected_count"] == 5
    assert audit["exact_duplicate_rows"] == 3

    df_valid, df_no_hand, n_dups = pipeline.clean_data(df_raw)
    assert len(df_no_hand) == 5
    assert n_dups == 3
    # 7 gestures * 10 samples + 5 controlled outliers = 75 valid hand rows
    assert len(df_valid) == 75

def test_landmark_normalization():
    df_raw = generate_synthetic_dataset(n_samples_per_gesture=5, no_hand_count=2)
    pipeline = PreprocessingPipeline()
    df_valid, _, _ = pipeline.clean_data(df_raw)
    df_norm = pipeline.normalize_landmarks(df_valid)

    # In normalized landmarks, wrist (landmark 0) must be (0, 0, 0)
    assert np.allclose(df_norm["norm_landmark_0_x"], 0.0, atol=1e-5)
    assert np.allclose(df_norm["norm_landmark_0_y"], 0.0, atol=1e-5)
    assert np.allclose(df_norm["norm_landmark_0_z"], 0.0, atol=1e-5)

def test_eight_engineered_features():
    df_raw = generate_synthetic_dataset(n_samples_per_gesture=5, no_hand_count=0)
    pipeline = PreprocessingPipeline()
    df_valid, _, _ = pipeline.clean_data(df_raw)
    df_feat = pipeline.engineer_features(df_valid)

    for feat in FEATURE_NAMES:
        assert feat in df_feat.columns
        assert not df_feat[feat].isnull().any()
        assert np.isfinite(df_feat[feat]).all()

