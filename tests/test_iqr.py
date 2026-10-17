import pytest
import numpy as np
import pandas as pd
from src.gesturetune.outlier_iqr import IQROutlierDetector

def test_iqr_toy_fences():
    detector = IQROutlierDetector()
    # Data: 1, 2, 3, 4, 5, 6, 7, 8, 9, 100 (100 is an outlier)
    data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0])
    fences = detector.compute_fences(data)
    
    # Q1 ~ 3.25, Q3 ~ 7.75, IQR ~ 4.5
    assert fences["q3"] > fences["q1"]
    assert fences["upper"] < 100.0
    assert 100.0 > fences["upper"]

def test_iqr_winsorisation():
    detector = IQROutlierDetector()
    df = pd.DataFrame({
        "hand_width": [10.0, 11.0, 10.5, 12.0, 11.5, 9.5, 10.0, 100.0],
        "hand_height": [20.0, 21.0, 20.5, 22.0, 21.5, 19.5, 20.0, 20.2],
        "gesture_label": ["open_palm"] * 8
    })
    _, df_flags = detector.detect_outliers(df)
    assert df_flags["hand_width_outlier"].iloc[-1] == True

    df_orig, df_win, df_rem, _ = detector.apply_treatments(df, df_flags)
    assert df_win["hand_width"].iloc[-1] < 100.0
    assert len(df_rem) == 7

