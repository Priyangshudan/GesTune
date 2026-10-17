import pytest
import numpy as np
import pandas as pd
from src.gesturetune.isolation_forest import AnomalyAnalyzer
from src.gesturetune.preprocess import FEATURE_NAMES

def test_isolation_forest_execution():
    analyzer = AnomalyAnalyzer()
    np.random.seed(42)
    n = 60
    data = {feat: np.random.normal(0, 1, n) for feat in FEATURE_NAMES}
    data["gesture_label"] = ["open_palm"] * n
    data["record_id"] = list(range(n))
    
    # Introduce an extreme anomaly
    data["hand_spread"][0] = 50.0

    df = pd.DataFrame(data)
    flags, scores, iso = analyzer.run_isolation_forest(df)

    assert len(flags) == n
    assert len(scores) == n
    assert flags[0] == True  # The extreme outlier should be flagged
    assert scores[0] > np.mean(scores)

def test_jaccard_and_contingency_table():
    analyzer = AnomalyAnalyzer()
    df = pd.DataFrame({
        "gesture_label": ["fist"] * 10,
        "record_id": list(range(10))
    })
    if_flags = np.array([True, True, False, False, False, False, False, False, False, False])
    if_scores = np.random.uniform(0.4, 0.7, 10)
    
    comp = analyzer.compare_with_iqr(df, if_flags, if_scores)
    assert "table_x" in comp
    assert "jaccard" in comp
    assert 0.0 <= comp["jaccard"] <= 1.0

