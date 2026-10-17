import pytest
import numpy as np
from src.gesturetune.pca_svd_analysis import PCASVDAnalysis

def test_pca_svd_numerical_equivalence():
    analyzer = PCASVDAnalysis()
    np.random.seed(42)
    # Generate random matrix 50 samples, 8 features
    X = np.random.randn(50, 8)
    feature_names = [f"feat_{i}" for i in range(8)]
    labels = np.array(["g1"] * 25 + ["g2"] * 25)

    res = analyzer.analyze_matrix(X, feature_names, "Test Scenario", labels=labels)

    # 1. Verification of lambda_j = s_j^2 / (n - 1)
    assert res["equivalence_passed"] is True
    assert res["max_eig_diff"] < 1e-4

    # 2. Check cumulative variance monotonic to 1.0
    cum_var = res["cumulative_variance"]
    assert np.all(np.diff(cum_var) >= -1e-6)
    assert np.isclose(cum_var[-1], 1.0)

    # 3. Check low-rank reconstruction matches theoretical residual
    for k_idx in range(len(res["ranks"])):
        err = res["reconstruction_errors"][k_idx]
        theo = res["theoretical_residuals"][k_idx]
        assert np.isclose(err, theo, rtol=1e-4, atol=1e-4)

