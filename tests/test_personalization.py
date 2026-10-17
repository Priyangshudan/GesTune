import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.gesturetune.feature_extraction import LandmarkFeatureExtractor, landmarks_to_array
from src.gesturetune.synthetic_data import generate_gesture_skeleton
from src.gesturetune.personalized_model import PersonalizedGestureModel
from src.gesturetune.hybrid_recognizer import HybridGestureRecognizer
from src.gesturetune.stabilized_controller import StabilizedGestureController
from src.gesturetune.music_controller import MusicController

def skeleton_to_dict_list(coords: np.ndarray):
    return [{"x": float(pt[0]), "y": float(pt[1]), "z": float(pt[2])} for pt in coords]

def test_feature_extraction_invariance_translation_and_scale():
    extractor = LandmarkFeatureExtractor()
    base_skel = generate_gesture_skeleton("index_finger")

    # 1. Base feature
    feat_base = extractor.extract_features(skeleton_to_dict_list(base_skel))
    assert len(feat_base) == LandmarkFeatureExtractor.get_feature_dim()
    assert not np.isnan(feat_base).any()

    # 2. Translate hand across image (e.g. +0.25 in x, -0.3 in y)
    trans_skel = base_skel.copy()
    trans_skel[:, 0] += 0.25
    trans_skel[:, 1] -= 0.30
    feat_trans = extractor.extract_features(skeleton_to_dict_list(trans_skel))

    # Because features are wrist-relative, translated features should match base within precision
    np.testing.assert_allclose(feat_base[:63], feat_trans[:63], atol=1e-4)

    # 3. Scale hand (e.g. 1.5x larger palm)
    wrist = base_skel[0]
    scaled_skel = wrist + (base_skel - wrist) * 1.5
    feat_scaled = extractor.extract_features(skeleton_to_dict_list(scaled_skel))
    np.testing.assert_allclose(feat_base[:63], feat_scaled[:63], atol=1e-4)

def test_feature_extraction_degenerate_and_nans():
    extractor = LandmarkFeatureExtractor()
    # All zeros
    zeros = np.zeros((21, 3))
    feat_zeros = extractor.extract_features(zeros)
    assert not np.isnan(feat_zeros).any()
    assert len(feat_zeros) == LandmarkFeatureExtractor.get_feature_dim()

    # NaNs input
    nans = np.full((21, 3), np.nan)
    feat_nans = extractor.extract_features(nans)
    assert not np.isnan(feat_nans).any()
    assert np.all(feat_nans == 0.0)

def test_personalized_model_training_and_serialization():
    extractor = LandmarkFeatureExtractor()
    vocab = ["index_finger", "two_fingers", "point_left", "point_right", "open_palm", "fist", "thumbs_up"]

    X = []
    y = []
    groups = []

    # Generate 5 attempts x 5 samples per gesture
    for g_idx, g in enumerate(vocab):
        base_skel = generate_gesture_skeleton(g)
        for attempt in range(1, 4):
            for s in range(5):
                noise = np.random.normal(0, 0.005, size=base_skel.shape)
                feat = extractor.extract_features(skeleton_to_dict_list(base_skel + noise))
                X.append(feat)
                y.append(g)
                groups.append(g_idx * 10 + attempt)

    X = np.array(X)
    model = PersonalizedGestureModel(vocabulary=vocab, confidence_threshold=0.5)
    eval_res = model.train(X, y, groups=groups)

    assert "accuracy" in eval_res
    assert eval_res["accuracy"] > 0.85

    # Test serialization
    with tempfile.TemporaryDirectory() as tmpdir:
        prof_json = Path(tmpdir) / "default.json"
        model_joblib = Path(tmpdir) / "default_model.joblib"
        model.save_profile(prof_json, model_joblib)

        assert prof_json.exists()
        assert model_joblib.exists()

        # Load back
        loaded = PersonalizedGestureModel.load_profile(prof_json)
        assert loaded is not None
        assert loaded.classifier is not None
        assert loaded.classes_ == sorted(vocab)

        # Test prediction
        test_skel = generate_gesture_skeleton("thumbs_up")
        pred, conf = loaded.predict(skeleton_to_dict_list(test_skel))
        assert pred == "thumbs_up"
        assert conf >= 0.5

def test_hybrid_recognizer_fallback():
    # Hybrid with non-existent profile should fall back to rule-based recognizer
    non_existent = Path("data/gesture_profiles/does_not_exist_test.json")
    hybrid = HybridGestureRecognizer(profile_path=non_existent)
    assert not hybrid.is_personalized()

    test_skel = generate_gesture_skeleton("point_right")
    pred, conf, source = hybrid.recognize(skeleton_to_dict_list(test_skel), is_mirrored_preview=True)
    assert pred == "point_right"
    assert source == "rule_fallback"

def test_stabilized_controller_release_rearm_prevents_repeated_play_pause():
    music = MusicController()
    controller = StabilizedGestureController(
        music_controller=music,
        window_size=3,
        consistency_ratio=0.6,
        cooldown_seconds=0.1
    )

    # Frame 1: index_finger
    stable, action, state = controller.process_prediction("index_finger")
    assert action is None  # Window not yet filled

    # Frame 2: index_finger
    stable, action, state = controller.process_prediction("index_finger")
    assert stable == "index_finger"
    assert action is not None  # Fired once!
    assert state == "WAITING_FOR_RELEASE"

    # Frames 3 through 10: Continuously holding index_finger!
    for _ in range(8):
        stable, action, state = controller.process_prediction("index_finger")
        # MUST NOT FIRE AGAIN!
        assert action is None
        assert state == "WAITING_FOR_RELEASE"

    # Now release hand / neutral pose
    for _ in range(3):
        stable, action, state = controller.process_prediction(None)
    assert state == "NEUTRAL"
    # Wait for cooldown to expire (0.12s > 0.1s)
    import time
    time.sleep(0.12)

    # Pose index_finger again after re-arming:
    # Frame 1 fills window partially
    s1, a1, st1 = controller.process_prediction("index_finger")
    assert a1 is None
    # Frame 2 reaches 2/3 consistency threshold and fires!
    s2, a2, st2 = controller.process_prediction("index_finger")
    assert a2 is not None
    assert st2 == "WAITING_FOR_RELEASE"
