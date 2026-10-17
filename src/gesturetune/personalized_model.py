"""
Personalized gesture model and profile manager.
Handles:
1. Training scikit-learn models (RandomForestClassifier, MLP, ExtraTrees) on user calibration samples.
2. Grouped cross-validation by recording attempt (session) to prevent optimistic frame correlation bias.
3. Calculating precision, recall, F1-score, and confusion matrices.
4. Profile serialization/deserialization into data/gesture_profiles/default.json with versioning and validation.
5. Soft-voting probabilities with confidence thresholds to reject ambiguous/uncertain poses.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import time
import logging
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from .feature_extraction import LandmarkFeatureExtractor

logger = logging.getLogger(__name__)

DEFAULT_VOCABULARY = [
    "index_finger", "two_fingers", "point_left", "point_right",
    "open_palm", "fist", "thumbs_up", "thumbs_down"
]

PROFILE_SCHEMA_VERSION = "2.1.0"
COMPATIBLE_SCHEMA_VERSIONS = {"2.0.0", "2.1.0"}

class PersonalizedGestureModel:
    def __init__(self, vocabulary: Optional[List[str]] = None,
                 confidence_threshold: float = 0.55):
        self.vocabulary = vocabulary or list(DEFAULT_VOCABULARY)
        self.confidence_threshold = confidence_threshold
        self.classifier: Optional[RandomForestClassifier] = None
        self.feature_extractor = LandmarkFeatureExtractor()
        self.classes_: List[str] = sorted(self.vocabulary)
        self.metadata: Dict[str, Any] = {}
        self.rotational_calibration: Optional[Dict[str, Any]] = None

    def train(self, X: np.ndarray, y: List[str], groups: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Train a Random Forest classifier on calibration feature vectors.
        Evaluates performance using grouped CV across distinct registration attempts.
        """
        if len(X) < len(self.vocabulary) * 3:
            raise ValueError(f"Insufficient training samples ({len(X)} samples for {len(self.vocabulary)} gestures)")

        y_arr = np.array(y)
        unique_labels = sorted(list(set(y_arr)))
        self.classes_ = unique_labels

        # Train Random Forest classifier with calibrated probability output
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=3,
            class_weight="balanced",
            random_state=42
        )

        evaluation_results: Dict[str, Any] = {}

        # Evaluate via Grouped CV if groups (attempt indices) are available
        if groups is not None and len(set(groups)) >= 3:
            logo = LeaveOneGroupOut()
            y_true_all = []
            y_pred_all = []
            for train_idx, val_idx in logo.split(X, y_arr, groups=groups):
                # Ensure training fold covers classes
                if len(set(y_arr[train_idx])) == len(unique_labels):
                    fold_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
                    fold_model.fit(X[train_idx], y_arr[train_idx])
                    preds = fold_model.predict(X[val_idx])
                    y_true_all.extend(y_arr[val_idx])
                    y_pred_all.extend(preds)

            if len(y_true_all) > 0:
                report = classification_report(y_true_all, y_pred_all, labels=unique_labels, output_dict=True, zero_division=0)
                cm = confusion_matrix(y_true_all, y_pred_all, labels=unique_labels).tolist()
                evaluation_results = {
                    "method": "LeaveOneGroupOut (by attempt)",
                    "report": report,
                    "confusion_matrix": cm,
                    "accuracy": float(np.mean(np.array(y_true_all) == np.array(y_pred_all)))
                }
        else:
            # Fallback to Stratified K-Fold
            n_splits = min(5, min(np.bincount([unique_labels.index(lbl) for lbl in y_arr])))
            if n_splits >= 2:
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                y_true_all = []
                y_pred_all = []
                for train_idx, val_idx in skf.split(X, y_arr):
                    fold_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
                    fold_model.fit(X[train_idx], y_arr[train_idx])
                    preds = fold_model.predict(X[val_idx])
                    y_true_all.extend(y_arr[val_idx])
                    y_pred_all.extend(preds)
                report = classification_report(y_true_all, y_pred_all, labels=unique_labels, output_dict=True, zero_division=0)
                cm = confusion_matrix(y_true_all, y_pred_all, labels=unique_labels).tolist()
                evaluation_results = {
                    "method": f"{n_splits}-Fold Stratified CV",
                    "report": report,
                    "confusion_matrix": cm,
                    "accuracy": float(np.mean(np.array(y_true_all) == np.array(y_pred_all)))
                }

        # Fit final model on full dataset
        rf.fit(X, y_arr)
        self.classifier = rf

        self.metadata = {
            "trained_at": time.time(),
            "sample_count": len(X),
            "feature_dim": X.shape[1],
            "evaluation": evaluation_results,
            "classes": unique_labels
        }
        return evaluation_results

    def predict(self, landmarks: Any, is_mirrored_preview: bool = True) -> Tuple[Optional[str], float]:
        """
        Predict gesture from 21 landmarks with confidence gating.
        Returns:
            (predicted_gesture_or_None, confidence_score)
        """
        if self.classifier is None:
            return None, 0.0

        feat = self.feature_extractor.extract_features(landmarks, is_mirrored_preview=is_mirrored_preview)
        if np.all(feat == 0.0):
            return None, 0.0

        probs = self.classifier.predict_proba(feat.reshape(1, -1))[0]
        best_idx = int(np.argmax(probs))
        best_conf = float(probs[best_idx])
        best_label = self.classes_[best_idx]

        if best_conf >= self.confidence_threshold:
            return best_label, best_conf
        return None, best_conf

    def save_profile(self, profile_json_path: Path, model_joblib_path: Path,
                     calibration_csv_path: Optional[Path] = None,
                     rotational_calibration: Optional[Dict[str, Any]] = None):
        """Save persistent profile metadata, fitted model weights, and rotational volume calibration."""
        profile_json_path.parent.mkdir(parents=True, exist_ok=True)
        model_joblib_path.parent.mkdir(parents=True, exist_ok=True)

        if rotational_calibration is not None:
            self.rotational_calibration = rotational_calibration

        if self.classifier is not None:
            joblib.dump(self.classifier, model_joblib_path)

        profile_data = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "created_at": time.time(),
            "vocabulary": self.vocabulary,
            "classes": self.classes_,
            "confidence_threshold": self.confidence_threshold,
            "model_file": str(model_joblib_path.name),
            "calibration_samples_file": str(calibration_csv_path.name) if calibration_csv_path else None,
            "rotational_calibration": self.rotational_calibration,
            "metadata": self.metadata
        }

        with open(profile_json_path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2)

    @classmethod
    def load_profile(cls, profile_json_path: Path) -> Optional["PersonalizedGestureModel"]:
        """Load and validate profile from disk, handling backward compatibility with v2.0.0 profiles."""
        if not profile_json_path.exists():
            return None

        try:
            with open(profile_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            schema_ver = data.get("schema_version")
            if schema_ver not in COMPATIBLE_SCHEMA_VERSIONS:
                logger.warning(f"Incompatible profile schema version {schema_ver} (expected one of {COMPATIBLE_SCHEMA_VERSIONS})")
                return None

            vocab = data.get("vocabulary", [])
            conf_thresh = float(data.get("confidence_threshold", 0.55))
            model_inst = cls(vocabulary=vocab, confidence_threshold=conf_thresh)
            model_inst.classes_ = data.get("classes", vocab)
            model_inst.metadata = data.get("metadata", {})
            model_inst.rotational_calibration = data.get("rotational_calibration", None)

            model_filename = data.get("model_file")
            if model_filename:
                model_path = profile_json_path.parent / model_filename
                if model_path.exists():
                    model_inst.classifier = joblib.load(model_path)
                else:
                    logger.warning(f"Model file '{model_path}' referenced in profile not found.")
                    return None

            return model_inst
        except Exception as e:
            logger.error(f"Error loading gesture profile {profile_json_path}: {e}")
            return None

