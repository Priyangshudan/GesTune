"""
Hybrid Gesture Recognizer module.
Supports:
1. Personalized trained model prediction with probability confidence scoring.
2. Graceful fallback to geometric rule-based recognizer when no profile is available
   or when model confidence is low.
3. Class-specific confidence thresholds and uncertainty / neutral state outputs.
"""

from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

from .gesture_recognition import GestureRecognizer
from .personalized_model import PersonalizedGestureModel

class HybridGestureRecognizer:
    def __init__(self, profile_path: Optional[Path] = None,
                 fallback_rule_recognizer: Optional[GestureRecognizer] = None,
                 confidence_threshold: float = 0.55):
        self.profile_path = profile_path
        self.confidence_threshold = confidence_threshold
        self.rule_recognizer = fallback_rule_recognizer or GestureRecognizer()
        self.personalized_model: Optional[PersonalizedGestureModel] = None

        if profile_path and profile_path.exists():
            self.personalized_model = PersonalizedGestureModel.load_profile(profile_path)

    def is_personalized(self) -> bool:
        return self.personalized_model is not None and self.personalized_model.classifier is not None

    def recognize(self, landmarks: List[Dict[str, float]], is_mirrored_preview: bool = True) -> Tuple[Optional[str], float, str]:
        """
        Recognize gesture from 21 landmarks.
        Returns:
            (gesture_label_or_None, confidence, recognizer_source)
        """
        if not landmarks or len(landmarks) < 21:
            return None, 0.0, "neutral"

        # 1. Try personalized model if available
        if self.personalized_model and self.personalized_model.classifier is not None:
            pred_label, conf = self.personalized_model.predict(landmarks, is_mirrored_preview=is_mirrored_preview)
            if pred_label is not None and conf >= self.confidence_threshold:
                return pred_label, conf, "personalized"

        # 2. Fallback to geometric rule recognizer
        rule_pred = self.rule_recognizer.recognize(landmarks, is_mirrored_preview=is_mirrored_preview)
        if rule_pred is not None:
            return rule_pred, 0.85, "rule_fallback"

        return None, 0.0, "neutral"

