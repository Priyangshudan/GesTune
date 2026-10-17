"""
Data Preprocessing and Feature Engineering pipeline (preprocess.py).
Takes data/raw_landmarks.csv and performs:
1. Data audit and validation (schema, duplicates, invalid labels, range violations).
2. Structural missingness handling (keeping no-hand records for detection rate, isolating for geometry).
3. Landmark normalization: wrist origin translation and palm-length scaling (s = ||p9 - p0||).
4. Feature engineering:
   - confidence
   - hand_width
   - hand_height
   - hand_spread
   - thumb_index_distance
   - palm_size
   - movement_speed
   - acceleration
5. Skewness check & log1p transformation.
6. Z-score standardisation.
7. Saves outputs to data/gesture_features.csv and intermediate processed tables.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json
import logging
import numpy as np
import pandas as pd
from scipy.stats import skew

from . import load_config, get_raw_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "confidence",
    "hand_width",
    "hand_height",
    "hand_spread",
    "thumb_index_distance",
    "palm_size",
    "movement_speed",
    "acceleration"
]

class PreprocessingPipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.data_cfg = self.config.get("data", {})
        self.analysis_cfg = self.config.get("analysis", {})
        self.vocab = set(self.config.get("gestures", {}).get("vocabulary", []))
        self.raw_cols = get_raw_columns()

    def audit_raw_data(self, df_raw: pd.DataFrame) -> Dict[str, Any]:
        """Conduct a comprehensive data audit on the raw dataset."""
        n_rows, n_cols = df_raw.shape
        missing_per_col = df_raw.isnull().sum().to_dict()
        exact_duplicates = int(df_raw.duplicated().sum())

        unique_labels = [str(lbl).strip() for lbl in df_raw["gesture_label"].dropna().unique()]
        invalid_labels = [lbl for lbl in unique_labels if lbl not in self.vocab]

        # Coordinate checks
        x_cols = [f"landmark_{i}_x" for i in range(21) if f"landmark_{i}_x" in df_raw.columns]
        y_cols = [f"landmark_{i}_y" for i in range(21) if f"landmark_{i}_y" in df_raw.columns]
        
        range_violations = 0
        if x_cols and y_cols:
            x_valid = df_raw[x_cols].dropna()
            y_valid = df_raw[y_cols].dropna()
            range_violations += int(((x_valid < 0.0) | (x_valid > 1.0)).values.sum())
            range_violations += int(((y_valid < 0.0) | (y_valid > 1.0)).values.sum())

        hand_detected_counts = df_raw["hand_detected"].value_counts().to_dict()
        detected_count = int(hand_detected_counts.get(1, 0))
        undetected_count = int(hand_detected_counts.get(0, 0))
        no_hand_rate = float(undetected_count / n_rows) if n_rows > 0 else 0.0

        per_gesture_counts = df_raw["gesture_label"].value_counts().to_dict()
        no_hand_per_gesture = df_raw[df_raw["hand_detected"] == 0]["gesture_label"].value_counts().to_dict()

        audit_results = {
            "total_rows": n_rows,
            "total_columns": n_cols,
            "schema_valid_67": bool(n_cols == 67 and list(df_raw.columns) == self.raw_cols),
            "missing_values_per_col": missing_per_col,
            "exact_duplicate_rows": exact_duplicates,
            "unique_gesture_labels": unique_labels,
            "invalid_labels": invalid_labels,
            "coordinate_range_violations": range_violations,
            "hand_detected_count": detected_count,
            "hand_undetected_count": undetected_count,
            "no_hand_rate": no_hand_rate,
            "per_gesture_counts": per_gesture_counts,
            "no_hand_per_gesture": no_hand_per_gesture
        }
        return audit_results

    def clean_data(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
        """
        Clean data:
        - Remove exact duplicates.
        - Trim and lowercase gesture labels.
        - Separate structural no-hand rows from valid hand observations.
        """
        n_before = len(df_raw)
        df_cleaned = df_raw.drop_duplicates().copy()
        n_duplicates_removed = n_before - len(df_cleaned)

        df_cleaned["gesture_label"] = df_cleaned["gesture_label"].astype(str).str.strip().str.lower()
        # Keep only allowed vocabulary
        df_cleaned = df_cleaned[df_cleaned["gesture_label"].isin(self.vocab)].copy()

        df_no_hand = df_cleaned[df_cleaned["hand_detected"] == 0].copy()
        df_valid_hand = df_cleaned[df_cleaned["hand_detected"] == 1].copy()

        # Remove rows where landmarks are NaN even though hand_detected == 1
        coord_cols = [f"landmark_{i}_{c}" for i in range(21) for c in ("x", "y", "z")]
        df_valid_hand = df_valid_hand.dropna(subset=coord_cols).copy()

        return df_valid_hand, df_no_hand, n_duplicates_removed

    def normalize_landmarks(self, df_valid: pd.DataFrame) -> pd.DataFrame:
        """
        Perform landmark normalisation:
        Wrist (p0) as origin.
        Palm scale s = ||p9 - p0||.
        p'_i = (p_i - p0) / s
        """
        df_norm = df_valid.copy()
        valid_indices = []

        for idx, row in df_valid.iterrows():
            p0 = np.array([row["landmark_0_x"], row["landmark_0_y"], row["landmark_0_z"]])
            p9 = np.array([row["landmark_9_x"], row["landmark_9_y"], row["landmark_9_z"]])
            palm_length = float(np.linalg.norm(p9 - p0))

            if palm_length < 1e-4:
                logger.warning(f"Row {idx} has near-zero palm length ({palm_length:.6f}). Excluding from normalized set.")
                continue

            valid_indices.append(idx)
            for i in range(21):
                pi = np.array([row[f"landmark_{i}_x"], row[f"landmark_{i}_y"], row[f"landmark_{i}_z"]])
                pi_norm = (pi - p0) / palm_length
                df_norm.loc[idx, f"norm_landmark_{i}_x"] = pi_norm[0]
                df_norm.loc[idx, f"norm_landmark_{i}_y"] = pi_norm[1]
                df_norm.loc[idx, f"norm_landmark_{i}_z"] = pi_norm[2]

        return df_norm.loc[valid_indices].copy()

    def engineer_features(self, df_valid: pd.DataFrame) -> pd.DataFrame:
        """
        Extract the eight required features:
        1. confidence
        2. hand_width
        3. hand_height
        4. hand_spread
        5. thumb_index_distance
        6. palm_size
        7. movement_speed
        8. acceleration
        """
        rows = []
        df_sorted = df_valid.sort_values(by="timestamp").reset_index(drop=False)

        prev_time = None
        prev_ref_pt = None
        prev_speed = None

        for idx, row in df_sorted.iterrows():
            orig_idx = row["index"]
            conf = float(row["confidence"]) if not pd.isna(row["confidence"]) else 1.0

            # Gather 21 landmark points
            coords = []
            for i in range(21):
                coords.append([row[f"landmark_{i}_x"], row[f"landmark_{i}_y"], row[f"landmark_{i}_z"]])
            coords = np.array(coords)

            # Static features
            x_vals = coords[:, 0]
            y_vals = coords[:, 1]
            hand_width = float(np.max(x_vals) - np.min(x_vals))
            hand_height = float(np.max(y_vals) - np.min(y_vals))

            # Palm size = ||p9 - p0||
            p0 = coords[0]
            p9 = coords[9]
            palm_size = float(np.linalg.norm(p9 - p0))

            # Palm centre approx (mean of wrist 0, index MCP 5, middle MCP 9, pinky MCP 17)
            palm_centre = np.mean([coords[0], coords[5], coords[9], coords[17]], axis=0)

            # Hand spread: mean Euclidean distance of 5 fingertips (4, 8, 12, 16, 20) from palm centre
            fingertip_indices = [4, 8, 12, 16, 20]
            fingertip_dists = [np.linalg.norm(coords[f_idx] - palm_centre) for f_idx in fingertip_indices]
            hand_spread = float(np.mean(fingertip_dists))

            # Thumb-index distance: ||p8 - p4||
            thumb_index_dist = float(np.linalg.norm(coords[8] - coords[4]))

            # Kinematic features: movement_speed & acceleration
            t = float(row["timestamp"])
            ref_pt = coords[0]  # wrist as reference point

            # Check consecutive capture timing (if dt > 2.0s, consider it a new session/pause and reset)
            if prev_time is not None and (0.001 < (t - prev_time) <= 2.0):
                dt = t - prev_time
                disp = np.linalg.norm(ref_pt - prev_ref_pt)
                speed = float(disp / dt)

                if prev_speed is not None:
                    accel = float((speed - prev_speed) / dt)
                else:
                    accel = 0.0
            else:
                speed = 0.0
                accel = 0.0

            prev_time = t
            prev_ref_pt = ref_pt
            prev_speed = speed

            feature_dict = {
                "record_id": orig_idx,
                "timestamp": t,
                "gesture_label": row["gesture_label"],
                "confidence": conf,
                "hand_width": hand_width,
                "hand_height": hand_height,
                "hand_spread": hand_spread,
                "thumb_index_distance": thumb_index_dist,
                "palm_size": palm_size,
                "movement_speed": speed,
                "acceleration": accel
            }
            rows.append(feature_dict)

        df_feat = pd.DataFrame(rows)
        return df_feat

    def check_and_transform_skew(self, df_feat: pd.DataFrame, threshold: float = 1.0) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Log1p transform strongly right-skewed nonnegative features exceeding skewness threshold."""
        df_out = df_feat.copy()
        skew_report = {}
        for col in FEATURE_NAMES:
            if col in df_out.columns:
                vals = df_out[col].values
                val_skew = float(skew(vals))
                skew_report[col] = val_skew
                # If nonnegative and strongly right-skewed
                if val_skew > threshold and np.all(vals >= 0.0):
                    df_out[col] = np.log1p(df_out[col])
                    logger.info(f"Applied log1p transformation to '{col}' (skewness: {val_skew:.3f} > {threshold})")
        return df_out, skew_report

    def standardize_features(self, df_feat: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
        """Z-score standardisation: z = (x - mu) / sigma."""
        df_std = df_feat.copy()
        scaler_params = {}
        eps = float(self.analysis_cfg.get("near_zero_variance_threshold", 1e-6))

        for col in FEATURE_NAMES:
            if col in df_std.columns:
                mu = float(df_std[col].mean())
                sigma = float(df_std[col].std())
                if sigma < eps:
                    logger.warning(f"Feature '{col}' has near-zero variance ({sigma}). Values set to 0.0.")
                    df_std[col] = 0.0
                else:
                    df_std[col] = (df_std[col] - mu) / sigma
                scaler_params[col] = (mu, sigma)
        return df_std, scaler_params

    def run(self, raw_csv_path: Optional[str] = None) -> Dict[str, Any]:
        """Execute complete preprocessing pipeline."""
        raw_path = Path(raw_csv_path) if raw_csv_path else Path(self.data_cfg.get("raw_dataset_path", "data/raw_landmarks.csv"))
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw landmarks file does not exist: {raw_path}")

        df_raw = pd.read_csv(raw_path)
        audit = self.audit_raw_data(df_raw)

        df_valid, df_no_hand, n_dups = self.clean_data(df_raw)
        df_norm = self.normalize_landmarks(df_valid)
        df_features = self.engineer_features(df_valid)

        skew_thresh = float(self.analysis_cfg.get("skewness_log_threshold", 1.0))
        df_feat_skew, skew_report = self.check_and_transform_skew(df_features, threshold=skew_thresh)
        df_standardized, scaler_params = self.standardize_features(df_feat_skew)

        # Save outputs
        out_feat_path = Path(self.data_cfg.get("features_dataset_path", "data/gesture_features.csv"))
        out_feat_path.parent.mkdir(parents=True, exist_ok=True)
        df_features.to_csv(out_feat_path, index=False)

        proc_dir = Path(self.data_cfg.get("processed_dir", "data/processed"))
        proc_dir.mkdir(parents=True, exist_ok=True)
        df_norm.to_csv(proc_dir / "normalized_landmarks.csv", index=False)
        df_standardized.to_csv(proc_dir / "standardized_features.csv", index=False)

        # Write audit report to outputs/reports/audit_report.json
        rep_dir = Path(self.config.get("outputs", {}).get("reports_dir", "outputs/reports"))
        rep_dir.mkdir(parents=True, exist_ok=True)
        with open(rep_dir / "audit_report.json", "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2)

        logger.info(f"Preprocessing completed successfully. Processed {len(df_valid)} valid observations.")
        return {
            "audit": audit,
            "n_valid": len(df_valid),
            "n_no_hand": len(df_no_hand),
            "n_duplicates_removed": n_dups,
            "skew_report": skew_report
        }

if __name__ == "__main__":
    pipeline = PreprocessingPipeline()
    pipeline.run()

