"""
Interquartile Range (IQR) Outlier Detection module (outlier_iqr.py).
Computes:
  IQR = Q3 - Q1
  Lower fence = Q1 - 1.5 * IQR
  Upper fence = Q3 + 1.5 * IQR
Flags observations outside fences, both globally and per-gesture.
Implements two treatments:
  A. Winsorisation (capping at fences)
  B. Removal (separate sensitivity analysis dataset)
Produces:
  - outputs/tables/iqr_summary.csv
  - outputs/tables/iqr_record_flags.csv
  - outputs/tables/iqr_treatment_comparison.csv
  - outputs/figures/iqr_boxplots_before_after.png
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from . import load_config
from .preprocess import FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class IQROutlierDetector:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.feat_path = Path(self.config.get("data", {}).get("features_dataset_path", "data/gesture_features.csv"))
        self.tables_dir = Path(self.config.get("outputs", {}).get("tables_dir", "outputs/tables"))
        self.figures_dir = Path(self.config.get("outputs", {}).get("figures_dir", "outputs/figures"))
        self.multiplier = float(self.config.get("analysis", {}).get("iqr_multiplier", 1.5))
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def compute_fences(self, series: pd.Series) -> Dict[str, float]:
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - self.multiplier * iqr
        upper = q3 + self.multiplier * iqr
        return {"q1": q1, "q3": q3, "iqr": iqr, "lower": lower, "upper": upper}

    def detect_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute global IQR fences and per-record outlier flags.
        Returns (df_summary, df_flags).
        """
        n_total = len(df)
        summary_records = []
        df_flags = df[["record_id", "gesture_label"]].copy() if "record_id" in df.columns else df[["gesture_label"]].copy()
        df_flags["any_outlier"] = False

        for feat in FEATURE_NAMES:
            if feat not in df.columns:
                continue
            series = df[feat].dropna()
            fences = self.compute_fences(series)
            
            is_outlier = (df[feat] < fences["lower"]) | (df[feat] > fences["upper"])
            flag_count = int(is_outlier.sum())
            flag_pct = round((flag_count / n_total) * 100.0, 2) if n_total > 0 else 0.0

            df_flags[f"{feat}_outlier"] = is_outlier
            df_flags["any_outlier"] = df_flags["any_outlier"] | is_outlier

            summary_records.append({
                "Feature": feat,
                "Q1": round(fences["q1"], 4),
                "Q3": round(fences["q3"], 4),
                "IQR": round(fences["iqr"], 4),
                "Lower Fence": round(fences["lower"], 4),
                "Upper Fence": round(fences["upper"], 4),
                "Flagged Count": flag_count,
                "Flagged (%)": flag_pct
            })

        df_summary = pd.DataFrame(summary_records)
        return df_summary, df_flags

    def apply_treatments(self, df: pd.DataFrame, df_flags: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        A. Winsorisation: cap feature values at fences.
        B. Removal: drop rows flagged with any_outlier.
        Returns (df_original, df_winsorized, df_removed).
        """
        df_orig = df.copy()
        df_win = df.copy()

        for feat in FEATURE_NAMES:
            if feat not in df.columns:
                continue
            fences = self.compute_fences(df[feat].dropna())
            df_win[feat] = df_win[feat].clip(lower=fences["lower"], upper=fences["upper"])

        df_rem = df[~df_flags["any_outlier"]].copy()

        # Comparison metrics table
        comp_records = []
        for feat in FEATURE_NAMES:
            if feat not in df.columns:
                continue
            comp_records.append({
                "Feature": feat,
                "Original_Mean": round(float(df_orig[feat].mean()), 4),
                "Original_Std": round(float(df_orig[feat].std()), 4),
                "Winsorized_Mean": round(float(df_win[feat].mean()), 4),
                "Winsorized_Std": round(float(df_win[feat].std()), 4),
                "Removed_Mean": round(float(df_rem[feat].mean()), 4) if len(df_rem) > 0 else np.nan,
                "Removed_Std": round(float(df_rem[feat].std()), 4) if len(df_rem) > 0 else np.nan
            })
        df_comp = pd.DataFrame(comp_records)

        return df_orig, df_win, df_rem, df_comp

    def plot_before_after(self, df_orig: pd.DataFrame, df_win: pd.DataFrame):
        """Figure 6: Box plots of features before and after winsorisation."""
        feats = [f for f in FEATURE_NAMES if f in df_orig.columns and f != "confidence"]
        n_feats = len(feats)
        fig, axes = plt.subplots(nrows=2, ncols=n_feats, figsize=(2.8 * n_feats, 6), sharey=False)

        for i, feat in enumerate(feats):
            # Before
            sns.boxplot(y=df_orig[feat], ax=axes[0, i], color="#a1c9f4")
            axes[0, i].set_title(f"{feat}\n(Before)", fontsize=9, fontweight='bold')
            axes[0, i].set_ylabel("" if i > 0 else "Original Scale", fontsize=9)
            axes[0, i].tick_params(labelsize=8)

            # After
            sns.boxplot(y=df_win[feat], ax=axes[1, i], color="#8de5a1")
            axes[1, i].set_title(f"{feat}\n(Winsorized)", fontsize=9, fontweight='bold')
            axes[1, i].set_ylabel("" if i > 0 else "Winsorized Scale", fontsize=9)
            axes[1, i].tick_params(labelsize=8)

        plt.suptitle("Engineered Features: Before vs. After IQR Winsorisation Treatment", fontsize=12, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "iqr_boxplots_before_after.png", dpi=300)
        plt.close()

    def run(self, features_csv_path: Optional[str] = None):
        feat_path = Path(features_csv_path) if features_csv_path else self.feat_path
        if not feat_path.exists():
            raise FileNotFoundError(f"Features dataset does not exist: {feat_path}")

        df = pd.read_csv(feat_path)
        logger.info(f"Computing IQR fences and outlier flags for {len(df)} records...")
        df_summary, df_flags = self.detect_outliers(df)

        df_orig, df_win, df_rem, df_comp = self.apply_treatments(df, df_flags)

        # Save tables
        df_summary.to_csv(self.tables_dir / "iqr_summary.csv", index=False)
        df_flags.to_csv(self.tables_dir / "iqr_record_flags.csv", index=False)
        df_comp.to_csv(self.tables_dir / "iqr_treatment_comparison.csv", index=False)

        # Save processed versions
        proc_dir = Path(self.config.get("data", {}).get("processed_dir", "data/processed"))
        proc_dir.mkdir(parents=True, exist_ok=True)
        df_win.to_csv(proc_dir / "features_winsorized.csv", index=False)
        df_rem.to_csv(proc_dir / "features_outliers_removed.csv", index=False)

        self.plot_before_after(df_orig, df_win)
        logger.info(f"IQR outlier analysis completed. Any-outlier count: {df_flags['any_outlier'].sum()}/{len(df)}")

if __name__ == "__main__":
    detector = IQROutlierDetector()
    detector.run()

