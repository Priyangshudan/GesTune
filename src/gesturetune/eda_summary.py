"""
Exploratory Data Analysis module (eda_summary.py).
Generates descriptive statistics CSV reports and four primary figures:
1. outputs/figures/gesture_frequency.png
2. outputs/figures/movement_speed_histogram.png
3. outputs/figures/movement_speed_boxplot.png
4. outputs/figures/correlation_heatmap.png
and corresponding tables in outputs/tables/
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from . import load_config
from .preprocess import FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class EDASummary:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.feat_path = Path(self.config.get("data", {}).get("features_dataset_path", "data/gesture_features.csv"))
        self.raw_path = Path(self.config.get("data", {}).get("raw_dataset_path", "data/raw_landmarks.csv"))
        self.tables_dir = Path(self.config.get("outputs", {}).get("tables_dir", "outputs/tables"))
        self.figures_dir = Path(self.config.get("outputs", {}).get("figures_dir", "outputs/figures"))
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def generate_descriptive_stats(self, df_feat: pd.DataFrame) -> pd.DataFrame:
        """Compute Table VI: Mean, Median, Std dev, Min, Max, and Missing count."""
        records = []
        for feat in FEATURE_NAMES:
            if feat in df_feat.columns:
                series = df_feat[feat].dropna()
                records.append({
                    "Feature": feat,
                    "Mean": round(float(series.mean()), 4),
                    "Median": round(float(series.median()), 4),
                    "Std dev": round(float(series.std()), 4),
                    "Min": round(float(series.min()), 4),
                    "Max": round(float(series.max()), 4),
                    "Missing": int(df_feat[feat].isnull().sum())
                })
        df_stats = pd.DataFrame(records)
        df_stats.to_csv(self.tables_dir / "descriptive_statistics.csv", index=False)
        return df_stats

    def plot_gesture_frequency(self, df_raw: pd.DataFrame, df_feat: pd.DataFrame):
        """Figure 3: Gesture frequency distribution (Overall and valid)."""
        plt.figure(figsize=(9, 5))
        counts = df_feat["gesture_label"].value_counts().reset_index()
        counts.columns = ["Gesture", "Count"]

        palette = sns.color_palette("viridis", len(counts))
        barplot = sns.barplot(data=counts, x="Gesture", y="Count", palette=palette)
        plt.title("Gesture Frequency Distribution (Valid Hand Observations)", fontsize=13, fontweight='bold', pad=12)
        plt.xlabel("Gesture Label", fontsize=11)
        plt.ylabel("Number of Observations", fontsize=11)
        plt.xticks(rotation=25, ha="right")

        for p in barplot.patches:
            height = p.get_height()
            barplot.annotate(f"{int(height)}",
                             (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='bottom', fontsize=10, xytext=(0, 3),
                             textcoords='offset points')

        plt.tight_layout()
        plt.savefig(self.figures_dir / "gesture_frequency.png", dpi=300)
        plt.close()

    def plot_movement_speed(self, df_feat: pd.DataFrame):
        """Figure 4: Distribution of movement speed overall (histogram) and by gesture (boxplot)."""
        if "movement_speed" not in df_feat.columns:
            return

        # Histogram
        plt.figure(figsize=(8, 4.5))
        sns.histplot(df_feat["movement_speed"], bins=30, kde=True, color="#2b5c8f")
        plt.title("Distribution of Movement Speed (Overall)", fontsize=12, fontweight='bold', pad=10)
        plt.xlabel("Movement Speed (Norm. screen units / sec)", fontsize=10)
        plt.ylabel("Frequency", fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "movement_speed_histogram.png", dpi=300)
        plt.close()

        # Boxplot by gesture
        plt.figure(figsize=(9, 5))
        sns.boxplot(data=df_feat, x="gesture_label", y="movement_speed", palette="Set2")
        plt.title("Movement Speed by Gesture Class", fontsize=12, fontweight='bold', pad=10)
        plt.xlabel("Gesture Label", fontsize=10)
        plt.ylabel("Movement Speed (Norm. screen units / sec)", fontsize=10)
        plt.xticks(rotation=25, ha="right")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "movement_speed_boxplot.png", dpi=300)
        plt.close()

    def plot_correlation_heatmap(self, df_feat: pd.DataFrame):
        """Figure 5: Pearson correlation heatmap among engineered features."""
        avail_features = [f for f in FEATURE_NAMES if f in df_feat.columns]
        corr_matrix = df_feat[avail_features].corr(method="pearson")
        corr_matrix.to_csv(self.tables_dir / "correlation_matrix.csv")

        plt.figure(figsize=(8.5, 7))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1.0, vmax=1.0,
                    linewidths=0.5, cbar_kws={'label': 'Pearson Correlation'})
        plt.title("Pearson Correlation Heatmap of Engineered Features", fontsize=12, fontweight='bold', pad=12)
        plt.xticks(rotation=35, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "correlation_heatmap.png", dpi=300)
        plt.close()

    def run(self, features_csv_path: Optional[str] = None):
        feat_path = Path(features_csv_path) if features_csv_path else self.feat_path
        if not feat_path.exists():
            raise FileNotFoundError(f"Features dataset does not exist: {feat_path}")

        df_feat = pd.read_csv(feat_path)
        df_raw = pd.read_csv(self.raw_path) if self.raw_path.exists() else df_feat

        logger.info(f"Generating descriptive statistics for {len(df_feat)} records...")
        self.generate_descriptive_stats(df_feat)

        logger.info("Generating EDA figures...")
        self.plot_gesture_frequency(df_raw, df_feat)
        self.plot_movement_speed(df_feat)
        self.plot_correlation_heatmap(df_feat)
        logger.info("EDA summary completed successfully.")

if __name__ == "__main__":
    eda = EDASummary()
    eda.run()

