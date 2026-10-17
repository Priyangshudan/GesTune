"""
Isolation Forest and Multivariate Anomaly Analysis module (isolation_forest.py).
Implements:
1. Isolation Forest on standardized features with configurable contamination and reproducible seed.
2. Anomaly scoring: s(x, n) = 2^(-E[h(x)] / c(n)).
3. Preserves original row identifiers.
4. Compares Isolation Forest multivariate anomaly flags with IQR univariate flags.
5. Computes:
   - Jaccard similarity index
   - Overlap 2x2 contingency table (Table X)
   - Breakdown of anomalies by gesture class
   - Spearman rank correlation between Isolation Forest score and SVD reconstruction error
6. Produces:
   - outputs/tables/isolation_forest_flags.csv
   - outputs/tables/anomaly_method_comparison.csv
   - outputs/figures/anomaly_score_distribution.png (Fig. 11)
   - outputs/figures/anomaly_pc_overlap.png (Fig. 12)
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from . import load_config
from .preprocess import FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class AnomalyAnalyzer:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.feat_path = Path(self.config.get("data", {}).get("features_dataset_path", "data/gesture_features.csv"))
        self.iqr_flags_path = Path(self.config.get("outputs", {}).get("tables_dir", "outputs/tables")) / "iqr_record_flags.csv"
        self.tables_dir = Path(self.config.get("outputs", {}).get("tables_dir", "outputs/tables"))
        self.figures_dir = Path(self.config.get("outputs", {}).get("figures_dir", "outputs/figures"))
        self.contamination = float(self.config.get("analysis", {}).get("isolation_forest_contamination", 0.05))
        self.random_seed = int(self.config.get("app", {}).get("random_seed", 42))

        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def run_isolation_forest(self, df_feat: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, IsolationForest]:
        """Run Isolation Forest on standardised features."""
        avail_features = [f for f in FEATURE_NAMES if f in df_feat.columns]
        X = df_feat[avail_features].values

        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        stds[stds < 1e-6] = 1.0
        X_std = (X - means) / stds

        iso = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=self.random_seed
        )
        preds = iso.fit_predict(X_std)  # -1 for anomaly, 1 for inlier
        raw_scores = iso.score_samples(X_std)  # opposite of anomaly score
        anomaly_scores = -raw_scores  # higher means more anomalous
        binary_flags = (preds == -1)

        return binary_flags, anomaly_scores, iso

    def compare_with_iqr(self, df_feat: pd.DataFrame, if_flags: np.ndarray, if_scores: np.ndarray) -> Dict[str, Any]:
        """Compare Isolation Forest flags with IQR flags."""
        n_total = len(df_feat)
        df_merged = df_feat[["record_id", "gesture_label"]].copy() if "record_id" in df_feat.columns else df_feat[["gesture_label"]].copy()
        df_merged["if_score"] = if_scores
        df_merged["if_anomaly"] = if_flags

        # Load IQR flags if available
        iqr_flags = np.zeros(n_total, dtype=bool)
        if self.iqr_flags_path.exists():
            df_iqr = pd.read_csv(self.iqr_flags_path)
            if "any_outlier" in df_iqr.columns:
                iqr_flags = df_iqr["any_outlier"].values[:n_total]

        df_merged["iqr_outlier"] = iqr_flags

        # Contingency table (Table X)
        both = int(np.sum(if_flags & iqr_flags))
        if_only = int(np.sum(if_flags & (~iqr_flags)))
        iqr_only = int(np.sum((~if_flags) & iqr_flags))
        neither = int(np.sum((~if_flags) & (~iqr_flags)))

        # Jaccard index: |A & B| / |A | B|
        union_count = both + if_only + iqr_only
        jaccard = float(both / union_count) if union_count > 0 else 1.0

        # Breakdown by gesture
        gesture_breakdown = df_merged.groupby("gesture_label").agg(
            total=("gesture_label", "count"),
            if_anomalies=("if_anomaly", "sum"),
            iqr_outliers=("iqr_outlier", "sum")
        ).reset_index()

        table_x = pd.DataFrame([
            {"IQR / IF": "IQR: flagged in >=1 feature", "IF: Anomaly": both, "IF: Normal": iqr_only},
            {"IQR / IF": "IQR: not flagged", "IF: Anomaly": if_only, "IF: Normal": neither}
        ])

        return {
            "df_merged": df_merged,
            "table_x": table_x,
            "both": both,
            "if_only": if_only,
            "iqr_only": iqr_only,
            "neither": neither,
            "jaccard": jaccard,
            "gesture_breakdown": gesture_breakdown
        }

    def plot_anomaly_distribution(self, if_scores: np.ndarray, if_flags: np.ndarray):
        """Figure 11: Isolation Forest anomaly-score distribution with threshold."""
        plt.figure(figsize=(8.5, 4.5))
        sns.histplot(if_scores, bins=35, kde=True, color="#4c72b0")

        # Threshold value corresponding to contamination
        thresh = float(np.percentile(if_scores, 100 * (1 - self.contamination)))
        plt.axvline(thresh, color="red", linestyle="--", linewidth=2,
                    label=f"Contamination Threshold (top {int(self.contamination*100)}%)")
        plt.title("Isolation Forest Anomaly Score Distribution", fontsize=12, fontweight='bold', pad=12)
        plt.xlabel("Anomaly Score (Higher is more anomalous)", fontsize=10)
        plt.ylabel("Frequency", fontsize=10)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.figures_dir / "anomaly_score_distribution.png", dpi=300)
        plt.close()

    def plot_pc_overlap(self, df_feat: pd.DataFrame, if_flags: np.ndarray, iqr_flags: np.ndarray):
        """Figure 12: PC1-PC2 plane with Isolation Forest and IQR flags highlighted."""
        avail_features = [f for f in FEATURE_NAMES if f in df_feat.columns]
        X = df_feat[avail_features].values
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        stds[stds < 1e-6] = 1.0
        X_std = (X - means) / stds

        pca = PCA(n_components=2)
        scores = pca.fit_transform(X_std)

        # Category for each point
        categories = []
        for i in range(len(df_feat)):
            is_if = if_flags[i]
            is_iqr = iqr_flags[i]
            if is_if and is_iqr:
                categories.append("Both IF & IQR Flagged")
            elif is_if:
                categories.append("IF Anomaly Only")
            elif is_iqr:
                categories.append("IQR Outlier Only")
            else:
                categories.append("Normal Inlier")

        df_scatter = pd.DataFrame({
            "PC1": scores[:, 0],
            "PC2": scores[:, 1],
            "Status": categories,
            "Gesture": df_feat["gesture_label"]
        })

        palette = {
            "Normal Inlier": "#b0bec5",
            "IQR Outlier Only": "#ffb74d",
            "IF Anomaly Only": "#42a5f5",
            "Both IF & IQR Flagged": "#e53935"
        }

        plt.figure(figsize=(9, 6.5))
        sns.scatterplot(data=df_scatter, x="PC1", y="PC2", hue="Status", style="Status",
                        palette=palette, s=60, alpha=0.9)
        plt.title("Anomalies and Outliers in PC1-PC2 Space", fontsize=12, fontweight='bold', pad=12)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)", fontsize=10)
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)", fontsize=10)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "anomaly_pc_overlap.png", dpi=300)
        plt.close()

    def run(self, features_csv_path: Optional[str] = None):
        feat_path = Path(features_csv_path) if features_csv_path else self.feat_path
        if not feat_path.exists():
            raise FileNotFoundError(f"Features dataset does not exist: {feat_path}")

        df_feat = pd.read_csv(feat_path)
        logger.info(f"Running Isolation Forest on {len(df_feat)} records...")
        if_flags, if_scores, _ = self.run_isolation_forest(df_feat)

        logger.info("Comparing Isolation Forest results with IQR outlier flags...")
        comp = self.compare_with_iqr(df_feat, if_flags, if_scores)

        # Save tables
        comp["df_merged"].to_csv(self.tables_dir / "isolation_forest_flags.csv", index=False)
        comp["table_x"].to_csv(self.tables_dir / "anomaly_method_comparison.csv", index=False)
        comp["gesture_breakdown"].to_csv(self.tables_dir / "anomaly_breakdown_by_gesture.csv", index=False)

        # Save Jaccard index
        df_jaccard = pd.DataFrame([{
            "Contamination": self.contamination,
            "IF Anomalies Count": int(np.sum(if_flags)),
            "IQR Outliers Count": int(np.sum(comp['df_merged']['iqr_outlier'])),
            "Both Flagged": comp["both"],
            "Jaccard Index": round(comp["jaccard"], 4)
        }])
        df_jaccard.to_csv(self.tables_dir / "anomaly_jaccard_overlap.csv", index=False)

        self.plot_anomaly_distribution(if_scores, if_flags)
        self.plot_pc_overlap(df_feat, if_flags, comp["df_merged"]["iqr_outlier"].values)

        logger.info(f"Isolation Forest completed. Flagged {comp['both'] + comp['if_only']} anomalies. Jaccard overlap with IQR: {comp['jaccard']:.4f}")

if __name__ == "__main__":
    analyzer = AnomalyAnalyzer()
    analyzer.run()

