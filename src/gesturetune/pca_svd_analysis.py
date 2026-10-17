"""
Principal Component Analysis (PCA) and Singular Value Decomposition (SVD) module.
Implements:
1. PCA via scikit-learn on standardized matrices.
2. SVD independently via NumPy (Z = U S V^T).
3. Numerical equivalence verification:
   lambda_j = s_j^2 / (n - 1)
   Principal component directions agree up to sign.
4. Eckart-Young rank-k reconstruction and error analysis:
   ||Z - Z_k||_F^2 = sum_{j>k} s_j^2
5. Diagnostic metrics: Effective rank (95% energy), Condition number (s1 / sr).
6. Scree plots, Cumulative explained variance, PC1-PC2 scatter, Loadings heatmap,
   Singular value spectrum, Reconstruction error vs. rank.
7. Evaluates across multiple scenarios (Raw, Cleaned/Standardized, Outlier-Treated).
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from . import load_config
from .preprocess import FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class PCASVDAnalysis:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.tables_dir = Path(self.config.get("outputs", {}).get("tables_dir", "outputs/tables"))
        self.figures_dir = Path(self.config.get("outputs", {}).get("figures_dir", "outputs/figures"))
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.var_threshold = float(self.config.get("analysis", {}).get("pca_variance_threshold", 0.95))

    def analyze_matrix(self, X: np.ndarray, feature_names: List[str], scenario_name: str,
                       labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run complete PCA and SVD on a matrix X (n_samples x n_features).
        X should be centred (or standardised as appropriate).
        """
        n, d = X.shape
        if n < 2 or d < 1:
            raise ValueError(f"Insufficient dimensions for PCA/SVD: {n}x{d}")

        # Ensure centered for PCA/SVD equivalence
        X_centered = X - np.mean(X, axis=0)

        # 1. Scikit-learn PCA
        pca = PCA()
        pca.fit(X_centered)
        pca_eigenvalues = pca.explained_variance_
        pca_exp_var_ratio = pca.explained_variance_ratio_
        pca_cum_var = np.cumsum(pca_exp_var_ratio)
        pca_loadings = pca.components_  # shape: (d, d)
        pca_scores = pca.transform(X_centered)

        # Components needed for 95% variance
        k_95 = int(np.argmax(pca_cum_var >= self.var_threshold) + 1) if np.any(pca_cum_var >= self.var_threshold) else d

        # 2. Independent SVD via NumPy
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        svd_eigenvalues = (S ** 2) / (n - 1)
        
        # Condition number: s_max / s_min (ignoring near-zero)
        s_nonzero = S[S > 1e-12]
        condition_number = float(S[0] / s_nonzero[-1]) if len(s_nonzero) > 0 else np.inf

        # Effective rank (number of singular values retaining 95% energy sum(s^2))
        total_energy = np.sum(S ** 2)
        cum_energy = np.cumsum(S ** 2) / total_energy if total_energy > 0 else np.zeros_like(S)
        effective_rank = int(np.argmax(cum_energy >= self.var_threshold) + 1) if np.any(cum_energy >= self.var_threshold) else len(S)

        # 3. Equivalence verification
        max_eig_diff = float(np.max(np.abs(pca_eigenvalues - svd_eigenvalues)))
        
        # Check directional alignment up to sign
        dir_alignments = []
        for j in range(min(len(pca.components_), len(Vt))):
            cos_sim = float(np.abs(np.dot(pca.components_[j], Vt[j])))
            dir_alignments.append(cos_sim)
        min_dir_alignment = float(np.min(dir_alignments))
        equivalence_passed = bool(max_eig_diff < 1e-4 and min_dir_alignment > 0.99)

        # 4. Low-rank reconstruction errors
        recon_errors = []
        theoretical_residuals = []
        ranks = list(range(1, min(d + 1, 9)))
        for k in ranks:
            # Low rank approx: X_k = U[:, :k] * S[:k] @ Vt[:k, :]
            X_k = np.dot(U[:, :k] * S[:k], Vt[:k, :])
            err_frob_sq = float(np.sum((X_centered - X_k) ** 2))
            recon_errors.append(err_frob_sq)
            # Theoretical discarded singular values energy
            discarded_energy = float(np.sum(S[k:] ** 2)) if k < len(S) else 0.0
            theoretical_residuals.append(discarded_energy)

        # 5. Silhouette score in PC1-PC2 space if labels provided
        sil_score = None
        if labels is not None and len(np.unique(labels)) > 1 and n > len(np.unique(labels)):
            try:
                sil_score = float(silhouette_score(pca_scores[:, :2], labels))
            except Exception:
                sil_score = None

        return {
            "scenario": scenario_name,
            "n_samples": n,
            "n_features": d,
            "pca_eigenvalues": pca_eigenvalues,
            "explained_variance_ratio": pca_exp_var_ratio,
            "cumulative_variance": pca_cum_var,
            "k_95_variance": k_95,
            "pca_loadings": pca_loadings,
            "pca_scores": pca_scores,
            "singular_values": S,
            "svd_eigenvalues": svd_eigenvalues,
            "condition_number": condition_number,
            "effective_rank": effective_rank,
            "max_eig_diff": max_eig_diff,
            "min_dir_alignment": min_dir_alignment,
            "equivalence_passed": equivalence_passed,
            "ranks": ranks,
            "reconstruction_errors": recon_errors,
            "theoretical_residuals": theoretical_residuals,
            "silhouette_score_pc1_pc2": sil_score
        }

    def plot_pca_scree_and_cumvar(self, res: Dict[str, Any]):
        """Figure 7: Scree plot and cumulative explained variance."""
        exp_var = res["explained_variance_ratio"] * 100.0
        cum_var = res["cumulative_variance"] * 100.0
        k = len(exp_var)
        comps = [f"PC{i+1}" for i in range(k)]

        fig, ax1 = plt.subplots(figsize=(8.5, 4.5))

        color = '#1f77b4'
        ax1.set_xlabel('Principal Component', fontsize=11)
        ax1.set_ylabel('Individual Explained Variance (%)', color=color, fontsize=11)
        bars = ax1.bar(comps, exp_var, color=color, alpha=0.7, label='Individual Variance')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = '#d62728'
        ax2.set_ylabel('Cumulative Explained Variance (%)', color=color, fontsize=11)
        lines = ax2.plot(comps, cum_var, color=color, marker='o', linewidth=2, label='Cumulative Variance')
        ax2.axhline(95, color='gray', linestyle='--', alpha=0.7, label='95% Threshold')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim([0, 105])

        plt.title(f"Scree Plot & Cumulative Explained Variance ({res['scenario']})", fontsize=12, fontweight='bold', pad=12)
        fig.tight_layout()
        plt.savefig(self.figures_dir / f"scree_plot_{res['scenario'].lower().replace(' ', '_')}.png", dpi=300)
        plt.close()

    def plot_pc1_pc2_scatter(self, res: Dict[str, Any], labels: np.ndarray, filename_suffix: str = ""):
        """Figure 8: PC1 vs PC2 scatter plot coloured by gesture."""
        scores = res["pca_scores"]
        df_plot = pd.DataFrame({
            "PC1": scores[:, 0],
            "PC2": scores[:, 1],
            "Gesture": labels
        })

        plt.figure(figsize=(9, 6.5))
        sns.scatterplot(data=df_plot, x="PC1", y="PC2", hue="Gesture", style="Gesture", s=55, alpha=0.85, palette="tab10")
        
        pc1_pct = res["explained_variance_ratio"][0] * 100.0
        pc2_pct = res["explained_variance_ratio"][1] * 100.0
        plt.xlabel(f"PC1 ({pc1_pct:.1f}% Variance)", fontsize=11)
        plt.ylabel(f"PC2 ({pc2_pct:.1f}% Variance)", fontsize=11)
        plt.title(f"Gestures in the PC1-PC2 Plane ({res['scenario']})", fontsize=12, fontweight='bold', pad=12)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        
        fname = f"pc1_pc2_scatter_{filename_suffix}.png" if filename_suffix else f"pc1_pc2_scatter_{res['scenario'].lower().replace(' ', '_')}.png"
        plt.savefig(self.figures_dir / fname, dpi=300)
        plt.close()

    def plot_loadings_heatmap(self, res: Dict[str, Any], feature_names: List[str]):
        """Figure 9: Feature loadings on leading components."""
        loadings = res["pca_loadings"][:min(5, len(res["pca_loadings"])), :]
        df_loadings = pd.DataFrame(loadings, index=[f"PC{i+1}" for i in range(len(loadings))], columns=feature_names)

        plt.figure(figsize=(9.5, 4.5))
        sns.heatmap(df_loadings, annot=True, fmt=".2f", cmap="vlag", center=0, cbar_kws={'label': 'Loading Coefficient'})
        plt.title(f"Feature Loadings on Leading Principal Components ({res['scenario']})", fontsize=12, fontweight='bold', pad=12)
        plt.xlabel("Features", fontsize=10)
        plt.ylabel("Principal Components", fontsize=10)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(self.figures_dir / f"loadings_heatmap_{res['scenario'].lower().replace(' ', '_')}.png", dpi=300)
        plt.close()

    def plot_svd_diagnostics(self, res: Dict[str, Any]):
        """Figure 10: Singular value spectrum and reconstruction error vs rank."""
        S = res["singular_values"]
        ranks = res["ranks"]
        recon_err = res["reconstruction_errors"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        # Spectrum
        ax1.plot(range(1, len(S) + 1), S, marker='s', color='#2ca02c', linewidth=2)
        ax1.set_title("Singular Value Spectrum", fontsize=11, fontweight='bold')
        ax1.set_xlabel("Singular Component Index", fontsize=10)
        ax1.set_ylabel("Singular Value s_k", fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Reconstruction error
        ax2.plot(ranks, recon_err, marker='o', color='#ff7f0e', linewidth=2, label="Measured ||Z - Z_k||_F^2")
        ax2.plot(ranks, res["theoretical_residuals"][:len(ranks)], linestyle='--', color='#1f77b4', label="Theoretical sum(s_j^2)")
        ax2.set_title("Rank-k Reconstruction Error", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Rank k", fontsize=10)
        ax2.set_ylabel("Squared Frobenius Error", fontsize=10)
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.suptitle(f"SVD Diagnostics ({res['scenario']})", fontsize=12, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "svd_diagnostics.png", dpi=300)
        plt.close()

    def run(self, features_csv_path: Optional[str] = None) -> Dict[str, Any]:
        feat_path = Path(features_csv_path) if features_csv_path else Path(self.config.get("data", {}).get("features_dataset_path", "data/gesture_features.csv"))
        proc_dir = Path(self.config.get("data", {}).get("processed_dir", "data/processed"))

        df_feat = pd.read_csv(feat_path)
        avail_features = [f for f in FEATURE_NAMES if f in df_feat.columns]

        # Standardise for Scenario B (Cleaned + Standardised)
        X_raw = df_feat[avail_features].values
        labels = df_feat["gesture_label"].values

        # Compute Scenario B
        means = np.mean(X_raw, axis=0)
        stds = np.std(X_raw, axis=0)
        stds[stds < 1e-6] = 1.0
        X_std = (X_raw - means) / stds

        logger.info("Executing PCA and SVD on Scenario B (Standardised features)...")
        res_b = self.analyze_matrix(X_std, avail_features, scenario_name="Scenario B (Cleaned and Standardised)", labels=labels)

        # Save Table VIII: Explained Variance
        df_table_viii = pd.DataFrame({
            "Component": [f"PC{i+1}" for i in range(len(res_b["pca_eigenvalues"]))],
            "Eigenvalue": [round(v, 4) for v in res_b["pca_eigenvalues"]],
            "Explained Variance (%)": [round(v * 100.0, 2) for v in res_b["explained_variance_ratio"]],
            "Cumulative (%)": [round(v * 100.0, 2) for v in res_b["cumulative_variance"]]
        })
        df_table_viii.to_csv(self.tables_dir / "pca_explained_variance_scenario_b.csv", index=False)

        # Save Table IX: SVD Singular Values and Reconstruction Error
        df_table_ix = pd.DataFrame({
            "Rank k": res_b["ranks"],
            "Singular Value s_k": [round(s, 4) for s in res_b["singular_values"][:len(res_b["ranks"])]],
            "s_k^2 / (n - 1)": [round(v, 4) for v in res_b["svd_eigenvalues"][:len(res_b["ranks"])]],
            "Reconstruction Error (rank k)": [round(e, 4) for e in res_b["reconstruction_errors"]],
            "Theoretical Residual": [round(t, 4) for t in res_b["theoretical_residuals"][:len(res_b["ranks"])]]
        })
        df_table_ix.to_csv(self.tables_dir / "svd_singular_values_and_reconstruction_scenario_b.csv", index=False)

        # Equivalence summary CSV
        df_equiv = pd.DataFrame([{
            "Max Absolute Eigenvalue Difference": res_b["max_eig_diff"],
            "Min Direction Cosine Agreement": res_b["min_dir_alignment"],
            "Condition Number": res_b["condition_number"],
            "Effective Rank (95% Energy)": res_b["effective_rank"],
            "PCA Components for 95% Variance": res_b["k_95_variance"],
            "Equivalence Passed": res_b["equivalence_passed"]
        }])
        df_equiv.to_csv(self.tables_dir / "pca_svd_equivalence_check.csv", index=False)

        # Generate plots
        self.plot_pca_scree_and_cumvar(res_b)
        self.plot_pc1_pc2_scatter(res_b, labels, filename_suffix="scenario_b")
        self.plot_loadings_heatmap(res_b, avail_features)
        self.plot_svd_diagnostics(res_b)

        logger.info(f"PCA/SVD equivalence check passed: {res_b['equivalence_passed']} (max diff: {res_b['max_eig_diff']:.6e})")
        return res_b

if __name__ == "__main__":
    analyzer = PCASVDAnalysis()
    analyzer.run()

