import pytest
from pathlib import Path
import tempfile
import pandas as pd

from src.gesturetune.synthetic_data import generate_synthetic_dataset
from scripts.run_pipeline import run_pipeline

def test_full_pipeline_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_csv = Path(tmpdir) / "raw_landmarks.csv"
        df_synthetic = generate_synthetic_dataset(n_samples_per_gesture=15, no_hand_count=5, seed=42)
        df_synthetic.to_csv(raw_csv, index=False)

        # Run pipeline
        run_pipeline(raw_csv=str(raw_csv))

        # Verify tables created
        tables_dir = Path("outputs/tables")
        assert (tables_dir / "descriptive_statistics.csv").exists()
        assert (tables_dir / "iqr_summary.csv").exists()
        assert (tables_dir / "pca_explained_variance_scenario_b.csv").exists()
        assert (tables_dir / "svd_singular_values_and_reconstruction_scenario_b.csv").exists()
        assert (tables_dir / "pca_svd_equivalence_check.csv").exists()
        assert (tables_dir / "isolation_forest_flags.csv").exists()
        assert (tables_dir / "processing_stage_comparison.csv").exists()
        assert (tables_dir / "gesture_accuracy_table.csv").exists()

        # Verify figures created
        figures_dir = Path("outputs/figures")
        assert (figures_dir / "gesture_frequency.png").exists()
        assert (figures_dir / "movement_speed_histogram.png").exists()
        assert (figures_dir / "movement_speed_boxplot.png").exists()
        assert (figures_dir / "correlation_heatmap.png").exists()
        assert (figures_dir / "iqr_boxplots_before_after.png").exists()
        assert (figures_dir / "scree_plot_scenario_b_(cleaned_and_standardised).png").exists()
        assert (figures_dir / "pc1_pc2_scatter_scenario_b.png").exists()
        assert (figures_dir / "loadings_heatmap_scenario_b_(cleaned_and_standardised).png").exists()
        assert (figures_dir / "svd_diagnostics.png").exists()
        assert (figures_dir / "anomaly_score_distribution.png").exists()
        assert (figures_dir / "anomaly_pc_overlap.png").exists()
        assert (figures_dir / "processing_stages_pc_comparison.png").exists()
        assert (figures_dir / "confusion_matrix.png").exists()

        # Verify consolidated file
        assert Path("outputs/processing_stage_comparison.csv").exists()

