"""
Master pipeline runner script (run_pipeline.py).
Executes the complete GestureTune analytics workflow end to end:
1. Preprocessing & Feature Engineering (preprocess.py)
2. Exploratory Data Analysis & Figures (eda_summary.py)
3. IQR Outlier Detection & Treatments (outlier_iqr.py)
4. PCA & SVD Analysis & Equivalence check (pca_svd_analysis.py)
5. Isolation Forest & Anomaly comparison (isolation_forest.py)
6. Multi-Stage Evaluation & Confusion Matrix (evaluation.py)
"""

import sys
import logging
from pathlib import Path
import argparse

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gesturetune.preprocess import PreprocessingPipeline
from src.gesturetune.eda_summary import EDASummary
from src.gesturetune.outlier_iqr import IQROutlierDetector
from src.gesturetune.pca_svd_analysis import PCASVDAnalysis
from src.gesturetune.isolation_forest import AnomalyAnalyzer
from src.gesturetune.evaluation import ModelEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_pipeline(raw_csv: str = "data/raw_landmarks.csv", config_path: str = "config/config.yaml"):
    logger.info("=" * 65)
    logger.info("STARTING GESTURETUNE END-TO-END ANALYTICS PIPELINE")
    logger.info(f"Raw Input Dataset: {raw_csv}")
    logger.info("=" * 65)

    raw_path = Path(raw_csv)
    if not raw_path.exists():
        logger.error(f"Cannot run pipeline: Raw dataset '{raw_path}' does not exist.")
        logger.error("Please record data using 'python -m src.gesturetune.capture' or provide a dataset.")
        sys.exit(1)

    # Stage 1: Preprocessing & Feature Engineering
    logger.info("\n--- [1/6] Preprocessing and Feature Engineering ---")
    preprocessor = PreprocessingPipeline(config_path=config_path)
    prep_res = preprocessor.run(raw_csv_path=raw_csv)

    # Stage 2: Exploratory Data Analysis
    logger.info("\n--- [2/6] Exploratory Data Analysis (EDA) ---")
    eda = EDASummary(config_path=config_path)
    eda.run()

    # Stage 3: IQR Outlier Detection
    logger.info("\n--- [3/6] IQR Outlier Detection and Treatments ---")
    iqr = IQROutlierDetector(config_path=config_path)
    iqr.run()

    # Stage 4: PCA and SVD Analysis
    logger.info("\n--- [4/6] PCA and SVD Dimensionality Reduction & Diagnostics ---")
    pca_svd = PCASVDAnalysis(config_path=config_path)
    pca_svd.run()

    # Stage 5: Isolation Forest Anomaly Detection
    logger.info("\n--- [5/6] Isolation Forest Multivariate Anomaly Analysis ---")
    iso = AnomalyAnalyzer(config_path=config_path)
    iso.run()

    # Stage 6: Evaluation & Before/After Comparison
    logger.info("\n--- [6/6] Multi-Stage Comparison & Recognition Evaluation ---")
    evaluator = ModelEvaluator(config_path=config_path)
    evaluator.run(raw_csv_path=raw_csv)

    logger.info("=" * 65)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("Outputs generated in:")
    logger.info("  - Tables:  outputs/tables/")
    logger.info("  - Figures: outputs/figures/")
    logger.info("  - Reports: outputs/reports/")
    logger.info("  - Consolidated Comparison: outputs/processing_stage_comparison.csv")
    logger.info("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the GestureTune analytics pipeline.")
    parser.add_argument("--data", default="data/raw_landmarks.csv", help="Path to raw_landmarks.csv")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    run_pipeline(raw_csv=args.data, config_path=args.config)

