# GestureTune: Gesture-Controlled Music Playback with Landmark-Based Data Analytics

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20API-brightgreen.svg)](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-orange.svg)](https://opencv.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-PCA%20%7C%20SVD%20%7C%20IsoForest-blue.svg)](https://scikit-learn.org/)

GestureTune is a vision-based human-computer interaction system that enables touch-free music playback control using real-time hand gestures captured via a standard webcam. Simultaneously, it serves as a reproducible data-analytics research testbed, transforming recorded landmark interactions into a 63-dimensional coordinate space to evaluate the mathematical effects of data cleaning, normalisation, standardisation, IQR outlier treatments, PCA, SVD, and Isolation Forest anomaly detection.

---

## 1. System Architecture and Features

### 1.1 Real-Time Gesture Music Controller
- **Modern Anti-Aliased Desktop UI**: Spotify-inspired aesthetic featuring a dark charcoal canvas (`#121214`), subtle violet/magenta accents (`#D946EF`), rounded cards, crisp TrueType typography via Windows Segoe UI with fallback caching, dedicated camera monitor with aspect-ratio preservation, live Now Playing & volume telemetry cards, status chip bar, and dynamic gesture reference footer.
- **Accurate Hybrid Recognition**:
  - High-precision personalized Random Forest classifier trained on translation- and scale-invariant 83D landmark features.
  - Seamless fallback to geometric rule-based recognition when no personalized profile is loaded or when confidence is below threshold.
  - Responsive temporal smoothing window and deliberate release/re-arm state machine preventing repeated triggers.
- **Webcam Tracking**: MediaPipe Hand Landmarker Tasks API tracks 21 three-dimensional landmarks in real-time.
- **Eight-Gesture Vocabulary**:
  - `Index finger`: Play / pause toggle
  - `Two fingers`: Mute / unmute toggle
  - `Point left`: Previous track
  - `Point right`: Next track
  - `Open palm`: Rotational volume control (in rotational mode) / Volume up (in discrete mode)
  - `Fist`: Volume down (in discrete mode)
  - `Thumbs up`: Like current track (persisted to `data/liked_tracks.json`)
  - `Thumbs down`: Unlike current track (removes from liked list)
- **Interaction Control**: 5-frame consecutive debouncing, 1.0s discrete cooldown, 0.5s volume repeat interval, horizontally mirrored camera preview, and safe release/re-arm state machine (resting hand cannot cause accidental re-triggering).
- **Audio Engine**: `pygame.mixer` supporting local `.mp3`, `.wav`, and `.ogg` files, with automatic fallback to virtual demo playback when audio files or audio hardware are absent.

### 1.2 Data Analytics & Dimensionality Reduction Pipeline
- **Dataset Schema**: Append-safe 67-column schema (4 metadata columns + 63 coordinates `landmark_0_x` ... `landmark_20_z`). Deliberate structural missingness is recorded with `hand_detected = 0` and coordinate `NaN`s for detection-failure analysis.
- **Preprocessing**: Translation to wrist origin $p_0$, palm-length scaling by $s = \|p_9 - p_0\|_2$, feature engineering of 8 static/kinematic features, skewness analysis with `log1p` transformation, and z-score standardisation.
- **IQR Outlier Detection**: Computes Tukey fences ($Q_1 - 1.5\text{IQR}, Q_3 + 1.5\text{IQR}$) globally and per gesture. Compares Winsorisation vs. sensitive observation removal.
- **PCA and Independent SVD Equivalence**: Verifies $\lambda_j = \frac{s_j^2}{n - 1}$ and eigenvector/singular vector directional alignment within numerical tolerance ($< 10^{-14}$). Computes singular value spectrum, condition number, effective rank (95% energy), and Eckart-Young rank-$k$ low-rank Frobenius reconstruction errors $\|Z - Z_k\|_F^2 = \sum_{j>k} s_j^2$.
- **Isolation Forest vs. IQR Overlap**: Identifies multivariate anomalies using path length scores $s(x, n) = 2^{-E[h(x)] / c(n)}$ and computes Jaccard index and contingency breakdown against univariate IQR flags.
- **Before-and-After Processing Comparison**: Compares Stage A (Raw unscaled), Stage B (Cleaned & Standardised), Stage C (Winsorised), and Stage D (Outlier-removed) across explained variance, silhouette separation, and k-NN cross-validated classification accuracy.

---

## 2. Directory Structure

```text
GesTune_v2/
├── config/
│   └── config.yaml                     # Central configuration parameters
├── data/
│   ├── raw_landmarks.csv               # Raw 67-column captured dataset
│   ├── benchmark_landmarks.csv         # Benchmark synthetic evaluation dataset
│   ├── gesture_features.csv            # Engineered 8-feature dataset
│   ├── liked_tracks.json               # Persistent record of liked tracks
│   └── processed/
│       ├── normalized_landmarks.csv    # Wrist-relative, palm-scaled coordinates
│       ├── standardized_features.csv   # Z-scored features
│       ├── features_winsorized.csv     # IQR-winsorised dataset
│       └── features_outliers_removed.csv # Outlier-removed sensitivity dataset
├── models/
│   └── hand_landmarker.task            # MediaPipe Tasks vision model asset
├── music/
│   └── README.md                       # Instructions for adding local tracks
├── outputs/
│   ├── processing_stage_comparison.csv # Consolidated multi-stage comparison
│   ├── tables/                         # Generated analytical CSV summaries
│   │   ├── descriptive_statistics.csv
│   │   ├── iqr_summary.csv
│   │   ├── iqr_record_flags.csv
│   │   ├── iqr_treatment_comparison.csv
│   │   ├── pca_explained_variance_scenario_b.csv
│   │   ├── svd_singular_values_and_reconstruction_scenario_b.csv
│   │   ├── pca_svd_equivalence_check.csv
│   │   ├── isolation_forest_flags.csv
│   │   ├── anomaly_method_comparison.csv
│   │   ├── anomaly_breakdown_by_gesture.csv
│   │   ├── anomaly_jaccard_overlap.csv
│   │   └── gesture_accuracy_table.csv
│   ├── figures/                        # Generated high-resolution plots
│   │   ├── gesture_frequency.png
│   │   ├── movement_speed_histogram.png
│   │   ├── movement_speed_boxplot.png
│   │   ├── correlation_heatmap.png
│   │   ├── iqr_boxplots_before_after.png
│   │   ├── scree_plot_scenario_b.png
│   │   ├── pc1_pc2_scatter_scenario_b.png
│   │   ├── loadings_heatmap_scenario_b.png
│   │   ├── svd_diagnostics.png
│   │   ├── anomaly_score_distribution.png
│   │   ├── anomaly_pc_overlap.png
│   │   ├── processing_stages_pc_comparison.png
│   │   └── confusion_matrix.png
│   └── reports/
│       └── audit_report.json           # Machine-readable data audit
├── src/
│   └── gesturetune/
│       ├── __init__.py
│       ├── capture.py                  # Live data recording CLI tool
│       ├── hand_tracker.py             # MediaPipe Tasks tracking wrapper
│       ├── gesture_recognition.py      # Rule-based 7-gesture recogniser
│       ├── gesture_controller.py       # Debouncing, cooldown & dispatch
│       ├── music_controller.py         # Pygame audio controller & player state
│       ├── gesture_player.py           # Live interactive GUI application
│       ├── preprocess.py               # Audit, cleaning, normalisation, features
│       ├── eda_summary.py              # Statistical distributions and EDA
│       ├── outlier_iqr.py              # IQR outlier detection and treatments
│       ├── pca_svd_analysis.py         # PCA & SVD analysis and equivalence check
│       ├── isolation_forest.py         # Isolation Forest anomaly scoring
│       ├── evaluation.py               # Multi-stage comparison & accuracy metrics
│       └── synthetic_data.py           # Benchmark fixture generator
├── scripts/
│   ├── generate_benchmark_data.py      # Generates benchmark dataset
│   └── run_pipeline.py                 # Master orchestrator script
├── tests/                              # Automated test suite (17 tests)
│   ├── test_capture_schema.py
│   ├── test_gesture_recognition.py
│   ├── test_preprocessing.py
│   ├── test_iqr.py
│   ├── test_pca_svd.py
│   ├── test_isolation_forest.py
│   └── test_pipeline.py
├── conftest.py
├── requirements.txt
└── README.md
```

---

## 3. Installation and Setup

### 3.1 Prerequisites
- Python 3.11 (64-bit) installed on Windows.
- Standard USB webcam or integrated camera.

### 3.2 Setup Virtual Environment
```powershell
# Create Python 3.11 virtual environment
py -3.11 -m venv .venv

# Activate environment
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Execution Workflow

### 4.1 Launch the Live Gesture Music Player (With Automatic First-Run Wizard)
```powershell
.venv\Scripts\python -m src.gesturetune.gesture_player
```
- **Volume Mode Options**:
  ```powershell
  # Default: Calibrated continuous rotational volume control
  .venv\Scripts\python -m src.gesturetune.gesture_player --volume-mode rotational

  # Fallback: Discrete open palm / fist volume steps
  .venv\Scripts\python -m src.gesturetune.gesture_player --volume-mode discrete
  ```
- **Interactive Controls**:
  - `v`: Toggle volume mode between **Rotational (Calibrated)** and **Discrete (Palm/Fist)** in real time.
  - `r`: Trigger on-demand recalibration wizard (calibrates all 8 gestures + rotational angles).
  - `q` or `ESC`: Quit safely.
- **Eight-Gesture Vocabulary**:
  - `Index finger`: Play / pause toggle
  - `Two fingers`: Mute / unmute toggle
  - `Point left`: Previous track
  - `Point right`: Next track
  - `Open palm`: Volume up (in discrete mode)
  - `Fist`: Volume down (in discrete mode)
  - `Thumbs up`: Like current track (persisted to `data/liked_tracks.json`)
  - `Thumbs down`: Unlike current track (removes from liked list)
- **Modern UI Features**:
  - **Header**: GestureTune brand badge, camera connection pill, recognition model tag (`ML PERSONALIZED` or `RULE FALLBACK`), and live FPS counter.
  - **Left Panel (Camera Monitor)**: Live webcam feed framed with accent borders, letterboxed to preserve aspect ratio, with landmark skeleton overlay and a subtle guide when no hand is present.
  - **Right Panel (Now Playing)**: Clean track title, playback state badge (`PLAYING`, `PAUSED`, `STOPPED`), liked indicator, horizontal volume slider with percentage, active volume mode indicator, and rotational telemetry dial.
  - **Status Chips Bar**: Dedicated status metrics for Raw Gesture, Stabilized Gesture, Confidence %, Interaction State Machine (`NEUTRAL`, `STABLE_POSE`, `WAITING_FOR_RELEASE`, `COOLDOWN`), and Last Executed Action.
  - **Reference Footer**: Multi-tier gesture reference that dynamically adapts to the selected volume mode (`Rotational` vs `Discrete`), alongside clear keyboard shortcuts (`V`, `R`, `Q`/`ESC`).

### 4.2 Record a Labelled Landmark Dataset
```powershell
.venv\Scripts\python -m src.gesturetune.capture
```
- Keys `1`-`8`: Select active gesture label (`1`: index_finger, `2`: two_fingers, `3`: point_left, `4`: point_right, `5`: open_palm, `6`: fist, `7`: thumbs_up, `8`: thumbs_down).
- Press `SPACE`: Record one observation (saves to `data/raw_landmarks.csv`).
- Press `'q'`: Exit capture.

### 4.3 Run the Complete End-to-End Analytics Pipeline
```powershell
# Run on captured data:
.venv\Scripts\python scripts/run_pipeline.py --data data/raw_landmarks.csv

# Or run on the included benchmark dataset:
.venv\Scripts\python scripts/run_pipeline.py --data data/benchmark_landmarks.csv
```

### 4.4 Run Automated Tests
```powershell
.venv\Scripts\pytest -v
```

---

## 5. Experimental Results and Key Findings

Running the analytics pipeline on the 273-observation dataset (`data/benchmark_landmarks.csv`) demonstrated:

1. **Preprocessing & Scaling Impact (Table XIII)**:
   - **Stage A (Raw unscaled)**: A single dominant high-variance feature accounted for **99.43%** of PC1 variance, with condition number **534.55** and negative silhouette score (**-0.2707**).
   - **Stage B (Cleaned & Standardised)**: Variance was distributed across **6 components** to reach 95% cumulative explained variance. PC1 explained **38.72%**, the condition number dropped to **11.23**, and cross-validated k-NN accuracy reached **100%**.
2. **PCA / SVD Equivalence Check (Table IX)**:
   - Maximum absolute difference between PCA eigenvalues and SVD singular values $\frac{s_k^2}{n-1}$: **$1.776 \times 10^{-15}$** (Passed within numerical precision).
   - Directional cosine agreement between eigenvectors and singular vectors: **$> 0.9999$** (Sign-aligned equivalence confirmed).
   - Eckart-Young reconstruction theorem verified: measured Frobenius residual $\|Z - Z_k\|_F^2$ matched theoretical discarded singular energy $\sum_{j>k} s_j^2$ at all ranks.
3. **Anomaly & Outlier Screening (Table X)**:
   - Univariate IQR flagged extreme points in spread and speed (61 records), while multivariate Isolation Forest flagged isolated geometric anomalies (13 records), yielding a Jaccard overlap of **0.1935**, confirming that univariate feature outliers and multivariate subspace anomalies detect complementary phenomena.
4. **Gesture Recognition Accuracy (Table XIV)**:
   - Rule-based geometric evaluation achieved **100.0%** accuracy across 6 gestures (fist, index finger, point left, point right, thumbs up, two fingers), with controlled noise in open palm yielding an overall clean separation.

