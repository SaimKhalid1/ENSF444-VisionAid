"""Command-line entry point for the VisionAid ENSF 444 final project.

Usage
-----
Run from the project root directory.

**Step 1 — first-time setup (download the dataset and install dependencies):**

    # Create and activate a virtual environment
    python -m venv .venv
    .venv\\Scripts\\activate          # Windows
    source .venv/bin/activate         # macOS / Linux

    # Install all required packages
    pip install -r requirements.txt

    # Download the faceshape dataset from GitHub into the expected location
    git clone https://github.com/dsmlr/faceshape data/external/faceshape_source

    The cloned repository should produce this directory structure:
        data/external/faceshape_source/
            published_dataset/
                heart/    (100 JPEG images)
                oblong/   (100 JPEG images)
                oval/     (100 JPEG images)
                round/    (100 JPEG images)
                square/   (100 JPEG images)

**Step 2 — run the full pipeline:**

    # Extract landmark features from the images AND train all models:
    python run_project.py --force-features

    # If the cached feature CSV already exists, skip re-extraction:
    python run_project.py

What the pipeline does
-----------------------
1. Loads each labeled image from data/external/faceshape_source/published_dataset/.
2. Detects the primary face in each image using an OpenCV Haar cascade.
3. Extracts 68 facial landmark points using the LBF facemark model
   (data/models/lbfmodel.yaml).
4. Normalizes landmark coordinates relative to the face bounding box.
5. Engineers 16 geometry ratio features (jaw widths, cheekbone ratios, etc.).
6. Saves the feature table to data/processed/landmark_features.csv.
7. Generates EDA plots (class distribution, PCA, correlation heatmap).
8. Trains four classifiers with GridSearchCV (5-fold stratified CV):
     - Logistic Regression
     - K-Nearest Neighbours
     - Random Forest
     - MLP Classifier (shallow neural network)
9. Evaluates each model on a held-out 20% test set.
10. Saves confusion matrices, classification reports, and model weights.
11. Maps the best model's predictions to eyewear frame style recommendations.

Outputs (written to results/ and models/)
------------------------------------------
results/
    class_distribution.png           — count of images per class
    pca_projection.png               — 2-D PCA scatter of all features
    geometry_correlation_heatmap.png — Pearson correlation of geometry features
    model_comparison.csv             — accuracy / F1 summary for all models
    experiment_summary.json          — best model name and accuracy
    <model>_confusion_matrix.png     — one PNG per model
    <model>_classification_report.csv— one CSV per model
    random_forest_feature_importance.png
    random_forest_top_feature_importance.csv
    recommendation_examples.csv      — 15 example frame recommendations

models/
    logistic_regression_best.joblib
    knn_best.joblib
    random_forest_best.joblib
    mlp_classifier_best.joblib

Notebooks
----------
Interactive walkthroughs are available under notebooks/:
    01_exploratory_data_analysis.ipynb  — EDA on the feature table
    02_model_training_and_evaluation.ipynb — training, tuning, and comparison

Launch with:
    jupyter notebook notebooks/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the src/ directory importable without requiring an editable install.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from visionaid import FaceShapeExperiment


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments object with the following attributes:

        force_features (bool):
            If True, re-extract landmark features from the raw images even
            if data/processed/landmark_features.csv already exists.
            Use this flag after downloading a fresh copy of the dataset.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the VisionAid face-shape classification and eyewear "
            "recommendation pipeline.\n\n"
            "See the module docstring at the top of this file for full "
            "setup and usage instructions."
        )
    )
    parser.add_argument(
        "--force-features",
        action="store_true",
        help=(
            "Rebuild the landmark feature CSV from the raw images even if "
            "data/processed/landmark_features.csv already exists. "
            "Requires the faceshape dataset to be present at "
            "data/external/faceshape_source/published_dataset/."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Execute the full VisionAid experiment pipeline and print a results summary.

    Returns
    -------
    int
        Exit code: 0 on success, non-zero on failure.
    """
    args = parse_args()

    # Instantiate the experiment.  All paths default to the standard project layout.
    experiment = FaceShapeExperiment()

    # run_all() orchestrates all stages: feature extraction (if needed), EDA, and
    # model training/evaluation.  All artifacts are saved to results/ and models/.
    metrics_df, summary = experiment.run_all(force_rebuild_features=args.force_features)

    # Print a human-readable results summary to the console.
    print("\n" + "=" * 60)
    print("VisionAid pipeline complete")
    print("=" * 60)
    print("\nModel comparison (sorted by test accuracy):")
    print(metrics_df.to_string(index=False))
    print()
    print("Experiment summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print()
    print("Artifacts saved to:")
    print("  results/   — plots, metrics, recommendations")
    print("  models/    — serialized trained model files (.joblib)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
