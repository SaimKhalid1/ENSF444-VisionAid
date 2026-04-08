"""VisionAid: facial-geometry-based eyewear frame recommendation.

This package implements the complete ENSF 444 final project pipeline:

1. Extract 68 facial landmarks from face images using OpenCV LBF.
2. Engineer geometry ratio features from the landmark positions.
3. Train and compare four scikit-learn classifiers to predict face shape.
4. Map the predicted face shape to a shortlist of eyewear frame styles.

Modules
-------
pipeline:
    ``FaceShapeExperiment`` — orchestrates the end-to-end ML workflow.
recommendation:
    ``get_recommendation`` — converts a predicted face-shape label to frame styles.
data_loader:
    Utility functions for loading and inspecting the landmark feature CSV.

Quick start
-----------
Run the full experiment from the command line (project root directory)::

    python run_project.py --force-features

Or import the experiment class directly::

    from src.visionaid import FaceShapeExperiment
    exp = FaceShapeExperiment()
    metrics_df, summary = exp.run_all()

Load the pre-computed feature table::

    from src.visionaid.data_loader import load_features, dataset_summary
    df = load_features()
    dataset_summary(df)
"""

from .pipeline import FaceShapeExperiment
from .recommendation import get_recommendation, list_supported_shapes, Recommendation
from .data_loader import (
    load_features,
    get_feature_columns,
    get_geometry_columns,
    get_landmark_columns,
    dataset_summary,
    LABEL_COLUMN,
    CLASSES,
)

__all__ = [
    # Pipeline
    "FaceShapeExperiment",
    # Recommendation
    "get_recommendation",
    "list_supported_shapes",
    "Recommendation",
    # Data loading
    "load_features",
    "get_feature_columns",
    "get_geometry_columns",
    "get_landmark_columns",
    "dataset_summary",
    "LABEL_COLUMN",
    "CLASSES",
]
