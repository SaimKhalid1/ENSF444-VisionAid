"""End-to-end classical ML pipeline for the VisionAid face-shape project.

Overview
--------
This module implements the complete supervised-learning workflow described in
the VisionAid project proposal:

1. **Dataset loading** — reads labeled face images from
   ``data/external/faceshape_source/published_dataset/``.
2. **Face detection** — uses OpenCV Haar cascade to locate the primary face
   in each image.
3. **Landmark extraction** — fits the LBF (Local Binary Feature) facemark model
   to obtain 68 facial landmark points.
4. **Feature engineering** — normalizes landmark coordinates relative to the
   face bounding box and computes 16 geometry ratio features.
5. **Exploratory data analysis** — generates class-distribution, PCA, and
   correlation-heatmap plots in ``results/``.
6. **Model training and tuning** — fits Logistic Regression, KNN, Random Forest,
   and MLPClassifier using ``GridSearchCV`` with stratified 5-fold CV.
7. **Evaluation** — reports test-set accuracy, precision, recall, F1, and saves
   confusion matrices and classification reports.
8. **Recommendation** — maps the best model's predictions to eyewear frame styles.

Quick start
-----------
Run the full pipeline from the command line (project root)::

    python run_project.py --force-features

Or import the class directly in a notebook::

    from src.visionaid.pipeline import FaceShapeExperiment
    exp = FaceShapeExperiment()
    metrics_df, summary = exp.run_all(force_rebuild_features=True)
    print(metrics_df)

Dataset requirement
-------------------
Feature extraction (``build_feature_table``) requires the faceshape images to be
present at ``data/external/faceshape_source/published_dataset/``.  Download them
with::

    git clone https://github.com/dsmlr/faceshape data/external/faceshape_source

If you only want to retrain the models without re-extracting landmarks, the
cached CSV at ``data/processed/landmark_features.csv`` is sufficient — just call
``run_project.py`` without ``--force-features``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .recommendation import get_recommendation


# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem paths used throughout the pipeline.

    All paths are relative to the working directory, which should be the
    project root.  Instantiate without arguments to use the defaults::

        paths = ProjectPaths()
        print(paths.processed_features_path)   # data/processed/landmark_features.csv

    Attributes
    ----------
    root:
        Project root directory.
    dataset_dir:
        Directory containing the raw face-image dataset organised into
        one sub-folder per class (heart/, oblong/, oval/, round/, square/).
    landmark_model_path:
        Path to the LBF facemark model weights file (lbfmodel.yaml).
        This file ships with the repository under data/models/.
    processed_features_path:
        Path for the cached landmark + geometry feature CSV.
    results_dir:
        Output directory for plots, metrics CSVs, and JSON summaries.
    models_dir:
        Output directory for serialized trained-model files (.joblib).
    """

    root: Path = Path(".")
    dataset_dir: Path = Path("data/external/faceshape_source/published_dataset")
    landmark_model_path: Path = Path("data/models/lbfmodel.yaml")
    processed_features_path: Path = Path("data/processed/landmark_features.csv")
    results_dir: Path = Path("results")
    models_dir: Path = Path("models")


# ---------------------------------------------------------------------------
# Main experiment class
# ---------------------------------------------------------------------------


class FaceShapeExperiment:
    """Run the VisionAid face-shape classification and eyewear recommendation workflow.

    This class wraps the entire experiment lifecycle.  All expensive steps write
    their results to disk so they do not need to be repeated on subsequent runs.

    Parameters
    ----------
    paths:
        A :class:`ProjectPaths` instance.  Pass ``None`` (default) to use the
        standard project directory layout.
    random_state:
        Integer seed used for all randomized steps (train/test split, CV
        shuffling, RandomForest, MLP).  Defaults to 42 for reproducibility.

    Examples
    --------
    Run the full pipeline and print the model comparison table::

        from src.visionaid.pipeline import FaceShapeExperiment
        exp = FaceShapeExperiment(random_state=42)
        metrics_df, summary = exp.run_all(force_rebuild_features=False)
        print(metrics_df.to_string(index=False))
    """

    def __init__(self, paths: ProjectPaths | None = None, random_state: int = 42) -> None:
        self.paths = paths or ProjectPaths()
        self.random_state = random_state

        # Load the OpenCV Haar cascade for frontal face detection.
        # haarcascade_frontalface_default.xml ships with every OpenCV installation;
        # cv2.data.haarcascades provides the correct platform-independent path.
        self._face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Load the LBF (Local Binary Feature) facemark model.
        # This model detects 68 landmark points on a face.  The weights file
        # (lbfmodel.yaml) must be present at paths.landmark_model_path.
        self._facemark = cv2.face.createFacemarkLBF()
        self._facemark.loadModel(str(self.paths.landmark_model_path))

        # Ensure output directories exist before any writing happens.
        self.paths.results_dir.mkdir(parents=True, exist_ok=True)
        self.paths.models_dir.mkdir(parents=True, exist_ok=True)
        self.paths.processed_features_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1: Feature extraction
    # ------------------------------------------------------------------

    def build_feature_table(self, force_rebuild: bool = False) -> pd.DataFrame:
        """Extract landmarks and geometry features from the image dataset.

        If the processed CSV already exists and ``force_rebuild`` is False,
        the cached CSV is loaded and returned immediately (fast path).

        Otherwise, every image in ``paths.dataset_dir`` is processed:

        1. The dominant face is detected with a Haar cascade.
        2. 68 facial landmarks are fitted with the LBF model.
        3. Landmark coordinates are normalized to [0, 1] relative to the
           face bounding box.
        4. Geometry ratios are computed from the raw landmark positions.
        5. All features are collected into a DataFrame and saved to
           ``paths.processed_features_path``.

        Parameters
        ----------
        force_rebuild:
            If ``True``, re-extract even if the CSV already exists.
            Use this after downloading a fresh copy of the dataset.

        Returns
        -------
        pd.DataFrame
            Feature table with one row per successfully processed image.
            Columns: label, image_path, face metadata, 136 landmark
            coordinates (lm_x_00…lm_y_67), and 16 geometry features (geom_*).
        """
        # Fast path: return cached CSV to avoid re-running slow landmark extraction.
        if self.paths.processed_features_path.exists() and not force_rebuild:
            return pd.read_csv(self.paths.processed_features_path)

        # Slow path: iterate over every class directory and every image file.
        records: list[dict[str, Any]] = []
        for label_dir in sorted(self.paths.dataset_dir.iterdir()):
            # Each sub-directory name is the face-shape class label.
            if not label_dir.is_dir():
                continue
            for image_path in sorted(label_dir.iterdir()):
                # _extract_record returns None if no face or landmarks were found.
                row = self._extract_record(image_path=image_path, label=label_dir.name)
                if row is not None:
                    records.append(row)

        df = pd.DataFrame(records)
        # Persist the feature table so that subsequent runs skip extraction.
        df.to_csv(self.paths.processed_features_path, index=False)
        return df

    def _extract_record(self, image_path: Path, label: str) -> dict[str, Any] | None:
        """Extract one row of features from a single face image.

        Returns ``None`` if:
        - the file cannot be read as an image,
        - no face is detected by the Haar cascade, or
        - landmark fitting fails on the detected face.

        Parameters
        ----------
        image_path:
            Absolute or relative path to the JPEG/PNG image.
        label:
            Ground-truth face-shape class for this image (e.g. ``"heart"``).

        Returns
        -------
        dict or None
            A flat dictionary of feature values, or ``None`` if processing failed.
        """
        # Load the image in BGR format (OpenCV default).
        image = cv2.imread(str(image_path))
        if image is None:
            # File is not a valid image (corrupted, wrong format, etc.).
            return None

        # Convert to grayscale for face detection — Haar cascades operate on
        # single-channel intensity images.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image.
        # scaleFactor=1.1 means each detection window is scaled by 10% per step.
        # minNeighbors=5 reduces false positives: a face box must have at least
        #   5 overlapping detections to be accepted.
        # minSize=(60, 60) ignores tiny detections (noise, background objects).
        faces = self._face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            # No face found — skip this image.
            return None

        # Use the largest detected face (by bounding-box area) as the subject.
        # This handles images where a background or partial face was also detected.
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

        # Fit the LBF facemark model to obtain 68 landmark (x, y) coordinates.
        # The model expects the face rectangle as a NumPy array of shape (N, 4).
        success, landmarks = self._facemark.fit(image, np.array([[x, y, w, h]], dtype=np.int32))
        if not success or not landmarks:
            # Landmark fitting failed — skip this image.
            return None

        # landmarks[0][0] has shape (68, 2) — 68 points, each with (x, y) in pixel space.
        points = landmarks[0][0].astype(np.float32)

        # Normalize landmark coordinates to [0, 1] relative to the face bounding box.
        # This makes the features invariant to image size and face position.
        normalized = points.copy()
        normalized[:, 0] = (normalized[:, 0] - x) / max(w, 1)   # x: left edge = 0, right = 1
        normalized[:, 1] = (normalized[:, 1] - y) / max(h, 1)   # y: top edge = 0, bottom = 1

        # Build the row dictionary with metadata and normalized landmark coordinates.
        row: dict[str, Any] = {
            "label": label,
            "image_path": str(image_path).replace("\\", "/"),  # normalize path separators
            "face_x": int(x),
            "face_y": int(y),
            "face_w": int(w),
            "face_h": int(h),
            "image_w": int(image.shape[1]),
            "image_h": int(image.shape[0]),
        }

        # Flatten the 68 normalized points into 136 columns: lm_x_00…lm_y_67.
        for idx, point in enumerate(normalized):
            row[f"lm_x_{idx:02d}"] = float(point[0])
            row[f"lm_y_{idx:02d}"] = float(point[1])

        # Append the 16 engineered geometry features computed from raw (pixel) landmarks.
        row.update(self._geometry_features(points, (x, y, w, h), image.shape))
        return row

    @staticmethod
    def _distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
        """Return the Euclidean distance between two 2-D landmark points.

        Parameters
        ----------
        point_a, point_b:
            Arrays of shape (2,) representing (x, y) pixel coordinates.

        Returns
        -------
        float
            The straight-line distance in pixels.
        """
        return float(np.linalg.norm(point_a - point_b))

    def _geometry_features(
        self,
        points: np.ndarray,
        face_box: tuple[int, int, int, int],
        image_shape: tuple[int, int, int],
    ) -> dict[str, float]:
        """Compute 16 hand-engineered geometry features from raw landmark positions.

        All features are dimensionless ratios so they are scale-invariant.
        The 68-point landmark numbering follows the standard iBUG convention:

        * Points 0–16:  jaw contour (left to right)
        * Points 17–26: eyebrow arches
        * Points 27–35: nose bridge and base
        * Points 36–47: eye corners and lids
        * Points 48–67: outer and inner lip contour

        References
        ----------
        iBUG 68-point facial landmark annotation standard:
            Sagonas, C., Tzimiropoulos, G., Zafeiriou, S., & Pantic, M. (2013).
            300 faces in-the-wild challenge: The first facial landmark
            localisation challenge. *ICCV Workshops*. IEEE.
            https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/

        LBF facemark model used to fit these landmarks:
            Ren, S., Cao, X., Wei, Y., & Sun, J. (2014). Face alignment at
            3000 FPS via regressing local binary features. *CVPR*, pp. 1685–1692.
            https://doi.org/10.1109/CVPR.2014.218

        The 16 geometry ratio features are hand-designed to capture the facial
        proportions that opticians use to categorise face shapes (jaw-to-face
        width, cheekbone ratios, forehead width, chin-to-jaw ratio, jaw angle).
        Feature design is informed by the same optician guidelines cited in
        ``recommendation.py`` and by the methodology in:
            Pasupa, K., Sunhem, W., & Loo, C. K. (2018). A hybrid approach to
            building face shape classifier for hairstyle recommender system.
            *Expert Systems with Applications*, 117, 1–16.
            https://doi.org/10.1016/j.eswa.2018.11.011

        Parameters
        ----------
        points:
            Raw landmark coordinates in pixel space, shape (68, 2).
        face_box:
            Tuple (x, y, w, h) — bounding box of the detected face in pixels.
        image_shape:
            Tuple (height, width, channels) — full image dimensions.

        Returns
        -------
        dict[str, float]
            A dictionary of 16 geometry feature names → float values.
        """
        x, y, w, h = face_box

        # Key facial distances (all in pixels, before normalization).
        # Points are indexed according to the 68-point iBUG annotation standard.

        # Horizontal width of the jaw from ear to ear (points 0 and 16).
        jaw_width = self._distance(points[0], points[16])

        # Width at the cheekbones (points 3 and 13), slightly inside the jaw.
        cheekbone_width = self._distance(points[3], points[13])

        # Approximate forehead width measured across the brow ridge (points 17 and 26).
        forehead_width = self._distance(points[17], points[26])

        # Narrow width near the chin tip (points 6 and 10).
        chin_width = self._distance(points[6], points[10])

        # Vertical face height from chin tip (point 8) to nose bridge (point 27).
        face_height = self._distance(points[8], points[27])

        # Nose width across the nostrils (points 31 and 35).
        nose_width = self._distance(points[31], points[35])

        # Mouth width from left to right lip corner (points 48 and 54).
        mouth_width = self._distance(points[48], points[54])

        # Horizontal distance between inner eye corners (points 39 and 42).
        # This is the inter-ocular distance.
        eye_distance = self._distance(points[39], points[42])

        # Left and right jaw curvature: distance from chin midpoint (8) to each
        # lower-jaw intermediate point (4 = left, 12 = right).
        jaw_curve_left = self._distance(points[4], points[8])
        jaw_curve_right = self._distance(points[12], points[8])

        # Jaw angle at the chin: the angle formed by the vectors from chin (8) to
        # lower-left jaw (6) and from chin (8) to lower-right jaw (10).
        # A more acute angle indicates a pointier chin (heart/oval); a wider angle
        # indicates a squarer jaw.
        numerator = np.dot(points[6] - points[8], points[10] - points[8])
        denominator = (
            np.linalg.norm(points[6] - points[8])
            * np.linalg.norm(points[10] - points[8])
            + 1e-8  # avoid division by zero
        )
        jaw_angle = float(np.arccos(np.clip(numerator / denominator, -1.0, 1.0)))

        image_h, image_w = image_shape[:2]

        return {
            # Face bounding-box aspect ratio (width / height).  Values > 1 indicate
            # a wider face; values < 1 indicate a taller face.
            "geom_face_aspect_ratio": float(w / max(h, 1)),

            # Fraction of the image area occupied by the face bounding box.
            # Encodes how close the subject is to the camera.
            "geom_face_area_ratio": float((w * h) / max(image_w * image_h, 1)),

            # Ratio of jaw width to face height.  High → wide/round; low → narrow/oblong.
            "geom_jaw_width_to_height": float(jaw_width / (face_height + 1e-8)),

            # Ratio of cheekbone width to face height.
            "geom_cheekbone_to_height": float(cheekbone_width / (face_height + 1e-8)),

            # Ratio of forehead width to face height.  High → broad forehead (heart).
            "geom_forehead_to_height": float(forehead_width / (face_height + 1e-8)),

            # Chin width relative to jaw width.  Low → tapered chin (heart/oval).
            "geom_chin_to_jaw": float(chin_width / (jaw_width + 1e-8)),

            # Cheekbone width relative to jaw width.  High → widest at cheeks.
            "geom_cheekbone_to_jaw": float(cheekbone_width / (jaw_width + 1e-8)),

            # Forehead width relative to jaw width.  High → heart/inverted triangle.
            "geom_forehead_to_jaw": float(forehead_width / (jaw_width + 1e-8)),

            # Mouth width relative to jaw width.
            "geom_mouth_to_jaw": float(mouth_width / (jaw_width + 1e-8)),

            # Nose width relative to jaw width.
            "geom_nose_to_jaw": float(nose_width / (jaw_width + 1e-8)),

            # Inter-ocular distance relative to jaw width.
            "geom_eye_distance_to_jaw": float(eye_distance / (jaw_width + 1e-8)),

            # Left jaw curvature relative to face height.  Encodes jaw roundness.
            "geom_jaw_curve_left_to_height": float(jaw_curve_left / (face_height + 1e-8)),

            # Right jaw curvature relative to face height.
            "geom_jaw_curve_right_to_height": float(jaw_curve_right / (face_height + 1e-8)),

            # Jaw angle at the chin in radians.  Wider → squarer chin.
            "geom_jaw_angle": jaw_angle,

            # Vertical position of the face top edge as a fraction of image height.
            # (Encodes framing/padding, not face shape — lower predictive value.)
            "geom_face_top_y": float(y / max(image_h, 1)),

            # Horizontal position of the face left edge as a fraction of image width.
            "geom_face_left_x": float(x / max(image_w, 1)),
        }

    # ------------------------------------------------------------------
    # Stage 2: Exploratory data analysis
    # ------------------------------------------------------------------

    def make_eda_plots(self, df: pd.DataFrame) -> None:
        """Generate EDA artifacts used in the final report.

        Produces three plots and saves them to ``paths.results_dir``:

        1. ``class_distribution.png`` — count of images per face-shape class.
        2. ``pca_projection.png`` — 2-D PCA scatter of all landmark + geometry
           features, coloured by class.
        3. ``geometry_correlation_heatmap.png`` — Pearson correlation matrix
           of the 16 engineered geometry features.

        Parameters
        ----------
        df:
            Feature table returned by :meth:`build_feature_table`.
        """
        # Use a clean whitegrid style for all plots in this session.
        sns.set_theme(style="whitegrid")

        # ------------------------------------------------------------------
        # Plot 1: Class distribution bar chart
        # ------------------------------------------------------------------
        plt.figure(figsize=(8, 5))
        order = sorted(df["label"].unique())  # alphabetical order for consistency
        sns.countplot(data=df, x="label", order=order, hue="label", palette="viridis", legend=False)
        plt.title("Class Distribution")
        plt.xlabel("Face Shape")
        plt.ylabel("Image Count")
        plt.tight_layout()
        plt.savefig(self.paths.results_dir / "class_distribution.png", dpi=200)
        plt.close()

        # ------------------------------------------------------------------
        # Plot 2: PCA projection
        # Reduce all features to 2 principal components and scatter-plot them.
        # If the classes form visible clusters, the features carry predictive signal.
        # ------------------------------------------------------------------
        feature_columns = self._feature_columns(df)

        # StandardScaler is applied before PCA so that high-variance landmark
        # coordinate columns do not dominate the decomposition.
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[feature_columns])

        pca = PCA(n_components=2, random_state=self.random_state)
        transformed = pca.fit_transform(scaled_features)

        # Build a small DataFrame for easier seaborn plotting.
        pca_df = pd.DataFrame(
            {
                "PC1": transformed[:, 0],
                "PC2": transformed[:, 1],
                "label": df["label"],
            }
        )

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="label", palette="tab10", s=55)
        plt.title(
            "PCA Projection of Landmark Features\n"
            f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}"
        )
        plt.tight_layout()
        plt.savefig(self.paths.results_dir / "pca_projection.png", dpi=200)
        plt.close()

        # ------------------------------------------------------------------
        # Plot 3: Geometry feature correlation heatmap
        # Shows how strongly the 16 engineered features co-vary.
        # Highly correlated pairs (e.g. jaw_width / cheekbone_width ratios)
        # are somewhat redundant; a future version might remove one from each pair.
        # ------------------------------------------------------------------
        geometry_columns = [col for col in feature_columns if col.startswith("geom_")]
        corr = df[geometry_columns].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
        plt.title("Correlation Heatmap of Engineered Geometry Features")
        plt.tight_layout()
        plt.savefig(self.paths.results_dir / "geometry_correlation_heatmap.png", dpi=200)
        plt.close()

    @staticmethod
    def _feature_columns(df: pd.DataFrame) -> list[str]:
        """Return all landmark (lm_*) and geometry (geom_*) feature column names.

        Excludes metadata columns such as ``label``, ``image_path``, and
        bounding-box dimensions, which are not used as model inputs.
        """
        return [
            col
            for col in df.columns
            if col.startswith("lm_") or col.startswith("geom_")
        ]

    # ------------------------------------------------------------------
    # Stage 3: Model training and evaluation
    # ------------------------------------------------------------------

    def train_and_evaluate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Fit multiple classifiers, evaluate on a held-out test set, and save artifacts.

        Workflow:
        1. Split features into 80 % train / 20 % test (stratified by class).
        2. For each model, run ``GridSearchCV`` with 5-fold stratified CV to find
           the best hyperparameters.
        3. Evaluate the best estimator on the test set.
        4. Save the trained model to ``models/<name>_best.joblib``.
        5. Save confusion matrix PNG and classification report CSV to ``results/``.
        6. After all models are evaluated, save the comparison table and a JSON
           summary.
        7. Generate eyewear recommendation examples from the best model's predictions.

        Parameters
        ----------
        df:
            Feature table returned by :meth:`build_feature_table`.

        Returns
        -------
        metrics_df:
            DataFrame with one row per model, sorted by descending test accuracy.
            Columns: model, best_cv_accuracy, test_accuracy, test_precision_weighted,
            test_recall_weighted, test_f1_weighted, best_params.
        summary:
            Dictionary with high-level experiment metadata (best model name,
            best accuracy, dataset size).
        """
        # Separate feature matrix (X) and label vector (y).
        X = df[self._feature_columns(df)]
        y = df["label"]

        # Stratified split ensures each class is proportionally represented in
        # both train and test sets, which matters most for smaller datasets.
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y,
        )

        # 5-fold stratified cross-validation used inside GridSearchCV.
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        # Retrieve the hyperparameter search spaces for all four models.
        search_spaces = self._build_search_spaces()

        metrics_rows: list[dict[str, Any]] = []
        best_model_name = ""
        best_model_score = -1.0
        best_predictions: pd.DataFrame | None = None

        for model_name, config in search_spaces.items():
            # GridSearchCV exhaustively searches all combinations in config["params"].
            # refit=True re-trains the best parameter combination on the full training set.
            search = GridSearchCV(
                estimator=config["pipeline"],
                param_grid=config["params"],
                cv=cv,
                scoring="accuracy",
                n_jobs=1,   # single-threaded to keep output deterministic
                refit=True,
            )
            search.fit(X_train, y_train)

            # Evaluate the best estimator found by GridSearchCV on the held-out test set.
            estimator = search.best_estimator_
            predictions = estimator.predict(X_test)

            metrics_row = {
                "model": model_name,
                "best_cv_accuracy": float(search.best_score_),
                "test_accuracy": float(accuracy_score(y_test, predictions)),
                "test_precision_weighted": float(
                    precision_score(y_test, predictions, average="weighted", zero_division=0)
                ),
                "test_recall_weighted": float(
                    recall_score(y_test, predictions, average="weighted", zero_division=0)
                ),
                "test_f1_weighted": float(
                    f1_score(y_test, predictions, average="weighted", zero_division=0)
                ),
                "best_params": json.dumps(search.best_params_, sort_keys=True),
            }
            metrics_rows.append(metrics_row)

            # Serialize the trained pipeline (imputer + optional scaler + model).
            joblib.dump(estimator, self.paths.models_dir / f"{model_name}_best.joblib")

            # Save per-model diagnostic artifacts.
            self._save_confusion_matrix(y_test, predictions, model_name)
            self._save_classification_report(y_test, predictions, model_name)

            # For Random Forest only: also save feature importance.
            if model_name == "random_forest":
                self._save_feature_importance_plot(
                    estimator.named_steps["model"].feature_importances_,
                    X.columns.tolist(),
                    model_name,
                )

            # Track which model performed best so we can generate recommendations
            # based on the highest-accuracy model's test-set predictions.
            if metrics_row["test_accuracy"] > best_model_score:
                best_model_score = metrics_row["test_accuracy"]
                best_model_name = model_name
                best_predictions = pd.DataFrame(
                    {
                        "image_path": df.loc[X_test.index, "image_path"].to_numpy(),
                        "actual_face_shape": y_test.to_numpy(),
                        "predicted_face_shape": predictions,
                    }
                )

        # Sort models by descending test accuracy for easy ranking.
        metrics_df = pd.DataFrame(metrics_rows).sort_values(
            by=["test_accuracy", "test_f1_weighted"], ascending=False
        )
        metrics_df.to_csv(self.paths.results_dir / "model_comparison.csv", index=False)

        # Compact JSON summary for the report and notebook.
        summary = {
            "dataset_size_after_feature_extraction": int(len(df)),
            "dropped_images": int(500 - len(df)),
            "best_model": best_model_name,
            "best_model_test_accuracy": float(best_model_score),
        }
        with open(self.paths.results_dir / "experiment_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        # Save example recommendations produced by the best model.
        if best_predictions is not None:
            best_predictions = best_predictions.reset_index(drop=True)
            recommendations_df = self._build_recommendation_examples(best_predictions)
            recommendations_df.to_csv(
                self.paths.results_dir / "recommendation_examples.csv", index=False
            )

        return metrics_df, summary

    def _build_search_spaces(self) -> dict[str, dict[str, Any]]:
        """Define sklearn Pipelines and GridSearchCV hyperparameter grids.

        Each entry in the returned dictionary contains:
        - ``"pipeline"``: a ``sklearn.pipeline.Pipeline`` with an imputer,
          optional scaler, and the model.
        - ``"params"``: a dict mapping ``"model__<param>"`` keys to value lists.

        The ``SimpleImputer`` is included in every pipeline as a safety net for
        any NaN values that might appear if a landmark coordinate could not be
        computed.  In practice, the feature CSV produced by this pipeline does
        not contain NaNs, but the imputer makes the pipeline robust to future
        changes.

        Returns
        -------
        dict[str, dict]
            Keys: ``"logistic_regression"``, ``"knn"``, ``"random_forest"``,
            ``"mlp_classifier"``.
        """
        return {
            # ---------------------------------------------------------------
            # Logistic Regression: a linear baseline.
            # C is the inverse regularization strength (larger C = less regularization).
            # class_weight="balanced" up-weights minority classes.
            # ---------------------------------------------------------------
            "logistic_regression": {
                "pipeline": Pipeline(
                    [
                        ("imputer", SimpleImputer()),
                        ("scaler", StandardScaler()),   # required for LR convergence
                        (
                            "model",
                            LogisticRegression(max_iter=5000, random_state=self.random_state),
                        ),
                    ]
                ),
                "params": {
                    "model__C": [0.3, 1.0, 3.0, 10.0],
                    "model__class_weight": [None, "balanced"],
                },
            },
            # ---------------------------------------------------------------
            # K-Nearest Neighbours: non-parametric distance-based classifier.
            # p=1 → Manhattan distance; p=2 → Euclidean distance.
            # StandardScaler is essential so that distances are computed on
            # comparable scales.
            # ---------------------------------------------------------------
            "knn": {
                "pipeline": Pipeline(
                    [
                        ("imputer", SimpleImputer()),
                        ("scaler", StandardScaler()),
                        ("model", KNeighborsClassifier()),
                    ]
                ),
                "params": {
                    "model__n_neighbors": [3, 5, 7, 9, 11],
                    "model__weights": ["uniform", "distance"],
                    "model__p": [1, 2],
                },
            },
            # ---------------------------------------------------------------
            # Random Forest: ensemble of decision trees.
            # No scaler needed — trees are invariant to feature scaling.
            # max_depth=None lets trees grow until all leaves are pure.
            # min_samples_leaf controls overfitting.
            # ---------------------------------------------------------------
            "random_forest": {
                "pipeline": Pipeline(
                    [
                        ("imputer", SimpleImputer()),
                        (
                            "model",
                            RandomForestClassifier(
                                random_state=self.random_state,
                            ),
                        ),
                    ]
                ),
                "params": {
                    "model__n_estimators": [200, 400],
                    "model__max_depth": [None, 12, 20],
                    "model__min_samples_leaf": [1, 2, 4],
                },
            },
            # ---------------------------------------------------------------
            # MLP Classifier: a shallow feed-forward neural network.
            # StandardScaler is applied so gradients behave predictably.
            # alpha is the L2 regularization strength.
            # ---------------------------------------------------------------
            "mlp_classifier": {
                "pipeline": Pipeline(
                    [
                        ("imputer", SimpleImputer()),
                        ("scaler", StandardScaler()),
                        ("model", MLPClassifier(max_iter=2000, random_state=self.random_state)),
                    ]
                ),
                "params": {
                    "model__hidden_layer_sizes": [(64,), (128, 64)],
                    "model__alpha": [0.0001, 0.001],
                },
            },
        }

    # ------------------------------------------------------------------
    # Artifact helpers
    # ------------------------------------------------------------------

    def _save_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> None:
        """Save a colour-coded confusion matrix PNG to results/.

        Parameters
        ----------
        y_true:
            True class labels from the test set.
        y_pred:
            Predicted class labels from the fitted model.
        model_name:
            Used to name the output file (``<model_name>_confusion_matrix.png``).
        """
        # Sort labels alphabetically so axes are consistent across all models.
        labels = sorted(y_true.unique())
        matrix = confusion_matrix(y_true, y_pred, labels=labels)

        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title(f"Confusion Matrix: {model_name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(self.paths.results_dir / f"{model_name}_confusion_matrix.png", dpi=200)
        plt.close()

    def _save_classification_report(self, y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> None:
        """Save a per-class precision / recall / F1 report CSV to results/.

        Parameters
        ----------
        y_true:
            True class labels.
        y_pred:
            Predicted class labels.
        model_name:
            Used to name the output file (``<model_name>_classification_report.csv``).
        """
        # output_dict=True returns a nested dict; converting via DataFrame gives
        # a clean CSV with rows for each class plus macro/weighted averages.
        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(
            self.paths.results_dir / f"{model_name}_classification_report.csv"
        )

    def _save_feature_importance_plot(
        self, importances: np.ndarray, feature_names: list[str], model_name: str
    ) -> None:
        """Save a horizontal bar chart of the top-15 Random Forest feature importances.

        Feature importances from a Random Forest are computed as the mean
        decrease in impurity (MDI) across all trees.  Higher values indicate
        features that are more useful for splitting nodes.

        Parameters
        ----------
        importances:
            Array of importance scores, one per feature, from
            ``RandomForestClassifier.feature_importances_``.
        feature_names:
            Corresponding feature names (same order as ``importances``).
        model_name:
            Used to name the output files.
        """
        importance_df = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(15)   # show only the 15 most important features for readability
        )

        # Also save the ranked list as a CSV for use in the report.
        importance_df.to_csv(
            self.paths.results_dir / f"{model_name}_top_feature_importance.csv", index=False
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=importance_df,
            x="importance",
            y="feature",
            hue="feature",
            palette="mako",
            legend=False,
        )
        plt.title("Top 15 Random Forest Feature Importances")
        plt.tight_layout()
        plt.savefig(self.paths.results_dir / f"{model_name}_feature_importance.png", dpi=200)
        plt.close()

    def _build_recommendation_examples(self, prediction_df: pd.DataFrame) -> pd.DataFrame:
        """Generate eyewear recommendation rows for the first 15 test-set predictions.

        For each predicted face shape, the recommendation module is called to
        retrieve three suggested frame styles and a human-readable rationale.

        Parameters
        ----------
        prediction_df:
            DataFrame with columns ``image_path``, ``actual_face_shape``,
            ``predicted_face_shape``.

        Returns
        -------
        pd.DataFrame
            One row per example with columns: image_path, actual_face_shape,
            predicted_face_shape, recommended_frame_1/2/3, recommendation_rationale.
        """
        rows: list[dict[str, Any]] = []
        for _, row in prediction_df.head(15).iterrows():
            # get_recommendation looks up the frame styles for the predicted shape.
            recommendation = get_recommendation(str(row["predicted_face_shape"]))
            rows.append(
                {
                    "image_path": row["image_path"],
                    "actual_face_shape": row["actual_face_shape"],
                    "predicted_face_shape": row["predicted_face_shape"],
                    "recommended_frame_1": recommendation.recommended_frames[0],
                    "recommended_frame_2": recommendation.recommended_frames[1],
                    "recommended_frame_3": recommendation.recommended_frames[2],
                    "recommendation_rationale": recommendation.rationale,
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Top-level orchestrator
    # ------------------------------------------------------------------

    def run_all(self, force_rebuild_features: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Execute all pipeline stages in order.

        This is the main entry point called by ``run_project.py``.

        Parameters
        ----------
        force_rebuild_features:
            If ``True``, re-extract landmark features from the raw images even
            if the cached CSV exists.  Pass ``True`` after downloading a fresh
            copy of the dataset.

        Returns
        -------
        metrics_df:
            Per-model evaluation metrics (see :meth:`train_and_evaluate`).
        summary:
            High-level experiment summary dict.
        """
        # Stage 1: load (or build) the feature table.
        df = self.build_feature_table(force_rebuild=force_rebuild_features)

        # Stage 2: generate EDA plots.
        self.make_eda_plots(df)

        # Stage 3: train models, evaluate, save artifacts.
        return self.train_and_evaluate(df)
