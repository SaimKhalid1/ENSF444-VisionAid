"""End-to-end classical ML pipeline for the VisionAid project."""

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


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = Path(".")
    dataset_dir: Path = Path("data/external/faceshape_source/published_dataset")
    landmark_model_path: Path = Path("data/models/lbfmodel.yaml")
    processed_features_path: Path = Path("data/processed/landmark_features.csv")
    results_dir: Path = Path("results")
    models_dir: Path = Path("models")


class FaceShapeExperiment:
    """Run the VisionAid face-shape-to-eyewear ML workflow."""

    def __init__(self, paths: ProjectPaths | None = None, random_state: int = 42) -> None:
        self.paths = paths or ProjectPaths()
        self.random_state = random_state
        self._face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._facemark = cv2.face.createFacemarkLBF()
        self._facemark.loadModel(str(self.paths.landmark_model_path))
        self.paths.results_dir.mkdir(parents=True, exist_ok=True)
        self.paths.models_dir.mkdir(parents=True, exist_ok=True)
        self.paths.processed_features_path.parent.mkdir(parents=True, exist_ok=True)

    def build_feature_table(self, force_rebuild: bool = False) -> pd.DataFrame:
        """Extract landmarks and engineered geometry features from the dataset."""
        if self.paths.processed_features_path.exists() and not force_rebuild:
            return pd.read_csv(self.paths.processed_features_path)

        records: list[dict[str, Any]] = []
        for label_dir in sorted(self.paths.dataset_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            for image_path in sorted(label_dir.iterdir()):
                row = self._extract_record(image_path=image_path, label=label_dir.name)
                if row is not None:
                    records.append(row)

        df = pd.DataFrame(records)
        df.to_csv(self.paths.processed_features_path, index=False)
        return df

    def _extract_record(self, image_path: Path, label: str) -> dict[str, Any] | None:
        image = cv2.imread(str(image_path))
        if image is None:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        success, landmarks = self._facemark.fit(image, np.array([[x, y, w, h]], dtype=np.int32))
        if not success or not landmarks:
            return None

        points = landmarks[0][0].astype(np.float32)
        normalized = points.copy()
        normalized[:, 0] = (normalized[:, 0] - x) / max(w, 1)
        normalized[:, 1] = (normalized[:, 1] - y) / max(h, 1)

        row: dict[str, Any] = {
            "label": label,
            "image_path": str(image_path).replace("\\", "/"),
            "face_x": int(x),
            "face_y": int(y),
            "face_w": int(w),
            "face_h": int(h),
            "image_w": int(image.shape[1]),
            "image_h": int(image.shape[0]),
        }

        for idx, point in enumerate(normalized):
            row[f"lm_x_{idx:02d}"] = float(point[0])
            row[f"lm_y_{idx:02d}"] = float(point[1])

        row.update(self._geometry_features(points, (x, y, w, h), image.shape))
        return row

    @staticmethod
    def _distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
        return float(np.linalg.norm(point_a - point_b))

    def _geometry_features(
        self,
        points: np.ndarray,
        face_box: tuple[int, int, int, int],
        image_shape: tuple[int, int, int],
    ) -> dict[str, float]:
        x, y, w, h = face_box

        jaw_width = self._distance(points[0], points[16])
        cheekbone_width = self._distance(points[3], points[13])
        forehead_width = self._distance(points[17], points[26])
        chin_width = self._distance(points[6], points[10])
        face_height = self._distance(points[8], points[27])
        nose_width = self._distance(points[31], points[35])
        mouth_width = self._distance(points[48], points[54])
        eye_distance = self._distance(points[39], points[42])
        jaw_curve_left = self._distance(points[4], points[8])
        jaw_curve_right = self._distance(points[12], points[8])

        numerator = np.dot(points[6] - points[8], points[10] - points[8])
        denominator = np.linalg.norm(points[6] - points[8]) * np.linalg.norm(points[10] - points[8]) + 1e-8
        jaw_angle = float(np.arccos(np.clip(numerator / denominator, -1.0, 1.0)))

        image_h, image_w = image_shape[:2]
        return {
            "geom_face_aspect_ratio": float(w / max(h, 1)),
            "geom_face_area_ratio": float((w * h) / max(image_w * image_h, 1)),
            "geom_jaw_width_to_height": float(jaw_width / (face_height + 1e-8)),
            "geom_cheekbone_to_height": float(cheekbone_width / (face_height + 1e-8)),
            "geom_forehead_to_height": float(forehead_width / (face_height + 1e-8)),
            "geom_chin_to_jaw": float(chin_width / (jaw_width + 1e-8)),
            "geom_cheekbone_to_jaw": float(cheekbone_width / (jaw_width + 1e-8)),
            "geom_forehead_to_jaw": float(forehead_width / (jaw_width + 1e-8)),
            "geom_mouth_to_jaw": float(mouth_width / (jaw_width + 1e-8)),
            "geom_nose_to_jaw": float(nose_width / (jaw_width + 1e-8)),
            "geom_eye_distance_to_jaw": float(eye_distance / (jaw_width + 1e-8)),
            "geom_jaw_curve_left_to_height": float(jaw_curve_left / (face_height + 1e-8)),
            "geom_jaw_curve_right_to_height": float(jaw_curve_right / (face_height + 1e-8)),
            "geom_jaw_angle": jaw_angle,
            "geom_face_top_y": float(y / max(image_h, 1)),
            "geom_face_left_x": float(x / max(image_w, 1)),
        }

    def make_eda_plots(self, df: pd.DataFrame) -> None:
        """Generate EDA artifacts used in the final report."""
        sns.set_theme(style="whitegrid")

        plt.figure(figsize=(8, 5))
        order = sorted(df["label"].unique())
        sns.countplot(data=df, x="label", order=order, hue="label", palette="viridis", legend=False)
        plt.title("Class Distribution")
        plt.xlabel("Face Shape")
        plt.ylabel("Image Count")
        plt.tight_layout()
        plt.savefig(self.paths.results_dir / "class_distribution.png", dpi=200)
        plt.close()

        feature_columns = self._feature_columns(df)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[feature_columns])
        pca = PCA(n_components=2, random_state=self.random_state)
        transformed = pca.fit_transform(scaled_features)
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

        geometry_columns = [column for column in feature_columns if column.startswith("geom_")]
        corr = df[geometry_columns].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
        plt.title("Correlation Heatmap of Engineered Geometry Features")
        plt.tight_layout()
        plt.savefig(self.paths.results_dir / "geometry_correlation_heatmap.png", dpi=200)
        plt.close()

    @staticmethod
    def _feature_columns(df: pd.DataFrame) -> list[str]:
        return [
            column
            for column in df.columns
            if column.startswith("lm_") or column.startswith("geom_")
        ]

    def train_and_evaluate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Fit multiple models, evaluate them, and save artifacts."""
        X = df[self._feature_columns(df)]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y,
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        search_spaces = self._build_search_spaces()

        metrics_rows: list[dict[str, Any]] = []
        best_model_name = ""
        best_model_score = -1.0
        best_predictions: pd.DataFrame | None = None

        for model_name, config in search_spaces.items():
            search = GridSearchCV(
                estimator=config["pipeline"],
                param_grid=config["params"],
                cv=cv,
                scoring="accuracy",
                n_jobs=1,
                refit=True,
            )
            search.fit(X_train, y_train)

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

            joblib.dump(estimator, self.paths.models_dir / f"{model_name}_best.joblib")
            self._save_confusion_matrix(y_test, predictions, model_name)
            self._save_classification_report(y_test, predictions, model_name)

            if model_name == "random_forest":
                self._save_feature_importance_plot(
                    estimator.named_steps["model"].feature_importances_,
                    X.columns.tolist(),
                    model_name,
                )

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

        metrics_df = pd.DataFrame(metrics_rows).sort_values(
            by=["test_accuracy", "test_f1_weighted"], ascending=False
        )
        metrics_df.to_csv(self.paths.results_dir / "model_comparison.csv", index=False)

        summary = {
            "dataset_size_after_feature_extraction": int(len(df)),
            "dropped_images": int(500 - len(df)),
            "best_model": best_model_name,
            "best_model_test_accuracy": float(best_model_score),
        }
        with open(self.paths.results_dir / "experiment_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        if best_predictions is not None:
            best_predictions = best_predictions.reset_index(drop=True)
            recommendations_df = self._build_recommendation_examples(best_predictions)
            recommendations_df.to_csv(
                self.paths.results_dir / "recommendation_examples.csv", index=False
            )

        return metrics_df, summary

    def _build_search_spaces(self) -> dict[str, dict[str, Any]]:
        return {
            "logistic_regression": {
                "pipeline": Pipeline(
                    [
                        ("imputer", SimpleImputer()),
                        ("scaler", StandardScaler()),
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

    def _save_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> None:
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
        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(
            self.paths.results_dir / f"{model_name}_classification_report.csv"
        )

    def _save_feature_importance_plot(
        self, importances: np.ndarray, feature_names: list[str], model_name: str
    ) -> None:
        importance_df = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(15)
        )
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
        rows: list[dict[str, Any]] = []
        for _, row in prediction_df.head(15).iterrows():
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

    def run_all(self, force_rebuild_features: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
        df = self.build_feature_table(force_rebuild=force_rebuild_features)
        self.make_eda_plots(df)
        return self.train_and_evaluate(df)
