# VisionAid: Eyewear Recommendation from Facial Geometry

**ENSF 444 — Final Project**

VisionAid is a fictional startup that wants to recommend eyeglass frames from a user's selfie.
This project builds and compares multiple machine learning classifiers that predict **face shape**
from facial landmark geometry, then converts the predicted shape to a shortlist of **eyewear frame styles**.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Setup and installation](#3-setup-and-installation)
4. [How to run](#4-how-to-run)
5. [Project structure](#5-project-structure)
6. [ML pipeline summary](#6-ml-pipeline-summary)
7. [Results](#7-results)
8. [Notebooks](#8-notebooks)
9. [Ethical considerations](#9-ethical-considerations)

---

## 1. Project overview

**Business problem:** Buying glasses online is difficult without trying frames in person.
VisionAid wants an automated pipeline that infers facial structure from a selfie and returns
a shortlist of frame styles likely to suit the user.

**Technical approach:**
1. Detect the face in each image using an OpenCV Haar cascade.
2. Extract **68 facial landmarks** with the OpenCV LBF facemark model.
3. Normalize coordinates and engineer **16 geometry ratio features** (jaw width, cheekbone ratios, forehead ratio, jaw angle, etc.).
4. Train and compare four scikit-learn classifiers to predict one of five face-shape categories.
5. Map the predicted face shape to recommended eyewear frame styles.

**Dataset deviation note:**
The original proposal cited a Tianchi glasses-recommendation dataset. When verified on
April 7, 2026, that link resolved to an unrelated Taobao advertising dataset. The project
was updated to use the public **dsmlr/faceshape** dataset, which supports the same overall
client workflow while remaining fully reproducible.

---

## 2. Dataset

**Source:** [github.com/dsmlr/faceshape](https://github.com/dsmlr/faceshape)

| Property | Value |
|---|---|
| Total images | 500 |
| Classes | `heart`, `oblong`, `oval`, `round`, `square` (100 each) |
| Successfully processed | 493 (7 dropped — no face detected or landmark fit failed) |
| Format | JPEG images, one sub-folder per class |

**Local path:** `data/external/faceshape_source/published_dataset/`

---

## 3. Setup and installation

### Prerequisites

- Python 3.9 or later
- Git

### Step 1 — Clone this repository

```bash
git clone <this-repo-url>
cd ENSF444-VisionAid
```

### Step 2 — Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

Key packages:
| Package | Purpose |
|---|---|
| `opencv-contrib-python-headless` | Face detection and 68-point landmark extraction |
| `scikit-learn` | ML classifiers, GridSearchCV, evaluation metrics |
| `pandas` / `numpy` | Feature table manipulation |
| `matplotlib` / `seaborn` | EDA plots and result visualizations |
| `joblib` | Model serialization |
| `jupyter` | Running the analysis notebooks (optional) |

### Step 4 — Download the faceshape dataset

```bash
git clone https://github.com/dsmlr/faceshape data/external/faceshape_source
```

After cloning, the directory structure should be:

```
data/external/faceshape_source/
    published_dataset/
        heart/     (100 images)
        oblong/    (100 images)
        oval/      (100 images)
        round/     (100 images)
        square/    (100 images)
```

The OpenCV landmark model is already included at `data/models/lbfmodel.yaml`.

---

## 4. How to run

### Full pipeline (extract features + train all models)

```bash
python run_project.py --force-features
```

Use `--force-features` whenever you have downloaded a fresh copy of the dataset
or want to re-run landmark extraction.

### Re-train models only (skip feature extraction)

If `data/processed/landmark_features.csv` already exists (e.g. from a previous run):

```bash
python run_project.py
```

### Expected runtime

| Stage | Approximate time |
|---|---|
| Feature extraction (500 images) | 5–20 minutes depending on CPU |
| GridSearchCV — all four models | 5–15 minutes |
| EDA plots and artifact saving | < 1 minute |

### Console output

At the end of the run you will see a model comparison table and a summary:

```
======================================
VisionAid pipeline complete
======================================

Model comparison (sorted by test accuracy):
 model  best_cv_accuracy  test_accuracy  ...
 logistic_regression    0.5991     0.6263  ...
 ...

Experiment summary:
  best_model: logistic_regression
  best_model_test_accuracy: 0.6263
```

---

## 5. Project structure

```
ENSF444-VisionAid/
│
├── run_project.py                    # CLI entry point — run this to execute the pipeline
│
├── src/visionaid/
│   ├── __init__.py                   # Package public API
│   ├── pipeline.py                   # End-to-end ML pipeline (FaceShapeExperiment class)
│   ├── recommendation.py             # Face-shape → eyewear frame recommendation rules
│   └── data_loader.py                # Utility functions for loading the feature CSV
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb    # EDA: distribution, PCA, correlations
│   └── 02_model_training_and_evaluation.ipynb # Training, tuning, comparison, recommendations
│
├── data/
│   ├── external/faceshape_source/    # Raw face images (download separately — see Setup)
│   ├── models/lbfmodel.yaml          # OpenCV LBF facemark model weights (included)
│   └── processed/landmark_features.csv  # Cached feature table (generated by pipeline)
│
├── models/
│   ├── logistic_regression_best.joblib
│   ├── knn_best.joblib
│   ├── random_forest_best.joblib
│   └── mlp_classifier_best.joblib
│
├── results/
│   ├── class_distribution.png
│   ├── pca_projection.png
│   ├── geometry_correlation_heatmap.png
│   ├── model_comparison.csv
│   ├── experiment_summary.json
│   ├── <model>_confusion_matrix.png          (one per model)
│   ├── <model>_classification_report.csv     (one per model)
│   ├── random_forest_feature_importance.png
│   ├── random_forest_top_feature_importance.csv
│   └── recommendation_examples.csv
│
├── docs/
│   ├── final_report.md               # Written project report
│   ├── proposal_deviation_note.md    # Explains the dataset change
│   ├── presentation_outline.md       # Slide-by-slide outline
│   ├── presentation_script.md        # Spoken script for the video
│   └── reflection_template.md        # Individual reflection template
│
├── requirements.txt                  # Python package dependencies
└── README.md                         # This file
```

---

## 6. ML pipeline summary

### Feature engineering

Each face image is processed as follows:

1. **Face detection** — OpenCV Haar cascade (`haarcascade_frontalface_default.xml`).
2. **Landmark extraction** — OpenCV LBF model produces 68 (x, y) points in pixel space.
3. **Normalization** — coordinates are scaled to [0, 1] relative to the face bounding box.
4. **Geometry features** — 16 ratios derived from landmark positions:

| Feature | What it measures |
|---|---|
| `geom_jaw_width_to_height` | Horizontal jaw span relative to face height |
| `geom_cheekbone_to_height` | Cheekbone width relative to face height |
| `geom_forehead_to_height` | Forehead width relative to face height |
| `geom_chin_to_jaw` | Chin narrowness — low = tapered (heart/oval) |
| `geom_cheekbone_to_jaw` | Widest point relative to jaw width |
| `geom_forehead_to_jaw` | Forehead relative to jaw — high = heart shape |
| `geom_jaw_angle` | Angle at the chin tip — wider = squarer jaw |
| `geom_eye_distance_to_jaw` | Inter-ocular distance relative to jaw width |
| ... (8 more) | Mouth, nose, and curvature ratios |

### Models compared

| Model | Type | Hyperparameters tuned |
|---|---|---|
| Logistic Regression | Linear | `C`, `class_weight` |
| K-Nearest Neighbours | Non-linear | `n_neighbors`, `weights`, `p` (distance metric) |
| Random Forest | Non-linear | `n_estimators`, `max_depth`, `min_samples_leaf` |
| MLP Classifier | Non-linear | `hidden_layer_sizes`, `alpha` |

All models use `GridSearchCV` with **5-fold stratified cross-validation**.

### Recommendation layer

Predicted face shape is mapped to eyewear frame styles:

| Predicted shape | Recommended frames |
|---|---|
| `heart` | round, aviator, cat-eye |
| `oblong` | square, aviator, cat-eye |
| `oval` | rectangular, square, aviator |
| `round` | rectangular, square, cat-eye |
| `square` | round, aviator, cat-eye |

---

## 7. Results

| Model | CV Accuracy | Test Accuracy | Weighted F1 |
|---|---:|---:|---:|
| **Logistic Regression** | 0.5991 | **0.6263** | **0.6237** |
| Random Forest | 0.5510 | 0.6061 | 0.6027 |
| MLP Classifier | 0.5509 | 0.5960 | 0.5947 |
| KNN | 0.4696 | 0.4747 | 0.4742 |

**Best model:** Logistic Regression with `C=1.0`, `class_weight="balanced"`.

**Per-class F1 (best model):**

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| heart | 0.59 | 0.65 | 0.62 |
| oblong | 0.65 | 0.75 | 0.70 |
| oval | 0.47 | 0.45 | 0.46 |
| round | 0.67 | 0.53 | 0.59 |
| square | 0.75 | 0.75 | 0.75 |

`square` and `oblong` are easiest to classify; `oval` is hardest due to feature overlap.

---

## 8. Notebooks

Interactive walkthroughs are available under `notebooks/`:

```bash
jupyter notebook notebooks/
```

| Notebook | Contents |
|---|---|
| `01_exploratory_data_analysis.ipynb` | Class distribution, PCA, geometry correlations, per-class landmark plots |
| `02_model_training_and_evaluation.ipynb` | Train/test split, GridSearchCV, model comparison, confusion matrices, feature importance, recommendations |

Both notebooks load data from `data/processed/landmark_features.csv` and do **not**
require the raw images or OpenCV to be configured.

---

## 9. Ethical considerations

- **Dataset diversity:** The dataset is small (500 images) and may not represent the full diversity of real users across age, gender presentation, ethnicity, lighting conditions, or image quality.  A production system would require significantly more data.
- **Aesthetic subjectivity:** Frame recommendations are based on general optician guidelines, but eyewear preference is personal and culturally influenced.  Recommendations should be presented as suggestions, not instructions.
- **Biometric privacy:** A selfie-based system processes facial images, which are sensitive biometric data.  Any real deployment would need to address data minimization, user consent, and secure storage.
- **Overconfidence risk:** With a test accuracy of ~62 %, the system will be wrong roughly one in three times.  Confidence estimates and user overrides should be part of any client-facing implementation.
