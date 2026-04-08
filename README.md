# VisionAid: Eyewear Recommendation from Facial Geometry

This repository contains a complete ENSF 444 final project implementation for the made-up client **VisionAid**. The project uses classical machine learning to classify **face shape from facial landmarks** and then converts the predicted face shape into **eyewear frame recommendations**.

## Why the dataset changed

The original proposal referenced `https://tianchi.aliyun.com/dataset/dataDetail?dataId=56` as a glasses recommendation dataset. When the project was verified on **April 7, 2026**, that URL resolved to a **Taobao advertising click-through-rate dataset**, so it could not support the proposed workflow.

To preserve the same client problem, this project uses the public **dsmlr/faceshape** dataset instead:

- Source repo: <https://github.com/dsmlr/faceshape>
- Dataset location in this project: `data/external/faceshape_source/published_dataset`
- Labels: `heart`, `oblong`, `oval`, `round`, `square`

This keeps the project aligned with the proposal's core idea:

1. Take a face image.
2. Extract facial geometry from landmarks.
3. Predict a face/fit category.
4. Translate that prediction into recommended eyewear classes.

## Project structure

- `run_project.py`: entry point for the full experiment
- `src/visionaid/pipeline.py`: feature extraction, EDA, model training, evaluation, artifact generation
- `src/visionaid/recommendation.py`: face-shape-to-eyewear recommendation rules
- `data/models/lbfmodel.yaml`: OpenCV landmark model
- `data/processed/landmark_features.csv`: cached landmark feature table
- `results/`: generated plots, reports, metrics, and example recommendations
- `docs/`: written deliverables supporting the project submission

## Models compared

The project compares four supervised learning models:

- Logistic Regression
- K-Nearest Neighbours
- Random Forest Classifier
- MLPClassifier

This satisfies the handout requirement to compare at least three models, including at least two non-linear models.

## How to run

From the project root:

```powershell
.venv\Scripts\python.exe run_project.py --force-features
```

If you already built the feature cache once, you can skip re-extraction:

```powershell
.venv\Scripts\python.exe run_project.py
```

## What the code does

1. Loads the face-shape image dataset.
2. Detects the main face in each image using OpenCV.
3. Extracts 68 facial landmarks using the LBF facemark model.
4. Builds a feature table from normalized landmarks and engineered geometry ratios.
5. Generates EDA plots:
   - class distribution
   - PCA projection
   - geometry correlation heatmap
6. Tunes and evaluates multiple classifiers with `GridSearchCV`.
7. Saves:
   - model comparison metrics
   - confusion matrices
   - classification reports
   - random forest feature importance
   - example eyewear recommendations

## Notes for grading

- The dataset is included in the repository under `data/external/faceshape_source/published_dataset`.
- The landmark model file required for feature extraction is included under `data/models/lbfmodel.yaml`.
- The code is organized as a reproducible workflow rather than a one-off notebook.
- The `results/` directory is generated automatically by running the project.

