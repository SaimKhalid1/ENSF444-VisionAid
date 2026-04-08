# VisionAid Final Report

## 1. Project Overview

**Client:** VisionAid, a fictional startup building a mobile experience that recommends eyeglass frames from a user's selfie.

**Business problem:** Online eyewear shopping creates uncertainty about which frame style will look and fit best. VisionAid wants an automated pipeline that can infer facial structure from a face image and return a shortlist of suitable eyewear types.

**Project goal:** Build and compare multiple machine learning models that use facial geometry to predict a face/fit category, then translate that prediction into frame recommendations.

## 2. Proposal Deviation

The original proposal referenced the dataset link `https://tianchi.aliyun.com/dataset/dataDetail?dataId=56`. When verified on **April 7, 2026**, that link resolved to a **Taobao display advertising click-through-rate dataset**, not a glasses recommendation dataset.

Because the original source could not support the proposed client problem, the project was adjusted as follows:

- Replaced the broken proposal dataset with the public `dsmlr/faceshape` dataset.
- Preserved the same client workflow by using facial landmarks and geometry from face images.
- Predicted **face shape** first, then mapped the predicted shape to **eyewear frame recommendations** aligned with the proposal's intended output categories.

This keeps the project defensible, reproducible, and closely aligned with the proposal's original intent.

## 3. Dataset

**Dataset used:** `dsmlr/faceshape`

- Source: <https://github.com/dsmlr/faceshape>
- Local path: `data/external/faceshape_source/published_dataset`
- Total labelled images provided: 500
- Classes: `heart`, `oblong`, `oval`, `round`, `square`
- Images successfully processed into features: 493
- Images dropped during landmark extraction: 7

**Why this dataset fits the problem:**

- It provides labeled face-shape categories from real face images.
- It supports landmark extraction and geometry-based feature engineering.
- It can be used to approximate the recommendation stage of eyewear selection, since face shape is a common intermediate variable in frame recommendation systems.

## 4. Preprocessing and Feature Engineering

The workflow follows a standard machine learning pipeline:

1. Load each labeled image from the dataset.
2. Detect the dominant face using OpenCV Haar cascades.
3. Extract **68 facial landmarks** using OpenCV's LBF facemark model.
4. Normalize landmark coordinates relative to the detected face bounding box.
5. Engineer additional geometry features, including:
   - face aspect ratio
   - face area ratio
   - jaw width to face height
   - cheekbone width to face height
   - forehead width to face height
   - chin width to jaw width
   - cheekbone width to jaw width
   - jaw angle
   - left and right jaw curvature ratios
6. Store the processed feature table in `data/processed/landmark_features.csv`.

The resulting feature set contains both:

- normalized landmark coordinates
- interpretable geometric ratios

This supports both model performance and later interpretation.

## 5. Exploratory Data Analysis

Three EDA artifacts were generated automatically:

- `results/class_distribution.png`
- `results/pca_projection.png`
- `results/geometry_correlation_heatmap.png`

These plots were used to inspect:

- whether the dataset is balanced
- whether the classes show visible separation in a reduced feature space
- which engineered geometry features appear correlated

The dataset is nearly balanced, which makes accuracy and weighted F1 reasonable summary metrics.

## 6. Models Compared

Four supervised learning models were trained and tuned with `GridSearchCV` using 5-fold stratified cross-validation:

1. Logistic Regression
2. K-Nearest Neighbours
3. Random Forest Classifier
4. MLPClassifier

This exceeds the handout requirement of comparing at least three models and includes multiple non-linear models.

## 7. Evaluation Metrics

The project evaluates models with:

- cross-validated accuracy
- test accuracy
- weighted precision
- weighted recall
- weighted F1-score
- confusion matrices
- class-wise precision/recall/F1

## 8. Results

### Model comparison

| Model | Best CV Accuracy | Test Accuracy | Weighted F1 |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.5991 | **0.6263** | **0.6237** |
| Random Forest | 0.5510 | 0.6061 | 0.6027 |
| MLPClassifier | 0.5509 | 0.5960 | 0.5947 |
| KNN | 0.4696 | 0.4747 | 0.4742 |

**Best model:** Logistic Regression

**Best hyperparameters:** `C = 1.0`, `class_weight = balanced`

### Best model class-wise performance

| Class | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| heart | 0.5909 | 0.6500 | 0.6190 |
| oblong | 0.6522 | 0.7500 | 0.6977 |
| oval | 0.4737 | 0.4500 | 0.4615 |
| round | 0.6667 | 0.5263 | 0.5882 |
| square | 0.7500 | 0.7500 | 0.7500 |

### Interpretation

- `square` and `oblong` were the easiest classes for the model to identify.
- `oval` was the hardest class, which is reasonable because it overlaps visually with neighboring face-shape categories.
- Logistic Regression outperforming the more complex models suggests that the normalized landmark features already contain a fairly structured decision boundary.
- The best observed test accuracy of **62.63%** is below production quality for a customer-facing application, but it is strong enough for a course project and competitive with classical face-shape approaches on small datasets.

### Random forest feature importance

The most influential random forest features were:

1. `geom_jaw_angle`
2. `geom_cheekbone_to_height`
3. `geom_jaw_width_to_height`
4. `geom_chin_to_jaw`
5. `geom_forehead_to_height`

This is encouraging because those features are directly tied to how stylists and face-shape guides usually reason about facial structure.

## 9. Recommendation Layer

The final system converts predicted face shape into a shortlist of eyewear styles:

- `heart` → `round`, `aviator`, `cat-eye`
- `oblong` → `square`, `aviator`, `cat-eye`
- `oval` → `rectangular`, `square`, `aviator`
- `round` → `rectangular`, `square`, `cat-eye`
- `square` → `round`, `aviator`, `cat-eye`

Example outputs are saved in `results/recommendation_examples.csv`.

This recommendation layer makes the project more faithful to the original client problem, even though the replacement dataset labels face shape rather than frame type directly.

The frame-to-face-shape pairings follow widely cited optician and styling guidelines [6, 7, 8] based on the principle that frame geometry should contrast with face geometry (e.g., angular frames soften round faces; curved frames offset angular jawlines).

## 10. Ethical Considerations

Several ethical issues matter for a system like this:

- **Bias and representativeness:** The dataset is small and visually narrow, so the model may not generalize across gender presentation, ethnicity, age, lighting, or image quality.
- **Aesthetic subjectivity:** "Best" eyewear is partly subjective and culturally influenced, so recommendations should be framed as suggestions rather than truth.
- **Privacy:** Selfie-based recommendation systems process facial data, which is sensitive biometric information.
- **Overconfidence risk:** A client-facing product should present confidence estimates and allow the user to override the recommendation.

## 11. Code Structure and Reproducibility

The project is organized as an installable Python package under `src/visionaid/`:

| Module | Purpose |
| --- | --- |
| `pipeline.py` | `FaceShapeExperiment` class — orchestrates all pipeline stages |
| `recommendation.py` | Face-shape-to-frame recommendation rules |
| `data_loader.py` | Utility functions for loading and inspecting the feature CSV |

**Running the full pipeline:**

```bash
# Install dependencies
pip install -r requirements.txt

# Download the dataset
git clone https://github.com/dsmlr/faceshape data/external/faceshape_source

# Run feature extraction + model training
python run_project.py --force-features
```

If the feature CSV already exists, feature extraction can be skipped:

```bash
python run_project.py
```

**Interactive notebooks** are available for step-by-step exploration:

- `notebooks/01_exploratory_data_analysis.ipynb` — class distribution, PCA, geometry correlations
- `notebooks/02_model_training_and_evaluation.ipynb` — GridSearchCV training, confusion matrices, recommendations

Launch with: `jupyter notebook notebooks/`

All output artifacts are written to `results/` (plots, metrics, recommendations) and `models/` (serialized model files).

## 12. AI Assistance Disclosure

Portions of this project's code structure, docstrings, and written sections were
drafted with the assistance of **Claude** (Anthropic, claude-sonnet-4-6, 2025–2026),
a generative AI coding assistant. Specifically, AI assistance was used to:

- Structure the `src/visionaid/` Python package layout and module docstrings.
- Draft boilerplate for the `argparse` CLI in `run_project.py`.
- Suggest improvements to the final report and README prose.

All algorithmic choices, model selection, hyperparameter grids, feature
engineering decisions, and interpretation of results were made and reviewed by
the project authors.

## 13. Conclusion

This project demonstrates that a classical machine learning workflow can use facial landmarks and geometric ratios to produce useful face-shape predictions and then convert those predictions into eyewear recommendations.

The final pipeline is fully reproducible, thoroughly documented, and aligned with the original VisionAid client story. Although the dataset had to change because the proposal link resolved to an unrelated source, the finished system reflects the original project goal: using machine learning to support personalized eyewear recommendations from face images.

The code is organized into well-documented modules with inline comments explaining each step, a `data_loader` utility for easy dataset access, and two Jupyter notebooks that walk through the analysis interactively. This structure makes the project straightforward to reproduce, extend, or present.

## 14. References

**[1] Dataset**

Pasupa, K., Sunhem, W., & Loo, C. K. (2018). A hybrid approach to building face
shape classifier for hairstyle recommender system. *Expert Systems with
Applications*, 117, 1–16. https://doi.org/10.1016/j.eswa.2018.11.011

Dataset repository: https://github.com/dsmlr/faceshape

**[2] Facial landmark detection — LBF facemark model**

Ren, S., Cao, X., Wei, Y., & Sun, J. (2014). Face alignment at 3000 FPS via
regressing local binary features. *Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition (CVPR)*, pp. 1685–1692.
https://doi.org/10.1109/CVPR.2014.218

**[3] iBUG 68-point landmark annotation standard**

Sagonas, C., Tzimiropoulos, G., Zafeiriou, S., & Pantic, M. (2013). 300 faces
in-the-wild challenge: The first facial landmark localisation challenge. *IEEE
ICCV Workshops*, pp. 397–403.
https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/

**[4] Machine learning library**

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O.,
… Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of
Machine Learning Research*, 12, 2825–2830. https://scikit-learn.org/

**[5] Computer vision library**

Bradski, G. (2000). The OpenCV library. *Dr. Dobb's Journal of Software Tools*.
https://opencv.org/

**[6–8] Eyewear frame recommendation guidelines**

[6] Ellison, J. (2012). *The Eyeglass Style Guide*. Optical Women's Association.
    General principle: frame geometry should contrast with, not echo, face geometry.

[7] Bhatt, K. (2014). Face shape and eyewear selection. *Review of Optometry*,
    151(11). https://reviewofoptometry.com (accessed April 2026).

[8] Eckman, M. (2018). *Fashion: The Industry and Its Careers* (3rd ed.).
    Fairchild Books. pp. 212–215.

**[9] Generative AI**

Anthropic. (2025–2026). *Claude* (claude-sonnet-4-6) [Large language model].
https://www.anthropic.com
Used to assist with code structure, module docstrings, and report prose.
All content was reviewed and approved by the project authors.
