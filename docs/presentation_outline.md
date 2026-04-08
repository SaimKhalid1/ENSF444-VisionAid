# 7-10 Minute Presentation Outline

## Slide 1: Title and Client

- Project title: VisionAid
- Team members
- Client: fictional startup for selfie-based eyewear recommendations
- Main question: can we use facial geometry and machine learning to recommend better frame styles?

## Slide 2: Problem Statement

- Buying glasses online is difficult without trying frames on
- Manual face-shape guides are subjective and inconsistent
- VisionAid wants a fast, selfie-based recommendation workflow

## Slide 3: Original Proposal and Deviation

- Original plan: use Alibaba Tianchi glasses recommendation dataset
- Verified on April 7, 2026: the proposal link resolved to a Taobao ad CTR dataset
- Adjustment: use a public labeled face-shape dataset and preserve the client goal through a recommendation layer

## Slide 4: Dataset

- Dataset: `dsmlr/faceshape`
- 500 labeled face images
- 5 classes: heart, oblong, oval, round, square
- 493 images successfully processed after landmark extraction

## Slide 5: ML Workflow

- Detect face with OpenCV
- Extract 68 facial landmarks
- Normalize landmark coordinates
- Engineer geometry ratios
- Train and tune multiple classifiers

## Slide 6: Models Compared

- Logistic Regression
- KNN
- Random Forest
- MLPClassifier
- Mention `GridSearchCV` and 5-fold stratified cross-validation

## Slide 7: Results

- Logistic Regression performed best
- Test accuracy: 62.63%
- Weighted F1: 62.37%
- Random Forest was close behind at 60.61%
- Oval faces were hardest to classify

## Slide 8: Interpretation

- Facial geometry contains enough signal for a usable classical ML baseline
- Some face-shape boundaries overlap, which lowers performance
- The project is promising as a recommender aid, but not reliable enough for unsupervised production use

## Slide 9: Eyewear Recommendation Layer

- Predicted face shape is mapped to frame styles
- Examples:
  - round → rectangular / square / cat-eye
  - square → round / aviator / cat-eye
  - oval → rectangular / square / aviator

## Slide 10: Ethics and Conclusion

- Bias and limited dataset diversity
- Facial privacy concerns
- Recommendations should be suggestions, not hard rules
- Final takeaway: the project successfully demonstrates a landmark-based ML workflow for personalized eyewear recommendations

