# Presentation Script Draft

## Opening

"Our project is called VisionAid. VisionAid is a fictional startup that wants to improve the online eyewear shopping experience by recommending glasses from a user's selfie. The main problem we focused on is that many people do not know which frame styles fit their face, and manual face-shape guides are often subjective."

## Problem and Goal

"Our goal was to see whether machine learning could use facial geometry to automate part of that recommendation process. More specifically, we wanted to build a workflow that starts with a face image, extracts facial landmarks, predicts a useful face category, and then converts that prediction into recommended eyewear styles."

## Proposal Deviation

"One important deviation from our proposal is the dataset. In the proposal, we referenced a Tianchi glasses recommendation dataset. However, when we verified the link on April 7, 2026, it resolved to a Taobao advertising click-through dataset instead. Because that source no longer matched the project scope, we replaced it with a public labeled face-shape dataset while keeping the same client problem and machine learning workflow."

## Dataset

"The dataset we used is the public dsmlr faceshape dataset from GitHub. It contains 500 labeled face images across five classes: heart, oblong, oval, round, and square. After face detection and landmark extraction, 493 images were successfully processed into our feature table."

## Preprocessing

"For preprocessing, we detected the dominant face in each image using OpenCV, then extracted 68 facial landmarks with the LBF facemark model. We normalized the landmark coordinates relative to the face bounding box and also engineered geometry features such as jaw width to face height, cheekbone ratios, forehead ratios, and jaw angle."

## Models

"We compared four supervised machine learning models: Logistic Regression, K-Nearest Neighbours, Random Forest, and an MLP neural network. We tuned each one with GridSearchCV using 5-fold stratified cross-validation, then evaluated them on a held-out test set."

## Results

"Our best-performing model was Logistic Regression. It achieved a cross-validated accuracy of about 59.9 percent and a test accuracy of 62.63 percent, with a weighted F1-score of 62.37 percent. Random Forest was the second-best model at about 60.61 percent test accuracy. Among the classes, square and oblong faces were the easiest to classify, while oval faces were the hardest."

## Interpretation

"These results suggest that normalized facial landmarks and geometry contain meaningful predictive signal, even with a relatively small dataset. At the same time, the overlap between some face-shape categories limits performance. So while the model is useful as a recommendation aid, it is not accurate enough to be treated as a final decision-maker in a real commercial product."

## Recommendation Layer

"To connect the model back to the client problem, we mapped predicted face shapes to eyewear frame styles. For example, a predicted round face maps to rectangular, square, or cat-eye frames, while a predicted square face maps to round, aviator, or cat-eye frames. This gives VisionAid a simple end-to-end recommendation workflow from selfie to frame shortlist."

## Ethics

"We also considered ethical concerns. The dataset is small and may not represent the full diversity of real users, which can introduce bias. Facial images are sensitive data, so privacy matters. And finally, style recommendations are subjective, so the system should always present suggestions rather than hard rules."

## Closing

"In conclusion, our project shows that a classical machine learning pipeline using facial landmarks can support personalized eyewear recommendations. Even though we had to change the dataset because the proposal source was not usable, the final system still addresses the original VisionAid client problem in a reproducible and well-documented way."

