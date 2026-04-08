# Proposal Deviation Note

## What changed

The proposal cited the Alibaba Tianchi link:

`https://tianchi.aliyun.com/dataset/dataDetail?dataId=56`

When verified on **April 7, 2026**, that URL resolved to a **Taobao display advertising click-through-rate dataset**, not a glasses recommendation dataset. Because of that, the originally proposed source could not be used to build a face-image-based eyewear recommendation project.

## How the project was adjusted

The implementation was changed to use the public `dsmlr/faceshape` dataset instead:

- GitHub: <https://github.com/dsmlr/faceshape>
- Labels: `heart`, `oblong`, `oval`, `round`, `square`
- Data type: facial images

The pipeline still follows the same overall VisionAid objective:

1. Start from a face image.
2. Extract facial landmarks and geometry.
3. Use machine learning to predict a face-related class.
4. Convert the prediction into eyewear recommendations.

## Why this deviation is reasonable

- The original source was not usable for the stated problem.
- The new dataset supports landmark extraction and classical ML.
- The output is still connected to the client goal of frame recommendation.
- The change is transparent, reproducible, and easy to justify in the presentation.

## Short presentation version

"Our proposal referenced a Tianchi dataset for glasses recommendation, but when we verified the link on April 7, 2026, it resolved to an unrelated advertising dataset. To keep the client problem intact, we switched to a public labeled face-shape dataset, extracted facial landmarks, predicted face shape, and then mapped that prediction to eyewear recommendations."

