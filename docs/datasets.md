# Datasets

The 14-class lesion set is a **documented composite** — no single public dataset contains all
14 classes. It is assembled honestly from:

| Source (Kaggle) | Classes contributed |
|---|---|
| `joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20` (MSLD v2) | Mpox, Chickenpox, Cowpox, Measles, HFMD, Healthy |
| `nodoubttome/skin-cancer9-classesisic` (ISIC) | Melanoma, MelanocyticNevi, BasalCellCarcinoma, ActinicKeratosis, BenignKeratosis, Dermatofibroma, VascularLesion, SquamousCellCarcinoma |
| `shuvoalok/monkeypox-dataset` | symptom module (binary Mpox screening) |

Fetch everything with `trinet download` (requires `~/.kaggle` credentials).

## Split (leakage-free)

`trinet prepare` builds a stratified **80/10/10** split on **original images only**. This is
critical: MSLD ships pre-augmented images, and letting augmented copies of a training image
leak into validation/test is a prime cause of inflated accuracy in prior work. Augmentation
is applied at *training time only*.

Resulting split: **2426 train / 303 val / 303 test** unique images across 14 classes.

## Symptom data

The Kaggle symptom set is largely synthetic and weakly predictive; the cleaning step drops the
original `sum` feature (dubious methodology) and the pipeline reports honest numbers (~64–70%,
not the previously-claimed 97.86%).

## Two tasks, two evaluations

- **14-class fine-grained diagnosis** — hard; ~77% (dermoscopy cancer classes dominate errors).
- **Binary Mpox screening** — the clinical question; ~98% accuracy / ~99.6% AUC.

These are different classification problems and are reported separately.
