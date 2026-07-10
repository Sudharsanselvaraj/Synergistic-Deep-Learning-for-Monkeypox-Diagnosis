# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/) and this project adheres to
[Semantic Versioning](https://semver.org/).

## [0.9.0] - 2026-07-10

Initial public release of **Tri-Net v2** — the official implementation of the Tri-Net study,
re-engineered as an installable, tested, reproducible framework and modernised end to end. The
original study's scripts are preserved under `archive/`.

### Added
- Installable `trinet` package (`src/trinet`) with a Typer CLI
  (`train / evaluate / predict / export / explain / doctor / info / reproduce`).
- Leakage-free 14-class split from MSLD v2 + ISIC; frozen-backbone feature caching.
- Modern backbones: ConvNeXt-Tiny (strongest single backbone, 69.3%) and EfficientNetV2-S,
  alongside the original EfficientNetB4 / InceptionResNetV2 / DenseNet201.
- Learned feature-fusion ablation: Mean / PSO / Concat-MLP / Gated-Attention / Transformer.
- Evaluation suite: confusion matrices, ROC, Cohen's Kappa, McNemar, 5-fold CV, ensemble
  diversity analysis, Grad-CAM, and efficiency profiling.
- Binary Mpox-screening evaluation (98.4% accuracy, 99.6% AUC, with Wilson CIs).
- Symptom module with a leakage-free feature pipeline and classical baselines.
- Engineering: unit tests, GitHub Actions CI (tests + lint + release), Docker, model zoo,
  examples, issue/PR templates.

### Reproducibility & integrity
- Deterministic fusion training so reported results reproduce exactly across runs.
- Checkpoint-integrity protocol: the deployed model is verified in a fresh subprocess to
  reproduce identical logits (`max|Δ| < 1e-5`) before it is saved.

### Findings
- Modern backbones transfer better than older ones under an identical protocol.
- Feature-level fusion (Concat-MLP) outperforms probability-level fusion (Mean / PSO).
- Higher-capacity fusion (Transformer / gated attention) overfits at this dataset scale.
- Fine-tuning frozen backbones does not help at this data scale.
