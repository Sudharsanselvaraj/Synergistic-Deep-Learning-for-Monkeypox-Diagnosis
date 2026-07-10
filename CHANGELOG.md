# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/) and this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- CLI commands: `predict` (single-image inference), `doctor` (environment check),
  `export` (SavedModel/ONNX), `explain` (Grad-CAM).
- `trinet.inference` package (predict, export) and `examples/` (predict, batch, export, custom).
- GitHub Actions (tests, lint, release), Docker (`docker/`), Model Zoo (`docs/model_zoo.md`).

### Fixed
- Fusion checkpoint now saves **weights + meta** and rebuilds the architecture from code, with a
  post-save verification. A prior full-model `.keras` save had silently produced a broken
  reload; reported metrics were always computed from the correct in-memory model.

## [2.0.0] - 2026-07-10

Complete rebuild as **Tri-Net v2** — a reproducible framework replacing the original,
largely-unimplemented paper code (preserved under `archive/`).

### Added
- Installable `trinet` package (`src/trinet`) with a Typer CLI (`trinet train/evaluate/...`).
- Leakage-free 14-class split from MSLD v2 + ISIC; frozen-backbone feature caching.
- Real PSO ensemble (the paper's missing headline) plus grid/random/equal baselines.
- Modern backbones: ConvNeXt-Tiny (best single, 69.3%) and EfficientNetV2-S.
- Fusion-strategy ablation: Mean / PSO / Concat-MLP / Gated-Attention / Transformer.
- Evaluation suite: confusion, ROC, Cohen's Kappa, McNemar, 5-fold CV, ensemble diversity,
  Grad-CAM, and efficiency profiling.
- Binary Mpox-screening evaluation (98.35% accuracy, 99.58% AUC).
- Symptom module with an honest `sum`-feature leakage ablation.

### Changed
- 14-class accuracy improved 72.6% → **77.23%** (Concat-MLP fusion over ConvNeXt-based ensemble).

### Findings (honest)
- Fine-tuning does not help at this data scale (negative result).
- Higher-capacity fusion (Transformer/attention) did not generalize as well as Concat-MLP
  under this dataset size and training protocol.
- The original 97.86% symptom result does not reproduce (~64–70% by any standard model).
