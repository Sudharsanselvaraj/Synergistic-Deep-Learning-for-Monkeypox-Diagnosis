<div align="center">

<img src="assets/mpox_logo.png" width="220" alt="MPOX">



### A Reproducible Deep-Learning Framework for Multi-Class Skin-Lesion and Symptom-Based Monkeypox (Mpox) Diagnosis

*Official implementation of the Tri-Net study — upgraded with modern backbones, learned feature fusion, and a fully reproducible evaluation framework.*

[![tests](https://github.com/Sudharsanselvaraj/Synergistic-Deep-Learning-for-Monkeypox-Diagnosis/actions/workflows/tests.yml/badge.svg)](https://github.com/Sudharsanselvaraj/Synergistic-Deep-Learning-for-Monkeypox-Diagnosis/actions/workflows/tests.yml)
[![lint](https://github.com/Sudharsanselvaraj/Synergistic-Deep-Learning-for-Monkeypox-Diagnosis/actions/workflows/lint.yml/badge.svg)](https://github.com/Sudharsanselvaraj/Synergistic-Deep-Learning-for-Monkeypox-Diagnosis/actions/workflows/lint.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)](https://www.tensorflow.org/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)](docker/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-purple)](https://github.com/astral-sh/ruff)
[![Reproducible](https://img.shields.io/badge/results-100%25%20reproducible-brightgreen)](docs/reproducibility.md)

**[Abstract](#abstract) • [What's New in v2](#whats-new-in-v2) • [Results](#results) • [Quick Start](#quick-start) • [Model Zoo](#model-zoo) • [Architecture](#architecture) • [Docs](#documentation) • [Citation](#citation)**

</div>

---

## Abstract

The 2022–2024 re-emergence of Monkeypox (Mpox) renewed interest in rapid, non-invasive
computer-aided diagnosis. **Tri-Net** proposes a unified deep-learning approach that couples
skin-lesion image analysis with symptom-based prediction. **Tri-Net v2** re-engineers the study
into an installable, tested, and fully reproducible framework and modernises it end to end. It
evaluates a challenging **14-class** dermatological setting under a strictly **leakage-free**
train/validation/test split, benchmarks four convolutional backbones (including current-generation
**ConvNeXt** and **EfficientNetV2**) as frozen feature extractors, and introduces a **learned
feature-fusion** ablation across five strategies (mean, PSO-weighted, concatenation-MLP, gated
attention, transformer encoder). The best configuration attains **77.2% top-1 accuracy** and
**97.0% macro-AUC** on the fine-grained 14-class task, while the clinically-actionable **binary
Mpox-vs-rest** screening formulation reaches **98.4% accuracy** and **99.6% AUC** with reported
sensitivity, specificity, and Wilson confidence intervals. The framework ships a complete
evaluation suite (5-fold cross-validation, McNemar's test, Cohen's κ, ensemble diversity,
Grad-CAM), a unified CLI, CI, Docker, and a model zoo. Every figure and number is regenerated
from code — nothing is hand-typed.

---

## About

This is the official repository for **Tri-Net: Unified Deep Learning for Skin Lesion and
Symptom-Based Monkeypox Detection**. **Version 2** modernises the framework end to end and turns
it into a reproducible, production-grade research artifact: current-generation backbones, a
learned feature-fusion ablation, a rigorous leakage-free evaluation protocol, and an installable
package with a CLI, tests, CI, and Docker. The original study's scripts are preserved under
[`archive/`](archive/) for provenance and comparison.

**Version 2 confirms strong binary Mpox screening (98.4% accuracy, 99.6% AUC)** — consistent with
the original study — and additionally establishes a more demanding **14-class differential-diagnosis
benchmark (77.2%)** under a strict, leakage-free evaluation protocol.

## What's New in v2

- **Modern backbones** — ConvNeXt-Tiny (the strongest single backbone, 69.3%) and EfficientNetV2-S added alongside the original EfficientNetB4 / InceptionResNetV2 / DenseNet201.
- **Learned feature fusion** — a five-way ablation (Mean, PSO, Concat-MLP, Gated-Attention, Transformer); feature-level fusion is the champion.
- **Rigorous 14-class benchmark** — a leakage-free 80/10/10 split (original images only) that never lets augmented copies straddle train/val/test.
- **Dual-task evaluation** — fine-grained 14-class diagnosis reported separately from binary Mpox screening, each with appropriate metrics and confidence intervals.
- **Full reproducibility** — installable `trinet` package, unified CLI, unit tests, GitHub Actions CI, Docker, and a model zoo. Every number regenerates from committed code.
- **Complete evaluation suite** — confusion matrices, ROC, Cohen's κ, McNemar's test, 5-fold CV, ensemble diversity analysis, and Grad-CAM interpretability.
- **Checkpoint integrity** — a save-time protocol verifies that serialized models reproduce identical logits across processes, so deployed weights behave exactly as evaluated.

## Results

| Task | Metric | Score |
|---|---|---|
| Binary **Mpox screening** (clinical task) | Accuracy / AUC | **98.35% / 99.58%** |
| 14-class fine-grained diagnosis | Accuracy | **77.23%** |

The 14-class result is on a demanding benchmark: 14 classes, leakage-free 80/10/10 split,
~3,000 real images, cross-validated — a substantially harder setting than the binary or
low-cardinality tasks common in the Mpox imaging literature.

<details>
<summary><b>Headline progression (click to expand)</b></summary>

<br>

| Configuration | Test accuracy |
|---|---:|
| Baseline ensemble (three original backbones, Mean/PSO) | 72.60% |
| + ConvNeXt-Tiny backbone (Mean ensemble) | 75.25% |
| **+ Concat-MLP feature fusion (champion)** | **77.23%** |

Net gain from the v2 upgrades: **+4.63 points**. The ensemble's diversity-derived oracle ceiling
is ~80%, so fusion captures most of the headroom available in this ensemble.

</details>

<details>
<summary><b>Single-backbone benchmark (frozen features)</b></summary>

<br>

| Backbone | Accuracy | Macro-F1 | AUC | κ |
|---|---:|---:|---:|---:|
| EfficientNetB4 | 58.4 | 63.3 | 95.0 | 54.2 |
| DenseNet201 | 66.3 | 67.2 | 95.6 | 62.8 |
| InceptionResNetV2 | 68.0 | 68.8 | 96.2 | 64.4 |
| **ConvNeXt-Tiny** | **69.3** | **70.3** | **96.5** | **65.9** |

</details>

<details>
<summary><b>Fusion-strategy ablation (best-3 backbones)</b></summary>

<br>

| Strategy | Learnable | Params | Accuracy | Macro-F1 | AUC | κ |
|---|---|---:|---:|---:|---:|---:|
| **Concat + MLP** | yes | 2.2 M | **77.2** | **79.1** | 97.0 | **74.6** |
| Mean | no | 0 | 75.3 | 77.2 | 97.1 | 72.5 |
| PSO | global | 3 | 73.3 | 74.7 | 96.8 | 70.4 |
| Transformer | yes | 1.5 M | 65.0 | 68.3 | 96.6 | 61.6 |
| Gated Attention | yes | 1.2 M | 63.4 | 63.2 | 95.6 | 59.8 |

Feature-level fusion beats probability fusion; the highest-capacity variants overfit at this dataset size.

</details>

<details>
<summary><b>Cross-validation, ensemble diversity & per-class</b></summary>

<br>

**5-fold CV** (mean ± SD): EfficientNetB4 59.0 ± 2.6 · InceptionResNetV2 59.2 ± 1.4 · DenseNet201 65.5 ± 0.6 · **Tri-Net 69.3 ± 0.8** (beats every base in all folds).

**Diversity**: pairwise disagreement 24.9%, Yule's Q 0.75, error-correlation ρ 0.43, double-fault 19.7% → variance reduction, oracle ceiling ≈ 80%.

**Per-class split**: pox/infectious classes **89.3%** vs dermoscopy (skin-cancer) classes **73.2%** — confusions concentrate on clinically hard pairs (Melanoma↔Nevus, BCC↔SCC).

</details>

<details>
<summary><b>Binary screening — full breakdown</b></summary>

<br>

| Metric | Score | 95% CI (Wilson) |
|---|---:|---|
| Accuracy | 98.4% | 96.2 – 99.3 |
| AUC | 99.6% | — |
| Sensitivity | 86.2% | 69.4 – 94.5 |
| Specificity | 99.6% | 98.0 – 99.9 |

Specificity is very high; sensitivity has a wider interval (few positive test cases). Threshold
tuning toward sensitivity is a natural next step — see [`docs/benchmark.md`](docs/benchmark.md).

</details>

Full tables and ablations: **[`docs/benchmark.md`](docs/benchmark.md)**.

## Research Findings

Alongside the headline metrics, v2 documents several findings from a controlled evaluation:

| # | Finding |
|---|---|
| 1 | **Modern backbones transfer better** — under an identical frozen-feature protocol, ConvNeXt-Tiny > InceptionResNetV2 > DenseNet201 > EfficientNetB4. |
| 2 | **Feature-level fusion > probability-level fusion** — Concat-MLP (77.2%) outperforms both the equal-weight Mean (75.3%) and the PSO-weighted ensemble (73.3%). |
| 3 | **Higher-capacity fusion overfits at this scale** — Transformer and gated-attention fusion generalise worse than the simpler Concat-MLP; a statement about the data regime, not the architectures. |
| 4 | **Fine-tuning does not help at this data scale** — unfreezing backbone layers on ~2.4k images overfits and matches the frozen-ensemble baseline. |
| 5 | **Ensemble robustness is variance reduction** — moderate base-model diversity (disagreement ~25%, error-correlation ~0.43) explains why the ensemble improves over any single backbone. |
| 6 | **Tasks should be reported separately** — 14-class fine-grained diagnosis (77.2%) and binary Mpox screening (98.4%) are different problems with different difficulty and clinical meaning. |

Every finding above is reproducible from committed code on the held-out test set.

## Methodology

**Backbones & transfer learning.** Four ImageNet-pretrained backbones are compared, each with its own required preprocessing. Each *frozen* backbone feeds a lightweight head (Dense→BatchNorm→Dense→Dropout→Softmax); features are cached once so training runs in seconds/epoch. Class imbalance is handled with balanced class weights.

**Fusion strategies.** Five ways to combine backbones on identical features: **Mean** (equal-weight probability average), **PSO** (a global weight vector optimised by Particle Swarm Optimization), **Concat-MLP** (concatenate features → MLP), **Gated Attention** (per-image attention over backbone tokens), and **Transformer** (self-attention encoder).

**Symptom module.** A compact CNN with Logistic-Regression / Random-Forest / XGBoost baselines on the structured symptom dataset, using a leakage-free feature pipeline.

**Evaluation protocol.** Accuracy, macro-P/R/F1, one-vs-rest macro-AUC, and Cohen's κ on the held-out test set; 5-fold CV on original features; McNemar's test; ensemble diversity; Grad-CAM; and Wilson CIs for binary screening. Full details: [`docs/architecture.md`](docs/architecture.md).

## Architecture

```
datasets ──► features ──► base heads ──► ensemble ─┐
 (download,   (frozen      (per-backbone   (PSO /   ├─► evaluation
  prepare)     backbone     classifier)     mean)   │    (metrics, confusion,
               cache)                               │     ROC, Kappa, McNemar,
                                    fusion ─────────┘     diversity, Grad-CAM)
                                 (Concat-MLP / attention / transformer)
```

Backbones are frozen and cached to disk as feature vectors, so head training runs in seconds per epoch rather than hours.

| Backbone | Pooled dim | Notes |
|---|---:|---|
| EfficientNetB4 | 1792 | Metal-compatible |
| InceptionResNetV2 | 1536 | Metal-compatible |
| DenseNet201 | 1920 | Metal-compatible |
| **ConvNeXt-Tiny** | 768 | best single backbone; CPU extraction on Apple Metal |
| EfficientNetV2-S | 1280 | CPU extraction on Apple Metal |

Full details: [`docs/architecture.md`](docs/architecture.md).

## Quick Start

```bash
git clone https://github.com/Sudharsanselvaraj/Synergistic-Deep-Learning-for-Monkeypox-Diagnosis.git
cd Synergistic-Deep-Learning-for-Monkeypox-Diagnosis
python -m venv .venv && source .venv/bin/activate
pip install -e .              # add ".[metal]" on Apple Silicon, ".[dev]" for tests/linting
```

Run the full pipeline:

```bash
trinet download        # fetch datasets (needs a Kaggle API token)
trinet prepare         # leakage-free 14-class split + clean symptom data
trinet features        # cache frozen-backbone features
trinet train           # train classification heads
trinet fusion          # fusion-strategy ablation (Concat-MLP is the champion)
trinet evaluate --champion   # figures + tables from the best model
```

Or the whole image pipeline in one shot: `trinet benchmark`. Full CLI reference: `trinet --help`.

**Inference on a single image:**

```bash
trinet predict lesion.jpg
```

```python
from trinet.inference.predict import predict_image
print(predict_image("lesion.jpg"))
```

**As a library:**

```python
from trinet.models.fusion import build_concat_mlp
from trinet.evaluation.metrics import compute_scores
```

## Model Zoo

| Model | Task | Accuracy | AUC | Weights |
|---|---|---:|---:|:--:|
| **Tri-Net v2 (Concat-MLP fusion)** | 14-class | **77.2** | 97.0 | ✅ |
| ConvNeXt-Tiny (single backbone) | 14-class | 69.3 | 96.5 | ✅ |
| Mean ensemble | 14-class | 75.3 | **97.1** | — |
| PSO ensemble | 14-class | 73.3 | 96.8 | ✅ |
| **Tri-Net v2 (Mpox screening)** | binary | **98.4** | **99.6** | ✅ |

Full breakdown, params, and Macro-F1 / Kappa per model: **[`docs/model_zoo.md`](docs/model_zoo.md)**. Pretrained weights are attached to the matching [GitHub Release](https://github.com/Sudharsanselvaraj/Synergistic-Deep-Learning-for-Monkeypox-Diagnosis/releases).

## Documentation

| Doc | Covers |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Models, fusion strategies, full pipeline |
| [`docs/model_zoo.md`](docs/model_zoo.md) | Pretrained models, weights, usage |
| [`docs/datasets.md`](docs/datasets.md) | Data sources, licensing, 14-class composition |
| [`docs/benchmark.md`](docs/benchmark.md) | Full results and ablations |
| [`docs/reproducibility.md`](docs/reproducibility.md) | How to regenerate every number in this README |
| [`docs/roadmap.md`](docs/roadmap.md) | Project plan, milestones, Phase 2 goals |

## Reproducibility

- Global seed `42` fixed across NumPy, TensorFlow, and scikit-learn.
- The 80/10/10 split is stratified, seeded, and deterministic — the same images always land in the same partition.
- Metrics are computed only on the held-out test set; cross-validation uses un-augmented features so augmented copies never straddle folds.
- Generated artifacts (`outputs/`, `data/`) are git-ignored by design — regenerate everything with the CLI, nothing paper-facing is hand-typed.

```bash
trinet download && trinet prepare && trinet features
trinet train && trinet fusion && trinet evaluate --champion
trinet cross-val && trinet diversity && trinet gradcam
```

Reference environment: Apple M-series, 16 GB, Python 3.11, TensorFlow 2.16.2. Note: `tensorflow-metal` hangs on EfficientNetV2-S and errors on ConvNeXt — extract those on CPU with `trinet features --cpu`. Full notes: [`docs/reproducibility.md`](docs/reproducibility.md).

## Repository Structure

```
.
├── src/trinet/            installable package
│   ├── config.py          central configuration (paths, 14 classes, hyperparameters)
│   ├── cli.py             Typer command-line interface
│   ├── datasets/          dataset download + leakage-free preparation
│   ├── models/            backbones, fusion strategies, PSO ensemble
│   ├── training/          feature caching, head training, fine-tuning, symptom model
│   ├── evaluation/        metrics, figures, cross-validation, diversity, efficiency
│   ├── explainability/    Grad-CAM
│   ├── inference/         single-image prediction + model export
│   └── utils/             doctor / info diagnostics
├── experiments/           one-off studies (e.g. binary screening with Wilson CIs)
├── configs/               YAML experiment presets
├── docs/                  architecture, datasets, benchmark, reproducibility, model_zoo, roadmap
├── examples/              runnable usage examples (predict, batch, ONNX export, custom dataset)
├── scripts/               shell helpers (download, train, evaluate, benchmark)
├── tests/                 unit tests (metrics, models, fusion, dataset, checkpoint integrity)
├── docker/                Dockerfile + docker-compose
├── app/                   deployment stub (API / web / docker) — planned
├── archive/               original Tri-Net scripts, preserved unmodified
├── assets/                logo + figures used in docs and README
└── .github/               CI workflows + issue/PR templates
```

## Limitations

- Public composite datasets limit demographic/acquisition diversity; the test set is modest (303 images), so metrics carry non-trivial confidence intervals.
- **No external clinical validation** — results are experimental, not clinically certified.
- Image and symptom modules are trained on **disjoint patient populations**, so genuine multimodal (joint) fusion is future work pending a paired dataset.
- The dermoscopy sub-problem is inherently hard; ~85–90% is near the field-wide ceiling even with far larger datasets and metadata.

> **Research artifact, not a medical device.** Do not use for clinical decision-making without regulatory approval and prospective validation. See [`SECURITY.md`](SECURITY.md).

## Roadmap

- Complete the backbone table (EfficientNetV2-S) and an efficiency/latency profile.
- A **hierarchical classifier** (pox-group vs dermoscopy-group → within-group heads), motivated by the 99.7% group-separation result — the most promising route to lift fine-grained accuracy.
- Larger, higher-resolution dermoscopy data + patient metadata; self-supervised (DINOv2) features.
- A genuine multimodal model, contingent on a paired image+symptom dataset. See [`docs/roadmap.md`](docs/roadmap.md).

## Contributing

Contributions are welcome — see **[`CONTRIBUTING.md`](CONTRIBUTING.md)** for dev setup, coding guidelines, and PR expectations. The short version: reusable code lives in `src/trinet/`, experiment-specific code lives in `experiments/`, and every reported number must be regenerable from committed code on the held-out test set.

## Citation

If you use this work, please cite it — see **[`CITATION.cff`](CITATION.cff)** for the definitive entry. A BibTeX-style reference:

```bibtex
@software{trinet_v2_2026,
  author  = {Selvaraj, Sudharsan},
  title   = {Tri-Net v2: A Reproducible Deep-Learning Framework for Mpox Skin-Lesion Diagnosis},
  year    = {2026},
  url     = {https://github.com/Sudharsanselvaraj/Synergistic-Deep-Learning-for-Monkeypox-Diagnosis}
}
```

## Acknowledgments

Built on the MSLD v2 and ISIC skin-lesion datasets and the Kaggle Monkeypox symptom dataset —
full sourcing and licensing in [`docs/datasets.md`](docs/datasets.md).

## License

[MIT](LICENSE)
