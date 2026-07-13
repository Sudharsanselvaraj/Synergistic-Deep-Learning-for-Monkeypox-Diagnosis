<div align="center">

<img src="assets/mpox_logo.png" alt="Tri-Net" width="260"/>

# Tri-Net

**A reproducible, leakage-free deep-learning framework for Monkeypox and skin-lesion diagnosis**

*Monkeypox is diagnosed by eye — a clinician inspects a lesion and listens to symptoms — yet Mpox
shares its look with chickenpox, measles, and a dozen dermatological conditions. That makes early
diagnosis subjective and error-prone. Tri-Net attacks it as an honest machine-learning problem:
modern CNN backbones, learned feature-level fusion, and an evaluation protocol built to be
believed — every number regenerates from code on a strictly leakage-free split.*

[![Paper](https://img.shields.io/badge/Scientific%20Reports-published-8A1538)](https://doi.org/10.1038/s41598-026-61490-x)
[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41598--026--61490--x-blue)](https://doi.org/10.1038/s41598-026-61490-x)
[![PyPI](https://img.shields.io/pypi/v/mpox-trinet?color=blue&label=pypi)](https://pypi.org/project/Mpox-Trinet/)
[![tests](https://github.com/Sudharsanselvaraj/Synergistic-Deep-Learning-for-Monkeypox-Diagnosis/actions/workflows/tests.yml/badge.svg)](.github/workflows/tests.yml)
[![lint](https://github.com/Sudharsanselvaraj/Synergistic-Deep-Learning-for-Monkeypox-Diagnosis/actions/workflows/lint.yml/badge.svg)](.github/workflows/lint.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB)](pyproject.toml)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**[What is Tri-Net?](#what-is-tri-net) • [Why](#why-tri-net) • [Install](#installation) • [Quick start](#quick-start) • [Results](#results-measured-leakage-free) • [Architecture](#architecture) • [Docs](#documentation) • [Cite](#citation)**

</div>

---

## What is Tri-Net?

The recent re-emergence of Monkeypox made fast, precise diagnosis matter again — but the standard
workflow is a clinician **visually** examining skin lesions and interpreting **described**
symptoms. That is subjective, and Mpox overlaps heavily with other skin and viral conditions, so
mistakes cluster exactly where they are most costly: early, ambiguous cases.

**Tri-Net** approaches this with a unified deep-learning pipeline and reports **two clearly
separated tasks** rather than one flattering headline number:

- **14-class fine-grained diagnosis** — Mpox vs. 13 confounding conditions (chickenpox, cowpox,
  measles, HFMD, melanoma, basal-cell carcinoma, …). A genuinely hard problem: **77.2% accuracy /
  79.1% macro-F1** on a leakage-free split of ~3,000 real images.
- **Binary Mpox screening** — the clinically actionable *"is this Mpox?"* question: **98.4%
  accuracy, 99.6% AUC**, with Wilson confidence intervals.

The design principle is honesty over headlines. Many published Mpox models report 95–99% on
binary or 4-class tasks where **pre-augmented images leak** across train/test — inflating accuracy
without improving the model. Tri-Net splits on **original images only**, applies augmentation at
training time only, and regenerates every figure and table from saved measurements. Nothing in
the evaluation is hand-typed.

> **Honest scope.** This is a research artifact accompanying a published study — **not a medical
> device**. It is not for clinical use without regulatory approval and prospective validation. The
> repo is deliberately candid about what does *not* work: the symptom dataset is largely synthetic
> and weakly predictive (~64–70%, **not** the originally-claimed 97.86%), higher-capacity fusion
> overfits at this data scale, and fine-tuning frozen backbones does not help. See
> [Results](#results-measured-leakage-free) and [`docs/benchmark.md`](docs/benchmark.md).

---

## Why Tri-Net

- **It refuses to leak.** The 14-class split is stratified **80/10/10 on original images only**
  (2426 train / 303 val / 303 test), so augmented copies of a training image can never appear in
  validation or test — the single most common source of inflated Mpox-imaging accuracy.
- **It reports two tasks, honestly.** Fine-grained 14-class diagnosis and binary Mpox screening
  are different problems and are evaluated separately, never blended into one number.
- **Every number regenerates.** `trinet reproduce` rebuilds every figure and table from code on a
  seeded, deterministic pipeline — the deployed model is even re-verified in a fresh subprocess to
  reproduce identical logits (`max|Δ| < 1e-5`) before it is saved.
- **It benchmarks modern backbones under one protocol.** ConvNeXt-Tiny, EfficientNetV2-S,
  DenseNet201, InceptionResNetV2, EfficientNetB4 — all as frozen feature extractors with correct
  per-model preprocessing, so comparisons are apples-to-apples.
- **It's honest about negatives.** Fine-tuning doesn't help at ~2.4k images; Transformer / gated
  attention fusion overfit; the symptom module's 97.86% does not reproduce. These are reported as
  findings, not hidden.
- **It installs and runs.** A real Python package (`pip install mpox-trinet`), a unified `trinet`
  CLI, unit tests, CI, and Docker — not a folder of notebooks.

---

## Key features

| | | |
|---|---|---|
| **Leakage-free evaluation** | **Modern backbone benchmark** | **Learned feature fusion** |
| Stratified 80/10/10 split on *original* images only; augmentation at train time only. The composite 14-class set is documented, not hand-waved. | Five ImageNet backbones as frozen extractors, each with its own correct preprocessing and cached features (seconds/epoch head training). | Five fusion strategies — Mean, PSO, Concat-MLP (champion), Gated-Attention, Transformer — as a controlled ablation, not a single opaque model. |
| **Two-task reporting** | **Statistical evaluation** | **Explainability** |
| 14-class fine-grained diagnosis **and** binary Mpox screening, evaluated and reported separately. | Accuracy / macro-F1 / AUC / Cohen's κ, McNemar's test, 5-fold cross-validation, ensemble-diversity analysis, Wilson CIs. | Grad-CAM heatmaps showing where each prediction actually looks — regenerated from code. |
| **Reproducible pipeline** | **Checkpoint integrity** | **Packaged & tested** |
| Seeded (`42`), deterministic; `trinet reproduce` rebuilds every artifact from saved data. | The deployed champion is re-run in a fresh subprocess and must reproduce identical logits (`max|Δ| < 1e-5`) before saving. | Installable `mpox-trinet` package + `trinet` CLI, unit tests, GitHub Actions CI (tests + lint), Docker, model zoo. |

---

## Installation

```bash
pip install mpox-trinet
```

The distribution installs as **`mpox-trinet`**, imports as **`trinet`**, and exposes a **`trinet`**
CLI. On Apple Silicon, add the Metal extra for GPU acceleration:

```bash
pip install "mpox-trinet[metal]"
```

From a clone, for development:

```bash
git clone https://github.com/Sudharsanselvaraj/Synergistic-Deep-Learning-for-Monkeypox-Diagnosis.git
cd Synergistic-Deep-Learning-for-Monkeypox-Diagnosis
make venv                 # creates .venv, installs requirements.txt
pip install -e ".[dev]"   # editable install with test + lint tooling
```

**Requirements:** Python 3.11+, TensorFlow 2.16; a Kaggle API token (`~/.kaggle`) to fetch the
datasets. Full setup — including Docker and Apple-Metal specifics — in
[`docs/reproducibility.md`](docs/reproducibility.md).

---

## Quick start

```bash
trinet doctor              # environment & readiness diagnostics
trinet download            # fetch datasets from Kaggle (needs a Kaggle API token)
trinet prepare             # leakage-free 14-class split + clean symptom data
trinet features            # cache frozen-backbone features
trinet fusion              # train the fusion model + deploy the verified champion
trinet predict lesion.jpg  # run inference on one image
```

Run the full image pipeline in one shot with `trinet benchmark`, or regenerate **every** artifact
(splits → features → heads → fusion → evaluation → figures) with `trinet reproduce`. Full
reference: `trinet --help`.

### Python API

```python
from trinet.inference.predict import predict_image

result = predict_image("lesion.jpg")
print(result["diagnosis"])       # e.g. "Mpox"
print(result["confidence"])      # e.g. 0.83
print(result["mpox_screening"])  # {"label": "Mpox", "mpox_probability": 0.83}
```

Or compose the library directly:

```python
from trinet.models.fusion import build_concat_mlp
from trinet.evaluation.metrics import compute_scores
```

### CLI at a glance

| Stage | Commands |
|---|---|
| **Data** | `download` · `prepare` · `features` |
| **Train** | `train` (per-backbone heads) · `finetune` · `ensemble` (PSO/mean) · `fusion` (champion) · `symptom` |
| **Evaluate** | `evaluate [--champion]` · `cross-val` (5-fold) · `diversity` · `efficiency` |
| **Explain** | `gradcam` · `explain` |
| **Infer / ship** | `predict <image>` · `export --format savedmodel\|onnx` |
| **Utility** | `doctor` · `info` · `benchmark` · `reproduce` |

---

## Results (measured, leakage-free)

All numbers are produced by runnable code on a **leakage-free 14-class split**
(2426 train / 303 val / 303 test unique images) and regenerated by the pipeline — nothing is
hand-typed. Full tables, ablations, and cross-validation: [`docs/benchmark.md`](docs/benchmark.md).

**Headline progression** — what each modernization actually buys:

| Configuration | Test accuracy |
|---|--:|
| Reproducible baseline (paper's 3 backbones, Mean/PSO) | 72.6% |
| + ConvNeXt-Tiny backbone (Mean ensemble) | 75.25% |
| **+ Concat-MLP feature fusion (champion)** | **77.23%** |

Net gain over the reproducible baseline: **+4.63 pp**. The ensemble oracle ceiling (from the
diversity analysis) is ~80%, so fusion captures most of the available headroom.

**Single-backbone benchmark** (frozen features, identical pipeline):

| Backbone | Accuracy | Macro-F1 | AUC | κ |
|---|--:|--:|--:|--:|
| EfficientNetB4 | 58.4 | 63.3 | 95.0 | 54.2 |
| DenseNet201 | 66.3 | 67.2 | 95.6 | 62.8 |
| InceptionResNetV2 | 68.0 | 68.8 | 96.2 | 64.4 |
| **ConvNeXt-Tiny** | **69.3** | **70.3** | **96.5** | **65.9** |

**Fusion-strategy ablation** (best-3 backbones) — feature fusion beats probability fusion, and
more capacity is *not* automatically better:

| Strategy | Learnable | Params | Accuracy | Macro-F1 | AUC | κ |
|---|---|--:|--:|--:|--:|--:|
| **Concat-MLP** | yes | 2.2M | **77.23** | **79.08** | 96.98 | **74.64** |
| Mean | no | 0 | 75.25 | 77.24 | **97.11** | 72.53 |
| PSO | global | 3 | 73.27 | 74.74 | 96.75 | 70.39 |
| Transformer | yes | 1.5M | 65.02 | 68.33 | 96.63 | 61.59 |
| Gated Attention | yes | 1.2M | 63.37 | 63.21 | 95.61 | 59.80 |

**Binary Mpox screening** — the clinically actionable formulation:

| Metric | Score | 95% CI (Wilson) |
|---|--:|---|
| Accuracy | 98.35% | 96.2 – 99.3 |
| AUC | 99.58% | — |
| Sensitivity | 86.2% | 69.4 – 94.5 |
| Specificity | 99.6% | 98.0 – 99.9 |

<div align="center">

| | |
|:---:|:---:|
| <img src="assets/confusion_matrix.png" width="410"/> | <img src="assets/roc_curves.png" width="410"/> |
| <sub>14-class confusion matrix (leakage-free test set)</sub> | <sub>ROC — binary Mpox screening</sub> |

<img src="assets/gradcam.png" width="820"/>

<sub>Grad-CAM — where each backbone actually looks</sub>

</div>

> **On comparability.** 77.23% is on a demanding benchmark — 14 classes, leakage-free, ~3,000 real
> images, cross-validated. Studies reporting 95–99% on binary or 4-class tasks with augmentation
> leakage are **not** comparable. We do not attempt to recover the original paper's unreproducible
> 97.86% symptom result; the contribution is a benchmark whose conclusions can be trusted.

---

## Architecture

Tri-Net is an installable Python package (`src/trinet`) with a thin `trinet` CLI over a linear,
cacheable pipeline:

```
datasets ──► features ──► base heads ──► ensemble  ─┐
 (download,   (frozen      (per-backbone  (PSO /    ├─► evaluation
  prepare)     backbone     classifier)    mean)    │    (metrics, confusion,
               cache)                               │     ROC, κ, McNemar,
                                   fusion ──────────┘     diversity, Grad-CAM)
                                (Concat-MLP / attention / transformer)
```

**Backbones** (`trinet.models.backbones`) — ImageNet-pretrained, used **frozen**, each with its
own preprocessing (getting this wrong silently wrecks accuracy); features are cached to disk so
head training is seconds/epoch:

| Backbone | Pooled dim | Notes |
|---|--:|---|
| EfficientNetB4 | 1792 | Metal-compatible |
| InceptionResNetV2 | 1536 | Metal-compatible |
| DenseNet201 | 1920 | Metal-compatible |
| ConvNeXt-Tiny | 768 | best single backbone; CPU extraction on Apple Metal |
| EfficientNetV2-S | 1280 | CPU extraction on Apple Metal |

**Fusion** (`trinet.models.fusion`) — five strategies from trivial to expressive: **Mean** / **PSO**
combine per-model *probabilities*; **Concat-MLP** (the champion) concatenates *features* → MLP;
**Gated Attention** and **Transformer** add per-image attention over backbones (and overfit at
this data scale). Details: [`docs/architecture.md`](docs/architecture.md).

---

## Repository structure

```
src/trinet/         installable package — the whole framework
  datasets/         download · leakage-free lesion split · symptom cleaning
  models/           backbones (frozen extractors) · fusion · ensemble
  training/         feature caching · base heads · finetune · symptom
  evaluation/       metrics · figures · cross_val · diversity · efficiency
  explainability/   Grad-CAM
  inference/        predict · export (SavedModel / ONNX)
  cli.py            the `trinet` Typer CLI
configs/            YAML presets (convnext · fusion · ensemble · binary)
docs/               architecture · datasets · benchmark · reproducibility · model_zoo · roadmap
experiments/        one-off studies (binary screening with Wilson CIs)
examples/           runnable end-to-end usage examples
tests/              unit tests (metrics, models, fusion, checkpoint integrity)
scripts/            shell helpers
archive/            original Tri-Net study scripts, frozen for provenance
assets/             figures + logo
```

---

## Reproducibility

Every figure and table is regenerated from saved measurements — no hand-entered numbers. The
pipeline is seeded (`42`), deterministic, and evaluated on the held-out test set only; generated
artifacts (`outputs/`, `data/`) are git-ignored and rebuilt by the CLI.

```bash
trinet download && trinet prepare && trinet features
trinet train && trinet fusion && trinet evaluate --champion
trinet cross-val && trinet diversity && trinet gradcam
```

Fusion training is deterministic so reported results reproduce exactly across runs, and the
deployed champion is verified in a fresh subprocess to reproduce identical logits (`max|Δ| < 1e-5`)
before it is saved. Reference environment: Python 3.11, TensorFlow 2.16.2. Full notes (incl.
Apple-Metal specifics): [`docs/reproducibility.md`](docs/reproducibility.md).

---

## Documentation

| Doc | What's inside |
|---|---|
| [Architecture](docs/architecture.md) | Pipeline, backbones (with pooled dims), the five fusion strategies |
| [Datasets](docs/datasets.md) | The documented 14-class composite, the leakage-free split, symptom-data honesty |
| [Benchmark](docs/benchmark.md) | Full results, ablations, cross-validation, and the well-supported conclusions |
| [Reproducibility](docs/reproducibility.md) | Environment, determinism, Apple-Metal specifics, integrity protocol |
| [Model Zoo](docs/model_zoo.md) | Downloadable weights and how to load them |
| [Roadmap](docs/roadmap.md) | What's next |

---

## Original publication

> **Tri-Net: unified deep learning for skin lesion and symptom-based monkeypox detection**
> S. Sudharsan · Prabu Selvam† · Nirmala Veeramani† · B. Kiran Kumar · Nikola Ivković · Korhan Cengiz
> *Scientific Reports* (2026) · **published 13 July 2026** · open access (CC BY-NC-ND 4.0)
> [doi.org/10.1038/s41598-026-61490-x](https://doi.org/10.1038/s41598-026-61490-x)
> † Corresponding authors · Received 18 Nov 2025 · Accepted 06 Jul 2026

This repository is the code artifact referenced by the paper's **Code Availability** statement — a
from-scratch, tested, reproducible implementation that keeps the paper's core idea (a Tri-Net
ensemble plus a symptom-based classifier) and makes every reported number independently verifiable.
The original study scripts are preserved unmodified under [`archive/`](archive/) for provenance.

**Recommended Code Availability statement:**

> The custom code, trained model weights, and evaluation pipeline supporting the findings of this
> study are openly available at
> [github.com/Sudharsanselvaraj/Synergistic-Deep-Learning-for-Monkeypox-Diagnosis](https://github.com/Sudharsanselvaraj/Synergistic-Deep-Learning-for-Monkeypox-Diagnosis)
> and on PyPI (`pip install mpox-trinet`). Archive a tagged release via the GitHub–Zenodo
> integration to mint a software DOI.

---

## Citation

If you use Tri-Net in your research, please cite the paper and the software framework (see
[`CITATION.cff`](CITATION.cff) for the machine-readable entry):

> Sudharsan, S., Selvam, P., Veeramani, N. et al. Tri-Net: unified deep learning for skin lesion and
> symptom-based monkeypox detection. *Sci Rep* (2026). https://doi.org/10.1038/s41598-026-61490-x

```bibtex
@article{sudharsan2026trinet,
  title   = {Tri-Net: unified deep learning for skin lesion and symptom-based monkeypox detection},
  author  = {Sudharsan, S. and Selvam, Prabu and Veeramani, Nirmala and
             Kiran Kumar, B. and Ivkovi{\'c}, Nikola and Cengiz, Korhan},
  journal = {Scientific Reports},
  publisher = {Springer Nature},
  year    = {2026},
  doi     = {10.1038/s41598-026-61490-x},
  url     = {https://doi.org/10.1038/s41598-026-61490-x}
}

@software{trinet_v2_2026,
  author  = {Sudharsan, S.},
  title   = {Tri-Net v2: A Reproducible Deep-Learning Framework for Mpox Skin-Lesion Diagnosis},
  year    = {2026},
  url     = {https://github.com/Sudharsanselvaraj/Synergistic-Deep-Learning-for-Monkeypox-Diagnosis}
}
```

---

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup and the PR
checklist, and please follow the [Code of Conduct](CODE_OF_CONDUCT.md). The ground rules are the
same ones that make the results trustworthy: **never fabricate data**, keep the split leakage-free,
and verify changes against the test suite (`pytest`) and linter (`ruff`) before opening a PR.
Security reports: [SECURITY.md](SECURITY.md).

---

## Acknowledgements

Built on the MSLD v2 and ISIC skin-lesion datasets and the Kaggle Monkeypox symptom dataset — full
sourcing and licensing in [`docs/datasets.md`](docs/datasets.md) — and on TensorFlow/Keras and the
ImageNet-pretrained backbones it benchmarks. Tri-Net is a research artifact; its README and docs
try to stay as honest about scope and limitations as the paper itself has to be.

## License

Released under the [MIT License](LICENSE). **Research artifact — not a medical device**; not for
clinical use without regulatory approval and prospective validation.
