<div align="center">

# Tri-Net v2

**A reproducible deep-learning framework for Mpox skin-lesion diagnosis**

[![tests](https://github.com/Sudharsanselvaraj/Synergistic-Deep-Learning-for-Monkeypox-Diagnosis/actions/workflows/tests.yml/badge.svg)](https://github.com/Sudharsanselvaraj/Synergistic-Deep-Learning-for-Monkeypox-Diagnosis/actions/workflows/tests.yml)
[![lint](https://github.com/Sudharsanselvaraj/Synergistic-Deep-Learning-for-Monkeypox-Diagnosis/actions/workflows/lint.yml/badge.svg)](https://github.com/Sudharsanselvaraj/Synergistic-Deep-Learning-for-Monkeypox-Diagnosis/actions/workflows/lint.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-purple)](https://github.com/astral-sh/ruff)

</div>

Tri-Net v2 is an honest, leakage-free rebuild of the *Tri-Net* Mpox-diagnosis work. It couples
modern CNN backbones with **learned feature fusion** and a rigorous evaluation suite, and it
reports two clearly-separated tasks: **14-class fine-grained diagnosis** and **binary Mpox
screening**.

> Official reproducible implementation of *"Tri-Net: Unified Deep Learning for Skin Lesion and
> Symptom-Based Monkeypox Detection"*, rebuilt as **Tri-Net v2** — a research framework rather
> than one-off paper code. The original implementation is preserved under [`archive/`](archive/).

## Results

| Task | Metric | Score |
|---|---|---|
| Binary **Mpox screening** | Accuracy / AUC | **98.35% / 99.58%** |
| 14-class fine-grained diagnosis | Accuracy | **77.23%** |

The 14-class result is on a demanding benchmark: 14 classes, leakage-free split, ~3,000 real
images, cross-validated. See [`docs/benchmark.md`](docs/benchmark.md) for the full tables,
ablations, and honest negative results.

<div align="center"><img src="assets/confusion_matrix.png" width="520" alt="14-class confusion matrix"></div>

## Install

```bash
git clone https://github.com/Sudharsanselvaraj/Synergistic-Deep-Learning-for-Monkeypox-Diagnosis.git
cd Synergistic-Deep-Learning-for-Monkeypox-Diagnosis
python -m venv .venv && source .venv/bin/activate
pip install -e .              # add ".[metal]" on Apple Silicon, ".[dev]" for tests/linting
```

## Quick start

```bash
trinet download        # fetch datasets (needs a Kaggle API token)
trinet prepare         # leakage-free 14-class split + clean symptom data
trinet features        # cache frozen-backbone features
trinet train           # train classification heads
trinet fusion          # fusion-strategy ablation (Concat-MLP is the champion)
trinet evaluate --champion   # figures + tables from the best model
```

Run the whole image pipeline at once with `trinet benchmark`. Full CLI: `trinet --help`.

Use the library directly:

```python
from trinet.models.fusion import build_concat_mlp
from trinet.evaluation.metrics import compute_scores
```

## Model Zoo

| Model | Task | Accuracy | AUC | Weights |
|---|---|---:|---:|:--:|
| **Tri-Net v2 (Concat-MLP fusion)** | 14-class | **77.2** | 97.0 | ✅ |
| ConvNeXt-Tiny | 14-class | 69.3 | 96.5 | ✅ |
| **Tri-Net v2 (Mpox screening)** | binary | **98.4** | **99.6** | ✅ |

Full table in [`docs/model_zoo.md`](docs/model_zoo.md); weights on the
[Releases](https://github.com/Sudharsanselvaraj/Synergistic-Deep-Learning-for-Monkeypox-Diagnosis/releases) page.
Then run inference directly: `trinet predict lesion.jpg`.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — models, fusion, pipeline
- [`docs/model_zoo.md`](docs/model_zoo.md) — pretrained models + weights
- [`docs/datasets.md`](docs/datasets.md) — data sources and the 14-class composition
- [`docs/benchmark.md`](docs/benchmark.md) — results, ablations, honest findings
- [`docs/reproducibility.md`](docs/reproducibility.md) — how to reproduce every number
- [`docs/roadmap.md`](docs/roadmap.md) — project plan and future work

The original published-paper scripts are preserved under [`archive/`](archive/).

## Citation

If you use this work, please cite it — see [`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE)
