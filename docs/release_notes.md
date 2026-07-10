# Tri-Net v2 — Initial Open-Source Release (v0.9.0)

The first public release of **Tri-Net v2**, the official implementation of the Tri-Net study,
re-engineered as an installable, tested, and reproducible framework and modernised end to end.

## Highlights

- **Reproducible 14-class Mpox benchmark** on a strictly leakage-free 80/10/10 split.
- **Modern backbones** — ConvNeXt-Tiny and EfficientNetV2-S alongside the original three.
- **Learned feature-fusion ablation** (Mean, PSO, Concat-MLP, Gated-Attention, Transformer).
- **Dual-task evaluation** — 14-class diagnosis (77.2%) and binary Mpox screening (98.4% / 99.6% AUC).
- **Full pipeline** — installable `trinet` package, unified CLI, unit tests, CI, Docker, model zoo.
- **Checkpoint integrity** — the deployed model is verified to reproduce identical logits in a
  fresh process before it is saved.

## Pretrained weights

`trinet-v0.9.0-weights.zip` contains the deployed ensemble checkpoint. To run inference:

```bash
pip install mpox-trinet          # or `pip install -e .` from a clone
# download and unzip trinet-v0.9.0-weights.zip into outputs/checkpoints/
trinet predict lesion.jpg
```

The package installs as **`mpox-trinet`** and imports as **`trinet`** (with the `trinet` CLI).

The attached checkpoint's exact accuracy is recorded in its `fusion_meta.json`; the full
benchmark (best configuration 77.2%) is in [`docs/benchmark.md`](benchmark.md). Backbone weights
download automatically from ImageNet on first use.

## Status

This is an early public release (0.9.x). The research pipeline and results are stable and
reproducible; APIs may still evolve before a 1.0. Not a medical device — see the README and
`SECURITY.md`.
