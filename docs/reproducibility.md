# Reproducibility

Every reported number is produced by runnable code from a seeded, leakage-free split — nothing
is hand-typed. Generated artifacts land in `outputs/` (git-ignored); regenerate them with the
CLI below.

## Full run

```bash
pip install -e ".[metal]"        # ".[metal]" only on Apple Silicon
trinet download                  # datasets  (needs ~/.kaggle credentials)
trinet prepare                   # 14-class split + symptom cleaning
trinet features                  # cache frozen features (Phase-1 backbones)
trinet features --backbones ConvNeXtTiny EfficientNetV2S --cpu   # modern backbones (see note)
trinet train --backbones ConvNeXtTiny InceptionResNetV2 DenseNet201
trinet ensemble                  # PSO + baselines
trinet fusion                    # fusion ablation (Concat-MLP champion)
trinet evaluate --champion       # figures + tables from the best model
trinet cross-val && trinet diversity && trinet gradcam
trinet symptom                   # symptom module + baselines + ablation
```

## Determinism

- Global seed `42` (`CFG.seed`) across NumPy / TensorFlow / scikit-learn.
- The split is stratified and seeded; the same images always land in the same partition.
- Metrics use only the held-out test set; cross-validation uses original (un-augmented)
  features so augmented copies never straddle folds.

## Apple Silicon (Metal) notes

`tensorflow-metal` accelerates the Phase-1 backbones, but lacks ops for some modern
architectures: **EfficientNetV2-S hangs** and **ConvNeXt errors** on the Metal GPU. Extract
those on CPU with `trinet features --cpu`. Grad-CAM also runs on CPU (Metal cannot compile its
gradient graph). These are `tensorflow-metal` limitations, not framework bugs.

## Environment

Reference environment: Apple M-series, 16 GB, Python 3.11, TensorFlow 2.16.2. Exact pins are
in `pyproject.toml` / `requirements.txt`.
