# Architecture

Tri-Net v2 is organized as an installable Python package (`src/trinet`) with a thin CLI.

## Pipeline

```
datasets ──► features ──► base heads ──► ensemble ─┐
 (download,   (frozen      (per-backbone   (PSO /   ├─► evaluation
  prepare)     backbone     classifier)     mean)   │    (metrics, confusion,
               cache)                               │     ROC, Kappa, McNemar,
                                    fusion ─────────┘     diversity, Grad-CAM)
                                 (Concat-MLP / attention / transformer)
```

## Backbones (`trinet.models.backbones`)

ImageNet-pretrained feature extractors, each with its **own** preprocessing (getting this
wrong silently wrecks accuracy):

| Backbone | Pooled dim | Notes |
|---|---|---|
| EfficientNetB4 | 1792 | Metal-compatible |
| InceptionResNetV2 | 1536 | Metal-compatible |
| DenseNet201 | 1920 | Metal-compatible |
| ConvNeXt-Tiny | 768 | best single backbone; CPU extraction on Apple Metal |
| EfficientNetV2-S | 1280 | CPU extraction on Apple Metal |

Backbones are used as **frozen** feature extractors; features are cached to disk so head
training is seconds/epoch. Fine-tuning is available (`trinet finetune`) but did not help at
this data scale — see [benchmark.md](benchmark.md).

## Fusion (`trinet.models.fusion`)

Five strategies, from trivial to expressive:

- **Mean** / **PSO** — combine per-model *probabilities* (not learnable / global weights)
- **Concat + MLP** — concatenate *features* → MLP (the champion)
- **Gated Attention** — per-image attention over backbones
- **Transformer** — self-attention encoder over backbone tokens

Feature fusion beats probability fusion; the highest-capacity variants overfit on this
dataset size.

## Evaluation (`trinet.evaluation`, `trinet.explainability`)

`metrics` (accuracy/precision/recall/F1/AUC/Kappa/McNemar/bootstrap), `figures` (confusion,
ROC, Kappa), `cross_val` (5-fold), `diversity` (disagreement/Q/rho), `efficiency`
(params/FLOPs/latency), and `gradcam` (attention heatmaps, CPU).
