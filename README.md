# Tri-Net MPOX — Monkeypox Detection (rebuild)

Real, reproducible implementation of the paper *"Tri-Net: Unified Deep Learning for Skin
Lesion and Symptom-Based Monkeypox Detection."* The published paper's code was a skeleton;
this repo rebuilds the actual system — a **PSO-optimized 3-backbone ensemble** for 14-class
skin-lesion classification plus an honest **symptom classifier** — and reproduces every
figure and table from runnable code. See [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for the full
plan and the paper-vs-code gap we're closing.

Runs locally on **Apple Silicon (M-series)** via `tensorflow-metal`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python config.py          # create data/ + results/ dirs, sanity-check
```

Verify the Metal GPU is picked up:
```bash
.venv/bin/python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Data

Needs a **Kaggle API token**: kaggle.com → Settings → *Create New API Token* → save as
`~/.kaggle/kaggle.json`, then `chmod 600 ~/.kaggle/kaggle.json`.

```bash
.venv/bin/python -m src.data.download          # pulls MSLD v2 + HAM10000 + ISIC 2019 + symptom CSV
.venv/bin/python -m src.data.prepare_lesion    # build the 14-class train/val/test split
.venv/bin/python -m src.data.prepare_symptom   # clean symptom CSV (fixes the 'sum' leakage)
```

The 14-class lesion set is a documented composite (pox diseases + dermoscopy + healthy) — see
`PROJECT_PLAN.md → Datasets`.

## Pipeline (once data is ready)

```bash
.venv/bin/python -m src.models.extract_features   # cache frozen-backbone features (.npy)
.venv/bin/python -m src.models.train_base         # train the 3 classification heads
.venv/bin/python -m src.models.pso_ensemble       # PSO weight optimization
.venv/bin/python -m src.eval.run_all              # metrics, confusion, ROC, Kappa, Grad-CAM, ...
.venv/bin/python -m src.models.symptom_model      # symptom CNN + LR/RF/XGBoost baselines
```

Outputs land in `results/` (figures, tables, models). Nothing paper-facing is hand-typed.

## Layout

```
config.py          central config (paths, 14 class names, hyperparameters)
src/data/          download + prepare (lesion split, symptom cleaning)
src/models/        backbones, feature caching, base training, PSO ensemble, symptom model
src/eval/          metrics, confusion, ROC, Grad-CAM, cross-val, McNemar, Kappa, ablation
src/interface/     web / demo (Phase 2)
legacy/            the original published-paper scripts, preserved for reference
results/           generated figures, tables, trained models
```
