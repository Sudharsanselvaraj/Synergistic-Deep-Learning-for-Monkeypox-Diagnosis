# Tri-Net MPOX — Rebuild Plan (make the paper real)

> **Goal.** Turn the *published* Tri-Net paper ("Tri-Net: Unified Deep Learning for Skin
> Lesion and Symptom-Based Monkeypox Detection") into genuine, reproducible work. The paper
> is published; the code behind it is a ~15% skeleton. We rebuild it properly, in two phases:
> **Phase 1 — faithfully reproduce every claim** (real code, honest numbers); **Phase 2 —
> surpass it** with stronger models. Runs locally on Apple **M5 / 16 GB** via `tensorflow-metal`.

---

## The gap we are closing (paper vs. original code)

| Paper claims | Original repo reality |
|---|---|
| **PSO-optimized weighted ensemble** `P_final = Σ wᵢPᵢ`, weights (0.42,0.32,0.26) | **No PSO.** `lesion.py` just *concatenates* backbone features. |
| Trained 14-class Tri-Net, 97.86% | **No trained image model saved anywhere.** |
| 5-fold CV, McNemar, Cohen's Kappa, ROC/AUC, Grad-CAM, ablation | **None exist in code.** All figures/tables unreproducible. |
| Symptom CNN 97.86% | Uses a leaking `"sum"` feature; number not credible. |

Section 4 of the paper even opens with a copy-paste artifact from an unrelated paper
("RL-MHWO-GWO for energy-aware routing"), and the two confusion matrices contradict each
other — evidence the write-up outran the work. We fix all of it.

---

## Local-Mac strategy (the key to training a triple ensemble on 16 GB)

1. **Freeze backbones, cache features once.** Run each ImageNet backbone over the dataset a
   single time, save the pooled feature vectors to `data/features/*.npy`. Training the
   classification heads then takes **seconds/epoch**, not hours — no GPU thrash.
2. **`tensorflow-metal`** for Metal GPU acceleration on the M5.
3. **Fine-tuning** (unfreezing last blocks) is optional / Phase 2, done selectively.
4. Grad-CAM needs the full model but runs inference-only on a handful of images — cheap.

---

## The real architecture

### Module A — Skin-lesion classifier "Tri-Net" (14-class)
- **Backbones:** EfficientNetB4, InceptionResNetV2, DenseNet201 (ImageNet-pretrained).
- **Head (each):** GAP → Dense(256) → BN → Dense(128) → Dropout(0.5) → Softmax(14).
- **Ensemble:** train the 3 heads → collect per-model validation probabilities → **PSO**
  finds weights (w₁+w₂+w₃=1) maximizing val accuracy → `P_final = Σ wᵢ·Pᵢ`.
- **Evaluation:** accuracy / precision / recall / F1 / AUC, 14-class + binary confusion
  matrices, ROC, **Grad-CAM**, **5-fold CV**, **McNemar** vs base models, **Cohen's Kappa**,
  full **ablation** table. Compare PSO vs grid/random/Bayesian weight search.

### Module B — Symptom classifier (binary Mpox / Non-Mpox)
- **Fix the leakage** (drop the `sum` feature). Train the CNN honestly.
- Baselines: Logistic Regression, Random Forest, XGBoost — reported side by side.
- **Report the real numbers**, whatever they are (this dataset is weak; honesty over 97.86%).

---

## Datasets needed (Kaggle)

The paper's 14 classes are a **composite** of a pox-diseases set + dermoscopy (skin-cancer)
classes + healthy. We assemble and document it honestly:
- **Pox/infectious:** Mpox, Chickenpox, Cowpox, Measles, HFMD, Healthy — from the Monkeypox
  Skin Image datasets (MSLD / MSID).
- **Dermoscopy:** Actinic Keratosis, BCC, Benign Keratosis, Dermatofibroma, Melanocytic
  Nevi, Melanoma, Squamous Cell Carcinoma, Vascular Lesions — from HAM10000 / ISIC.
- **Symptom:** the Kaggle monkeypox symptom CSV.

**Requires a Kaggle API key** (`~/.kaggle/kaggle.json`). See README → Data.

---

## Milestones

- [ ] **M0 — Environment & scaffold** (venv + tensorflow-metal verified on M5) ← *in progress*
- [ ] **M1 — Data pipeline** (download → build 14-class splits → clean symptom CSV)
- [ ] **M2 — Feature caching** (freeze backbones, dump `.npy` features)
- [ ] **M3 — Base models** (train 3 heads, save weights + per-model probs)
- [ ] **M4 — PSO ensemble** (weight optimization; vs grid/random/Bayesian)
- [ ] **M5 — Full evaluation** (metrics, confusion, ROC, Kappa, McNemar, CV, Grad-CAM, ablation)
- [ ] **M6 — Symptom module** (honest CNN + baselines)
- [ ] **M7 — Reproduce paper tables/figures** (Phase 1 done — every claim backed by code)
- [ ] **M8 — Phase 2: surpass** (stronger backbones / true multimodal fusion)
- [ ] **M9 — Web + Grad-CAM demo interface**

Progress and honest results live in `results/`. Nothing paper-facing is hand-typed.
