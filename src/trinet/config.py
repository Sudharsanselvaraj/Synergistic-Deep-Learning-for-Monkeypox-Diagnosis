"""Central configuration for the Tri-Net MPOX project.

Everything paths, class names, and hyperparameters live here so scripts stay thin and
the whole pipeline is reproducible. Import as `from trinet.config import CFG`.
"""

from __future__ import annotations

from pathlib import Path

# repo root is three levels up from src/trinet/config.py
ROOT = Path(__file__).resolve().parents[2]


class CFG:
    # ---- paths ----
    root = ROOT
    data_raw = ROOT / "data" / "raw"
    data_proc = ROOT / "data" / "processed"  # train/ val/ test/ class-folders
    features = ROOT / "data" / "features"  # cached backbone features (.npy)
    outputs = ROOT / "outputs"  # generated artifacts (git-ignored)
    fig_dir = outputs / "figures"
    tbl_dir = outputs / "reports"
    model_dir = outputs / "checkpoints"
    pred_dir = outputs / "predictions"
    log_dir = outputs / "logs"
    results = outputs  # backwards-compat alias

    # ---- image lesion module ----
    # 14 classes, matching Fig. 3 of the paper. Order is fixed for reproducibility.
    lesion_classes = [
        "Mpox",
        "Chickenpox",
        "Cowpox",
        "Measles",
        "HFMD",
        "Healthy",  # infectious / pox
        "Melanoma",
        "MelanocyticNevi",
        "BasalCellCarcinoma",  # dermoscopy (skin cancer)
        "ActinicKeratosis",
        "BenignKeratosis",
        "Dermatofibroma",
        "VascularLesion",
        "SquamousCellCarcinoma",
    ]
    n_classes = len(lesion_classes)
    # The one positive class for the binary Mpox-vs-rest collapse used in Fig. 6.
    mpox_class = "Mpox"

    img_size = (224, 224)
    img_shape = (224, 224, 3)
    batch_size = 32
    seed = 42

    # split ratios (paper: 80/10/10)
    train_frac = 0.80
    val_frac = 0.10
    test_frac = 0.10

    # base backbones (name -> keras application + its preprocess module)
    # Phase-2 "Tri-Net v2": ConvNeXt-Tiny (69.3%, best single) replaces the weakest member
    # EfficientNetB4 (58.4%). Phase-1 paper set was EfficientNetB4/InceptionResNetV2/DenseNet201.
    backbones = ["ConvNeXtTiny", "InceptionResNetV2", "DenseNet201"]

    # feature caching: train split gets `aug_multiplier` variants (1 original + K-1 augmented)
    # through the frozen backbone; val/test are cached clean (no augmentation).
    aug_multiplier = 5

    # head training
    head_epochs = 40
    head_lr = 1e-3
    dropout = 0.5
    early_stop_patience = 8

    # fine-tuning (phase 2)
    finetune_epochs = 15  # early stopping keeps best; small data overfits fast
    finetune_lr = 1e-5
    finetune_unfreeze = 60  # last N backbone layers unfrozen

    # ---- PSO ensemble ----
    pso_particles = 20
    pso_iters = 100
    pso_w = 0.7  # inertia
    pso_c1 = 1.5  # cognitive
    pso_c2 = 1.5  # social

    # ---- symptom module ----
    symptom_dir = data_raw / "symptom"  # raw Kaggle download; the CSV inside is discovered by glob
    symptom_target = "MonkeyPox"
    symptom_drop = ["Patient_ID", "sum"]  # 'sum' was a leakage feature in the original code
    symptom_epochs = 70
    symptom_batch = 32


def ensure_dirs() -> None:
    for p in [
        CFG.data_raw,
        CFG.data_proc,
        CFG.features,
        CFG.outputs,
        CFG.fig_dir,
        CFG.tbl_dir,
        CFG.model_dir,
        CFG.pred_dir,
        CFG.log_dir,
    ]:
        p.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print(f"Project root: {CFG.root}")
    print(f"{CFG.n_classes} lesion classes: {CFG.lesion_classes}")
