"""Phase 2 — fine-tune the backbones end-to-end (the real accuracy lever).

The base models used FROZEN ImageNet features. Here we unfreeze the top `finetune_unfreeze`
layers of each backbone and train end-to-end at a low learning rate, warm-starting the head
from the cached-feature training. This adapts the convolutional features to skin lesions
instead of natural images — the main reason the frozen base models plateaued at 58-68%.

Per backbone we save the fine-tuned model + val/test probabilities (suffix `_ft`) so the PSO
ensemble and evaluation can be re-run on the improved models:
  results/models/ft_<bb>.keras, prob_val_ft_<bb>.npy, prob_test_ft_<bb>.npy

Usage:
    python -m src.models.finetune                 # all backbones
    python -m src.models.finetune --only DenseNet201 --epochs 10
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.utils import class_weight as cw

from trinet.config import CFG, ensure_dirs               # noqa: E402
from trinet.models.backbones import build_full_model  # noqa: E402
from trinet.evaluation.metrics import compute_scores        # noqa: E402

_AUG = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical", seed=CFG.seed),
    layers.RandomRotation(0.15, seed=CFG.seed),
    layers.RandomZoom(0.15, seed=CFG.seed),
    layers.RandomContrast(0.15, seed=CFG.seed),
], name="augment")


def _ds(split: str, augment: bool):
    ds = image_dataset_from_directory(
        CFG.data_proc / split, labels="inferred", label_mode="int",
        class_names=CFG.lesion_classes, image_size=CFG.img_size,
        batch_size=CFG.batch_size, shuffle=(split == "train"), seed=CFG.seed)
    if augment:
        ds = ds.map(lambda x, y: (_AUG(x, training=True), y), tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def _labels(split: str) -> np.ndarray:
    ds = image_dataset_from_directory(
        CFG.data_proc / split, labels="inferred", label_mode="int",
        class_names=CFG.lesion_classes, image_size=CFG.img_size,
        batch_size=CFG.batch_size, shuffle=False)
    return np.concatenate([y.numpy() for _, y in ds])


def _warm_start_head(full, bb):
    head = load_model(CFG.model_dir / f"head_{bb}.keras")
    hw = [l.get_weights() for l in head.layers if l.get_weights()]
    top = [l for l in full.layers if l.get_weights()][-len(hw):]
    for layer, w in zip(top, hw):
        try:
            layer.set_weights(w)
        except ValueError:
            pass  # shape mismatch (shouldn't happen) — leave randomly initialised


def finetune_one(bb: str, epochs: int) -> dict:
    tf.keras.utils.set_random_seed(CFG.seed)
    train_ds, val_ds = _ds("train", True), _ds("val", False)
    y_val, y_test = _labels("val"), _labels("test")

    full = build_full_model(bb, finetune_unfreeze=CFG.finetune_unfreeze)
    _warm_start_head(full, bb)
    full.compile(optimizer=tf.keras.optimizers.Adam(CFG.finetune_lr),
                 loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    ytr = _labels("train")
    w = cw.compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
    full.fit(train_ds, validation_data=val_ds, epochs=epochs,
             class_weight=dict(enumerate(w)),
             callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                                         restore_best_weights=True),
                        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                                             patience=2, min_lr=1e-7)],
             verbose=2)

    prob_val = full.predict(_ds("val", False), verbose=0)
    prob_test = full.predict(_ds("test", False), verbose=0)
    np.save(CFG.model_dir / f"prob_val_ft_{bb}.npy", prob_val)
    np.save(CFG.model_dir / f"prob_test_ft_{bb}.npy", prob_test)
    full.save(CFG.model_dir / f"ft_{bb}.keras")

    s = compute_scores(y_test, prob_test.argmax(1), prob_test)
    print(f"[✓] {bb} fine-tuned test: acc={100*s.accuracy:.2f} f1={100*s.f1:.2f} "
          f"auc={100*s.auc:.2f}")
    return {"backbone": bb, **s.as_pct()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=CFG.backbones)
    ap.add_argument("--epochs", type=int, default=CFG.finetune_epochs)
    args = ap.parse_args()
    ensure_dirs()
    rows = [finetune_one(bb, args.epochs) for bb in (args.only or CFG.backbones)]
    cols = ["backbone", "accuracy", "precision", "recall", "f1", "auc", "kappa", "n"]
    (CFG.tbl_dir / "finetune_scores.csv").write_text(
        "\n".join([",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in rows]))
    print(f"\n[✓] fine-tune scores -> {CFG.tbl_dir/'finetune_scores.csv'}")


if __name__ == "__main__":
    main()
