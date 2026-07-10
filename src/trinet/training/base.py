"""Train the three classification heads on cached features, save heads + per-model probs.

Reads data/features/*.npy (from extract_features), trains one head per backbone with class
weights (the dataset is imbalanced), and writes:
  results/models/head_<bb>.keras
  results/models/prob_val_<bb>.npy, prob_test_<bb>.npy   (softmax outputs — PSO ensemble input)
  results/tables/base_scores.csv                          (honest per-model test metrics)

Because heads train on cached feature vectors, each fit is seconds/epoch even for 40 epochs.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight as cw

from trinet.config import CFG, ensure_dirs                         # noqa: E402
from trinet.models.backbones import build_head, feature_dim    # noqa: E402
from trinet.evaluation.metrics import compute_scores                 # noqa: E402


def _load(bb: str):
    F = CFG.features
    return (
        np.load(F / f"X_train_{bb}.npy"), np.load(F / "y_train.npy"),
        np.load(F / f"X_val_{bb}.npy"),   np.load(F / "y_val.npy"),
        np.load(F / f"X_test_{bb}.npy"),  np.load(F / "y_test.npy"),
    )


def _class_weights(y: np.ndarray) -> dict:
    classes = np.unique(y)
    w = cw.compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(wi) for c, wi in zip(classes, w)}


def train_one(bb: str) -> dict:
    Xtr, ytr, Xva, yva, Xte, yte = _load(bb)
    tf.keras.utils.set_random_seed(CFG.seed)
    head = build_head(feature_dim(bb))
    head.compile(optimizer=tf.keras.optimizers.Adam(CFG.head_lr),
                 loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    ckpt = CFG.model_dir / f"head_{bb}.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=CFG.early_stop_patience,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=4,
                                             min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_accuracy",
                                           save_best_only=True),
    ]
    head.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=CFG.head_epochs,
             batch_size=CFG.batch_size, class_weight=_class_weights(ytr),
             callbacks=callbacks, verbose=2)

    prob_val = head.predict(Xva, batch_size=CFG.batch_size, verbose=0)
    prob_test = head.predict(Xte, batch_size=CFG.batch_size, verbose=0)
    np.save(CFG.model_dir / f"prob_val_{bb}.npy", prob_val)
    np.save(CFG.model_dir / f"prob_test_{bb}.npy", prob_test)

    s = compute_scores(yte, prob_test.argmax(1), prob_test)
    print(f"[✓] {bb} test: acc={100*s.accuracy:.2f} f1={100*s.f1:.2f} auc={100*s.auc:.2f}")
    return {"backbone": bb, **s.as_pct()}


def _score_saved(bb: str) -> dict:
    """Score an already-trained backbone from its saved test probabilities."""
    yte = np.load(CFG.features / "y_test.npy")
    prob = np.load(CFG.model_dir / f"prob_test_{bb}.npy")
    return {"backbone": bb, **compute_scores(yte, prob.argmax(1), prob).as_pct()}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="*", default=CFG.backbones)
    args = ap.parse_args()
    ensure_dirs()
    for bb in args.backbones:
        train_one(bb)

    # rebuild the table from ALL trained (frozen) backbones so a subset run never clobbers others
    trained = sorted({p.name[len("prob_test_"):-len(".npy")]
                      for p in CFG.model_dir.glob("prob_test_*.npy")
                      if not p.name.startswith("prob_test_ft_")})
    rows = [_score_saved(bb) for bb in trained]
    cols = ["backbone", "accuracy", "precision", "recall", "f1", "auc", "kappa", "n"]
    lines = [",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in rows]
    (CFG.tbl_dir / "base_scores.csv").write_text("\n".join(lines))
    (CFG.tbl_dir / "base_scores.json").write_text(json.dumps(rows, indent=2))
    print(f"\n[✓] base model scores ({len(rows)} backbones) -> {CFG.tbl_dir/'base_scores.csv'}")


if __name__ == "__main__":
    main()
