"""5-fold cross-validation for the base models + Tri-Net ensemble (paper Table 8, honest).

Done on ORIGINAL image features only (no augmentation) to keep folds leakage-free: augmented
copies of the same image must never straddle train/test. We recover the 3032 original feature
vectors (the k=0 slice of the cached train set + val + test) and run StratifiedKFold(5). Each
fold trains a fresh head per backbone and combines them with equal weights.

Output: results/tables/cross_val.csv  (per-fold + mean±std for each model)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight as cw

from trinet.config import CFG, ensure_dirs                       # noqa: E402
from trinet.models.backbones import build_head, feature_dim  # noqa: E402
from trinet.evaluation.metrics import compute_scores               # noqa: E402

CV_EPOCHS = 25


def _original_features():
    """Return {bb: X_orig (3032, d)} and y_orig, from un-augmented features."""
    y_val = np.load(CFG.features / "y_val.npy")
    y_test = np.load(CFG.features / "y_test.npy")
    y_train_full = np.load(CFG.features / "y_train.npy")
    n_orig = len(y_train_full) // CFG.aug_multiplier          # original train count (2426)
    y_train = y_train_full[:n_orig]
    y = np.concatenate([y_train, y_val, y_test])
    X = {}
    for bb in CFG.backbones:
        Xtr = np.load(CFG.features / f"X_train_{bb}.npy")[:n_orig]
        Xva = np.load(CFG.features / f"X_val_{bb}.npy")
        Xte = np.load(CFG.features / f"X_test_{bb}.npy")
        X[bb] = np.concatenate([Xtr, Xva, Xte], axis=0)
    return X, y


def _train_head(bb, Xtr, ytr, Xva, yva):
    tf.keras.utils.set_random_seed(CFG.seed)
    head = build_head(feature_dim(bb))
    head.compile(optimizer=tf.keras.optimizers.Adam(CFG.head_lr),
                 loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    w = cw.compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
    head.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=CV_EPOCHS,
             batch_size=CFG.batch_size, class_weight=dict(enumerate(w)),
             callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                                         restore_best_weights=True)],
             verbose=0)
    return head


def main() -> None:
    ensure_dirs()
    X, y = _original_features()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.seed)
    per_model = {m: [] for m in CFG.backbones + ["Tri-Net"]}
    for fold, (tr, te) in enumerate(skf.split(X[CFG.backbones[0]], y), start=1):
        probs_te = {}
        for bb in CFG.backbones:
            head = _train_head(bb, X[bb][tr], y[tr], X[bb][te], y[te])
            p = head.predict(X[bb][te], batch_size=CFG.batch_size, verbose=0)
            probs_te[bb] = p
            acc = compute_scores(y[te], p.argmax(1), p).accuracy
            per_model[bb].append(acc)
        ens = sum(probs_te[bb] for bb in CFG.backbones) / len(CFG.backbones)
        ens_acc = compute_scores(y[te], ens.argmax(1), ens).accuracy
        per_model["Tri-Net"].append(ens_acc)
        print(f"[fold {fold}] " + "  ".join(
            f"{m}={100*per_model[m][-1]:.2f}" for m in CFG.backbones + ["Tri-Net"]))

    lines = ["model,fold1,fold2,fold3,fold4,fold5,mean,std"]
    for m in CFG.backbones + ["Tri-Net"]:
        v = np.array(per_model[m]) * 100
        lines.append(f"{m}," + ",".join(f"{x:.2f}" for x in v) +
                     f",{v.mean():.2f},{v.std():.2f}")
    (CFG.tbl_dir / "cross_val.csv").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\n[✓] cross-val -> {CFG.tbl_dir/'cross_val.csv'}")


if __name__ == "__main__":
    main()
