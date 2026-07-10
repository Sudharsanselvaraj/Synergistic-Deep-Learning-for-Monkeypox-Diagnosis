"""Phase 2B — fusion-strategy ablation on cached features.

Compares five ways to combine the backbones, from trivial to expressive, on the same test set:

  | strategy        | operates on   | learnable | note                                   |
  |-----------------|---------------|-----------|----------------------------------------|
  | Mean            | probabilities | no        | equal-weight average                   |
  | PSO             | probabilities | global    | one weight vector for all images       |
  | Concat + MLP    | features      | yes       | concatenate features -> MLP            |
  | Gated Attention | features      | yes       | per-image attention over backbones     |
  | Transformer     | features      | yes       | self-attention encoder over backbones  |

The last three fuse FEATURES (more expressive than combining probabilities). Everything trains
on the cached features in seconds/epoch.

Output: results/tables/fusion_ablation.csv  + best learned model -> results/models/fusion.keras

Usage:
    python -m src.models.fusion --backbones EfficientNetV2S ConvNeXtTiny DenseNet201
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.utils import class_weight as cw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CFG, ensure_dirs           # noqa: E402
from src.models.backbones import feature_dim   # noqa: E402
from src.eval.metrics import compute_scores     # noqa: E402

PROJ_DIM = 256


def _load(bbs, split):
    y = np.load(CFG.features / f"y_{split}.npy")
    X = [np.load(CFG.features / f"X_{split}_{bb}.npy") for bb in bbs]
    return X, y


def _probs(bbs, split):
    return [np.load(CFG.model_dir / f"prob_{split}_{bb}.npy") for bb in bbs]


def _proj_tokens(inputs, d=PROJ_DIM):
    toks = []
    for i, inp in enumerate(inputs):
        t = layers.Dense(d, activation="relu", name=f"proj{i}")(inp)
        t = layers.LayerNormalization(name=f"ln{i}")(t)
        toks.append(t)
    return layers.Lambda(lambda xs: tf.stack(xs, axis=1), name="stack")(toks)  # (B,N,d)


def build_concat_mlp(dims, n_classes=CFG.n_classes) -> Model:
    inputs = [layers.Input(shape=(dim,), name=f"feat{i}") for i, dim in enumerate(dims)]
    x = layers.Concatenate()(inputs)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CFG.dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return Model(inputs, out, name="concat_mlp")


def build_gated_attention(dims, n_classes=CFG.n_classes, d=PROJ_DIM) -> Model:
    inputs = [layers.Input(shape=(dim,), name=f"feat{i}") for i, dim in enumerate(dims)]
    stacked = _proj_tokens(inputs, d)
    a = layers.Dense(128, activation="tanh", name="attn_h")(stacked)
    a = layers.Dense(1, name="attn_score")(a)
    w = layers.Softmax(axis=1, name="attn_weight")(a)
    fused = layers.Lambda(lambda z: tf.reduce_sum(z[0] * z[1], axis=1), name="fuse")([w, stacked])
    x = layers.Dense(128, activation="relu")(fused)
    x = layers.Dropout(CFG.dropout)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return Model(inputs, out, name="gated_attention")


def build_transformer(dims, n_classes=CFG.n_classes, d=PROJ_DIM, heads=4) -> Model:
    inputs = [layers.Input(shape=(dim,), name=f"feat{i}") for i, dim in enumerate(dims)]
    tokens = _proj_tokens(inputs, d)                                     # (B,N,d)
    attn = layers.MultiHeadAttention(num_heads=heads, key_dim=d // heads)(tokens, tokens)
    x = layers.LayerNormalization()(tokens + attn)
    ff = layers.Dense(d, activation="relu")(x)
    x = layers.LayerNormalization()(x + layers.Dense(d)(ff))
    pooled = layers.GlobalAveragePooling1D()(x)
    pooled = layers.Dropout(CFG.dropout)(pooled)
    out = layers.Dense(n_classes, activation="softmax")(pooled)
    return Model(inputs, out, name="transformer_fusion")


def _train(model, Xtr, ytr, Xva, yva, epochs):
    tf.keras.utils.set_random_seed(CFG.seed)
    model.compile(optimizer=tf.keras.optimizers.Adam(CFG.head_lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    w = cw.compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=CFG.batch_size,
              class_weight=dict(enumerate(w)),
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                                          restore_best_weights=True),
                         tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                                              patience=4, min_lr=1e-6)],
              verbose=0)
    return model


def _row(name, learnable, params, yte, prob):
    s = compute_scores(yte, prob.argmax(1), prob)
    return {"strategy": name, "learnable": learnable, "params": params, **s.as_pct()}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="*", default=CFG.backbones)
    ap.add_argument("--epochs", type=int, default=CFG.head_epochs)
    a = ap.parse_args()
    bbs, epochs = a.backbones, a.epochs
    ensure_dirs()

    Xtr, ytr = _load(bbs, "train")
    Xva, yva = _load(bbs, "val")
    Xte, yte = _load(bbs, "test")
    dims = [feature_dim(bb) for bb in bbs]
    rows = []

    # analytic baselines on probabilities (need per-backbone probs to exist)
    try:
        ptest = _probs(bbs, "test")
        rows.append(_row("Mean", "no", 0, yte, sum(ptest) / len(ptest)))
        pw = json.loads((CFG.model_dir / "pso_weights.json").read_text())["weights"]
        if len(pw) == len(bbs):
            rows.append(_row("PSO", "global", len(bbs), yte,
                             sum(w * p for w, p in zip(pw, ptest))))
    except FileNotFoundError:
        print("[i] skipping Mean/PSO (per-backbone probs not found for this set)")

    # learned feature-fusion strategies
    builders = [("Concat+MLP", build_concat_mlp), ("GatedAttention", build_gated_attention),
                ("Transformer", build_transformer)]
    best_acc, best_model = -1.0, None
    for name, build in builders:
        model = _train(build(dims), Xtr, ytr, Xva, yva, epochs)
        prob = model.predict(Xte, verbose=0)
        r = _row(name, "yes", model.count_params(), yte, prob)
        rows.append(r)
        if r["accuracy"] > best_acc:
            best_acc, best_model = r["accuracy"], model

    if best_model is not None:
        best_model.save(CFG.model_dir / "fusion.keras")
        np.save(CFG.model_dir / "prob_test_fusion.npy", best_model.predict(Xte, verbose=0))

    cols = ["strategy", "learnable", "params", "accuracy", "precision", "recall", "f1", "auc",
            "kappa", "n"]
    (CFG.tbl_dir / "fusion_ablation.csv").write_text(
        "\n".join([",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in rows]))
    print(f"\n{'strategy':<16}{'acc':>7}{'f1':>7}{'auc':>7}{'kappa':>7}  params")
    for r in rows:
        print(f"{r['strategy']:<16}{r['accuracy']:>7}{r['f1']:>7}{r['auc']:>7}"
              f"{r['kappa']:>7}  {r['params']}")
    print(f"\n[✓] fusion ablation ({'+'.join(bbs)}) -> {CFG.tbl_dir/'fusion_ablation.csv'}")


if __name__ == "__main__":
    main()
