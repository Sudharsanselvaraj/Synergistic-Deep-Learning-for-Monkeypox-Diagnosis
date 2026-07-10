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
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight as cw
from tensorflow.keras import Model, layers

from trinet.config import CFG, ensure_dirs  # noqa: E402
from trinet.evaluation.metrics import compute_scores  # noqa: E402
from trinet.models.backbones import feature_dim  # noqa: E402

PROJ_DIM = 256


def _load(bbs, split):
    y = np.load(CFG.features / f"y_{split}.npy")
    X = [np.load(CFG.features / f"X_{split}_{bb}.npy") for bb in bbs]
    return X, y


def _probs(bbs, split):
    return [np.load(CFG.model_dir / f"prob_{split}_{bb}.npy") for bb in bbs]


def _proj_tokens(inputs, d=PROJ_DIM):
    # project each backbone feature to a token, then stack to (B, N, d). We avoid Lambda
    # (which does not round-trip through the .keras format) by reshaping + concatenating.
    toks = []
    for i, inp in enumerate(inputs):
        t = layers.Dense(d, activation="relu", name=f"proj{i}")(inp)
        t = layers.LayerNormalization(name=f"ln{i}")(t)
        t = layers.Reshape((1, d), name=f"tok{i}")(t)  # (B,1,d)
        toks.append(t)
    return layers.Concatenate(axis=1, name="stack")(toks)  # (B,N,d)


def build_concat_mlp(dims, n_classes=CFG.n_classes) -> Model:
    # explicit layer names keep weights loadable across processes (auto names differ by
    # build order and silently break name/position-based weight loading)
    inputs = [layers.Input(shape=(dim,), name=f"feat{i}") for i, dim in enumerate(dims)]
    x = layers.Concatenate(name="cm_concat")(inputs)
    x = layers.Dense(512, activation="relu", name="cm_fc1")(x)
    x = layers.BatchNormalization(name="cm_bn")(x)
    x = layers.Dropout(CFG.dropout, name="cm_drop")(x)
    x = layers.Dense(128, activation="relu", name="cm_fc2")(x)
    out = layers.Dense(n_classes, activation="softmax", name="cm_out")(x)
    return Model(inputs, out, name="concat_mlp")


def build_gated_attention(dims, n_classes=CFG.n_classes, d=PROJ_DIM) -> Model:
    inputs = [layers.Input(shape=(dim,), name=f"feat{i}") for i, dim in enumerate(dims)]
    stacked = _proj_tokens(inputs, d)
    a = layers.Dense(128, activation="tanh", name="attn_h")(stacked)
    a = layers.Dense(1, name="attn_score")(a)
    w = layers.Softmax(axis=1, name="attn_weight")(a)  # (B,N,1), sums to 1 over N
    # weighted sum over the backbone axis via a batch dot (serializable, unlike Lambda):
    # Dot(axes=1) contracts the N axis -> (B,1,d), then flatten to (B,d)
    fused = layers.Dot(axes=1, name="fuse")([w, stacked])  # (B,1,d)
    fused = layers.Reshape((PROJ_DIM,), name="fuse_flat")(fused)  # (B,d)
    x = layers.Dense(128, activation="relu", name="ga_fc")(fused)
    x = layers.Dropout(CFG.dropout, name="ga_drop")(x)
    out = layers.Dense(n_classes, activation="softmax", name="ga_out")(x)
    return Model(inputs, out, name="gated_attention")


def build_transformer(dims, n_classes=CFG.n_classes, d=PROJ_DIM, heads=4) -> Model:
    inputs = [layers.Input(shape=(dim,), name=f"feat{i}") for i, dim in enumerate(dims)]
    tokens = _proj_tokens(inputs, d)  # (B,N,d)
    attn = layers.MultiHeadAttention(num_heads=heads, key_dim=d // heads, name="tf_mha")(
        tokens, tokens
    )
    x = layers.LayerNormalization(name="tf_ln1")(layers.Add(name="tf_add1")([tokens, attn]))
    ff = layers.Dense(d, activation="relu", name="tf_ff1")(x)
    ff = layers.Dense(d, name="tf_ff2")(ff)
    x = layers.LayerNormalization(name="tf_ln2")(layers.Add(name="tf_add2")([x, ff]))
    pooled = layers.GlobalAveragePooling1D(name="tf_pool")(x)
    pooled = layers.Dropout(CFG.dropout, name="tf_drop")(pooled)
    out = layers.Dense(n_classes, activation="softmax", name="tf_out")(pooled)
    return Model(inputs, out, name="transformer_fusion")


def _train(model, Xtr, ytr, Xva, yva, epochs):
    tf.keras.utils.set_random_seed(CFG.seed)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CFG.head_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    w = cw.compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
    model.fit(
        Xtr,
        ytr,
        validation_data=(Xva, yva),
        epochs=epochs,
        batch_size=CFG.batch_size,
        class_weight=dict(enumerate(w)),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.3, patience=4, min_lr=1e-6
            ),
        ],
        verbose=0,
    )
    return model


_BUILDERS = {
    "Concat+MLP": build_concat_mlp,
    "GatedAttention": build_gated_attention,
    "Transformer": build_transformer,
}


class _MeanEnsemble:
    """Equal-weight probability average over the per-backbone heads — a deployable 'model'.

    Takes the same list-of-feature-arrays input as the learned fusion models and returns the
    mean of each head's softmax, so it drops into the inference path unchanged.
    """

    def __init__(self, heads):
        self._heads = heads

    def predict(self, feats, verbose=0):
        probs = [h.predict(f, verbose=verbose) for h, f in zip(self._heads, feats)]
        return sum(probs) / len(probs)


def load_fusion(bbs=None):
    """Load the deployed ensemble. Reconstructs the exact model from disk (no rebuild-from-code),
    so predictions are identical across processes.

    - ``Mean``: loads the per-backbone heads and averages their probabilities.
    - learned fusion: loads the full ``fusion.keras`` model.
    """
    from tensorflow.keras.models import load_model

    meta = json.loads((CFG.model_dir / "fusion_meta.json").read_text())
    bbs = bbs or meta["backbones"]
    if meta["strategy"] == "Mean":
        heads = [load_model(CFG.model_dir / f"head_{bb}.keras") for bb in bbs]
        return _MeanEnsemble(heads), bbs
    model = load_model(CFG.model_dir / "fusion.keras", safe_mode=False, compile=False)
    return model, bbs


def _row(name, learnable, params, yte, prob):
    s = compute_scores(yte, prob.argmax(1), prob)
    return {"strategy": name, "learnable": learnable, "params": params, **s.as_pct()}


def _verify_crossprocess(prob_ref) -> float:
    """Reload the saved checkpoint in a FRESH subprocess and return max|Δlogit| vs prob_ref.

    A true cross-process check — the only reliable way to catch non-deterministic Keras
    deserialization that an in-process reload misses.
    """
    import subprocess
    import sys
    import tempfile

    out = Path(tempfile.mkdtemp()) / "reload.npy"
    script = (
        "import numpy as np, sys; import tensorflow as tf; "
        "tf.config.set_visible_devices([], 'GPU'); "
        "from trinet.models.fusion import load_fusion; from trinet.config import CFG; "
        "m, bbs = load_fusion(); "
        "X = [np.load(CFG.features / f'X_test_{b}.npy') for b in bbs]; "
        "np.save(sys.argv[1], m.predict(X, verbose=0))"
    )
    subprocess.run([sys.executable, "-c", script, str(out)], check=True, cwd=CFG.root)
    return float(np.abs(prob_ref - np.load(out)).max())


def main() -> None:
    import argparse

    # Train fusion on CPU so the saved checkpoint is numerically identical to how it is loaded
    # for inference (predict runs on CPU). A GPU(Metal)-trained model reloaded on CPU can
    # diverge enough to flip predictions. Fusion heads are tiny, so CPU training is fast.
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass  # GPU already initialised; op placement will still fall back gracefully
    # deterministic training so the reported fusion result reproduces exactly across runs
    tf.keras.utils.set_random_seed(CFG.seed)
    tf.config.experimental.enable_op_determinism()

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
    ptest = None
    try:
        ptest = _probs(bbs, "test")
        rows.append(_row("Mean", "no", 0, yte, sum(ptest) / len(ptest)))
        pw = json.loads((CFG.model_dir / "pso_weights.json").read_text())["weights"]
        if len(pw) == len(bbs):
            rows.append(_row("PSO", "global", len(bbs), yte, sum(w * p for w, p in zip(pw, ptest))))
    except FileNotFoundError:
        print("[i] skipping Mean/PSO (per-backbone probs not found for this set)")

    # learned feature-fusion strategies
    builders = [
        ("Concat+MLP", build_concat_mlp),
        ("GatedAttention", build_gated_attention),
        ("Transformer", build_transformer),
    ]
    models = {}
    for name, build in builders:
        model = _train(build(dims), Xtr, ytr, Xva, yva, epochs)
        prob = model.predict(Xte, verbose=0)
        rows.append(_row(name, "yes", model.count_params(), yte, prob))
        models[name] = model

    # Deploy the highest-accuracy ensemble that reproduces its logits in a FRESH PROCESS.
    # Candidates: the Mean ensemble (uses the saved per-backbone heads) and each learned fusion
    # model. In-process reload can look perfect while cross-process silently degrades, so we
    # verify in a real subprocess and skip any candidate that fails, keeping the best faithful one.
    acc_of = {r["strategy"]: r["accuracy"] for r in rows}
    candidates = []  # (accuracy, strategy, model_or_None, prob)
    heads_present = ptest is not None and all(
        (CFG.model_dir / f"head_{bb}.keras").exists() for bb in bbs
    )
    if heads_present:
        candidates.append((acc_of["Mean"], "Mean", None, sum(ptest) / len(ptest)))
    for name, model in models.items():
        candidates.append((acc_of[name], name, model, model.predict(Xte, verbose=0)))
    candidates.sort(key=lambda c: -c[0])

    deployed = None
    for acc, strategy, model, prob in candidates:
        if strategy == "Mean":
            (CFG.model_dir / "fusion.keras").unlink(missing_ok=True)  # Mean uses the heads
        else:
            model.save(CFG.model_dir / "fusion.keras")
        (CFG.model_dir / "fusion_meta.json").write_text(
            json.dumps({"strategy": strategy, "backbones": bbs, "test_accuracy": acc})
        )
        diff = _verify_crossprocess(prob)
        if diff < 1e-5:
            np.save(CFG.model_dir / "prob_test_fusion.npy", prob)
            deployed = strategy
            print(
                f"[checkpoint] deployed={deployed}  acc={acc:.2f}  "
                f"cross-process max|Δlogit|={diff:.2e}  [OK]"
            )
            break
        print(
            f"[checkpoint] {strategy} FAILED cross-process reload "
            f"(max|Δlogit|={diff:.2e}); trying next best"
        )
    if deployed is None:
        raise RuntimeError("No ensemble reproduced faithfully cross-process.")

    cols = [
        "strategy",
        "learnable",
        "params",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "kappa",
        "n",
    ]
    (CFG.tbl_dir / "fusion_ablation.csv").write_text(
        "\n".join([",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in rows])
    )
    print(f"\n{'strategy':<16}{'acc':>7}{'f1':>7}{'auc':>7}{'kappa':>7}  params")
    for r in rows:
        print(
            f"{r['strategy']:<16}{r['accuracy']:>7}{r['f1']:>7}{r['auc']:>7}"
            f"{r['kappa']:>7}  {r['params']}"
        )
    print(f"\n[✓] fusion ablation ({'+'.join(bbs)}) -> {CFG.tbl_dir / 'fusion_ablation.csv'}")


if __name__ == "__main__":
    main()
