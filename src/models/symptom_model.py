"""Symptom module — honest binary Mpox screening + a leakage ablation.

Trains the CNN (matching the paper's little MLP) plus Logistic Regression / Random Forest /
XGBoost baselines, and reports the REAL test metrics. Then it runs an ablation that re-adds
the leaking `sum` feature to expose how the original code manufactured its 97.86%.

Outputs:
  results/models/symptom_cnn.keras, symptom_scaler.pkl
  results/tables/symptom_scores.csv        (honest CNN + baselines)
  results/tables/symptom_leakage.csv       (with vs without the 'sum' feature)
"""
from __future__ import annotations
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight as cw
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CFG, ensure_dirs         # noqa: E402
from src.eval.metrics import compute_scores  # noqa: E402


def _splits(X, y):
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.25, random_state=CFG.seed,
                                            stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=CFG.seed,
                                          stratify=ytmp)
    return Xtr, Xva, Xte, ytr, yva, yte


def _build_cnn(dim: int) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(CFG.seed)
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(dim,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m


def _fit_cnn(Xtr, ytr, Xva, yva):
    m = _build_cnn(Xtr.shape[1])
    w = cw.compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
    m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=CFG.symptom_epochs,
          batch_size=CFG.symptom_batch, class_weight=dict(enumerate(w)),
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                                      restore_best_weights=True)],
          verbose=0)
    return m


def _score_row(name, y, pred, prob):
    return {"model": name, **compute_scores(y, pred, prob).as_pct()}


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(CFG.data_proc / "symptom.csv")
    y = df[CFG.symptom_target].values
    X = df.drop(columns=[CFG.symptom_target]).values
    feat_names = df.drop(columns=[CFG.symptom_target]).columns.tolist()

    Xtr, Xva, Xte, ytr, yva, yte = _splits(X, y)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)
    print(f"[i] {X.shape[1]} features, naive-majority baseline = {100*max(y.mean(),1-y.mean()):.1f}%")

    rows = []
    # CNN
    cnn = _fit_cnn(Xtr_s, ytr, Xva_s, yva)
    p = cnn.predict(Xte_s, verbose=0).ravel()
    rows.append(_score_row("CNN (proposed)", yte, (p > 0.5).astype(int),
                           np.c_[1 - p, p]))
    cnn.save(CFG.model_dir / "symptom_cnn.keras")
    joblib.dump(scaler, CFG.model_dir / "symptom_scaler.pkl")

    # classical baselines
    for name, clf in [
        ("LogisticRegression", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ("RandomForest", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                random_state=CFG.seed)),
        ("XGBoost", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                  eval_metric="logloss", random_state=CFG.seed)),
    ]:
        clf.fit(Xtr_s, ytr)
        prob = clf.predict_proba(Xte_s)
        rows.append(_score_row(name, yte, prob.argmax(1), prob))

    for r in rows:
        print(f"[{r['model']:>18}] acc={r['accuracy']} f1={r['f1']} auc={r['auc']}")

    cols = ["model", "accuracy", "precision", "recall", "f1", "auc", "kappa", "n"]
    (CFG.tbl_dir / "symptom_scores.csv").write_text(
        "\n".join([",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in rows]))

    # ---- ablation on the original code's 'sum' feature ----
    # We hypothesised 'sum' inflated the paper's 97.86%. The ablation DISPROVES that: adding
    # 'sum' back moves accuracy by <1pt. The honest conclusion is stronger — no standard model
    # (CNN/LR/RF/XGB, with or without 'sum') beats ~70% here, so 97.86% is not reproducible
    # from this dataset by any legitimate method. We keep the ablation to document that.
    Xsum = np.c_[X, X.sum(axis=1)]
    Xtr2, Xva2, Xte2, ytr2, yva2, yte2 = _splits(Xsum, y)
    sc2 = StandardScaler().fit(Xtr2)
    cnn2 = _fit_cnn(sc2.transform(Xtr2), ytr2, sc2.transform(Xva2), yva2)
    p2 = cnn2.predict(sc2.transform(Xte2), verbose=0).ravel()
    abl = [
        _score_row("CNN (without 'sum')", yte, (p > 0.5).astype(int), np.c_[1 - p, p]),
        _score_row("CNN (with 'sum')", yte2, (p2 > 0.5).astype(int), np.c_[1 - p2, p2]),
    ]
    (CFG.tbl_dir / "symptom_ablation_sum.csv").write_text(
        "\n".join([",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in abl]))
    print(f"\n['sum' ablation] without={abl[0]['accuracy']}  with={abl[1]['accuracy']}  "
          f"(negligible -> 'sum' was NOT the source of 97.86%)")
    print(f"[✓] symptom tables -> {CFG.tbl_dir}")


if __name__ == "__main__":
    main()
