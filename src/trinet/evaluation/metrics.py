"""Evaluation metrics shared across the project — computed honestly, no hand-typed numbers.

Everything downstream (tables, figures, the paper) reads from these functions so a single
source of truth produces every reported number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class Scores:
    accuracy: float
    precision: float  # macro
    recall: float  # macro
    f1: float  # macro
    auc: float  # macro one-vs-rest (nan if undefined)
    kappa: float
    n: int

    def as_pct(self) -> dict:
        d = asdict(self)
        for k in ("accuracy", "precision", "recall", "f1", "auc", "kappa"):
            d[k] = round(100 * d[k], 2) if d[k] == d[k] else float("nan")  # keep nan
        return d


def compute_scores(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None
) -> Scores:
    """Macro-averaged metrics. y_prob (N, C) enables AUC; pass None to skip."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    auc = float("nan")
    if y_prob is not None:
        try:
            classes = np.unique(y_true)
            if len(classes) == 2:
                pos = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] == 2 else y_prob.ravel()
                auc = roc_auc_score(y_true, pos)
            else:
                auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except ValueError:
            auc = float("nan")
    return Scores(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, average="macro", zero_division=0),
        recall=recall_score(y_true, y_pred, average="macro", zero_division=0),
        f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        auc=auc,
        kappa=cohen_kappa_score(y_true, y_pred),
        n=int(len(y_true)),
    )


def per_class_report(y_true, y_pred, class_names) -> dict:
    return classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )


def confusion(y_true, y_pred, n_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


def mcnemar(y_true, pred_a, pred_b) -> tuple[float, float]:
    """McNemar's test between two models' predictions. Returns (statistic, p_value).

    Uses the exact binomial test on discordant pairs (correct for small counts, which is the
    honest choice here), matching Table 7's intent.
    """
    from scipy.stats import binomtest

    y_true = np.asarray(y_true).ravel()
    a_correct = np.asarray(pred_a).ravel() == y_true
    b_correct = np.asarray(pred_b).ravel() == y_true
    b01 = int(np.sum(a_correct & ~b_correct))  # a right, b wrong
    b10 = int(np.sum(~a_correct & b_correct))  # a wrong, b right
    n = b01 + b10
    if n == 0:
        return 0.0, 1.0
    p = binomtest(min(b01, b10), n, 0.5).pvalue
    stat = (abs(b01 - b10) - 1) ** 2 / n if n > 0 else 0.0  # continuity-corrected chi-square
    return float(stat), float(p)


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of a metric across samples."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    boots = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
