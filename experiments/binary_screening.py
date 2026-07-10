"""Binary Mpox-vs-rest screening — the clinically-relevant task, reported honestly.

Collapses the 14-class champion (Concat+MLP fusion) predictions to Mpox / Non-Mpox and reports
accuracy, AUC, sensitivity, specificity and precision with Wilson confidence intervals. The
test set is class-imbalanced (few Mpox cases), so accuracy alone is not sufficient — sensitivity
and specificity are reported alongside.

    python experiments/binary_screening.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trinet.config import CFG  # noqa: E402


def _wilson(k: int, n: int, z: float = 1.96):
    """Wilson score 95% interval for a proportion, returned as percentages."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return 100 * (center - margin), 100 * (center + margin)


def main() -> None:
    y = np.load(CFG.features / "y_test.npy")
    prob = np.load(CFG.model_dir / "prob_test_fusion.npy")
    mpox = CFG.lesion_classes.index(CFG.mpox_class)

    yb = (y == mpox).astype(int)
    pb = (prob.argmax(1) == mpox).astype(int)
    tp = int(np.sum((yb == 1) & (pb == 1)))
    fn = int(np.sum((yb == 1) & (pb == 0)))
    tn = int(np.sum((yb == 0) & (pb == 0)))
    fp = int(np.sum((yb == 0) & (pb == 1)))
    n = len(yb)

    acc = (tp + tn) / n
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    auc = roc_auc_score(yb, prob[:, mpox])

    print("Binary Mpox-vs-rest screening (champion Concat+MLP)")
    print(f"  confusion: TP={tp} FN={fn} TN={tn} FP={fp}  (n={n})")
    print(
        f"  accuracy    : {100 * acc:5.2f}%   95% CI {_wilson(tp + tn, n)[0]:.2f}-{_wilson(tp + tn, n)[1]:.2f}"
    )
    print(
        f"  sensitivity : {100 * sens:5.2f}%   95% CI {_wilson(tp, tp + fn)[0]:.2f}-{_wilson(tp, tp + fn)[1]:.2f}"
    )
    print(
        f"  specificity : {100 * spec:5.2f}%   95% CI {_wilson(tn, tn + fp)[0]:.2f}-{_wilson(tn, tn + fp)[1]:.2f}"
    )
    print(f"  precision   : {100 * prec:5.2f}%")
    print(f"  AUC         : {100 * auc:5.2f}%")

    lines = [
        "metric,value,ci_low,ci_high",
        f"accuracy,{100 * acc:.2f},{_wilson(tp + tn, n)[0]:.2f},{_wilson(tp + tn, n)[1]:.2f}",
        f"sensitivity,{100 * sens:.2f},{_wilson(tp, tp + fn)[0]:.2f},{_wilson(tp, tp + fn)[1]:.2f}",
        f"specificity,{100 * spec:.2f},{_wilson(tn, tn + fp)[0]:.2f},{_wilson(tn, tn + fp)[1]:.2f}",
        f"precision,{100 * prec:.2f},,",
        f"auc,{100 * auc:.2f},,",
    ]
    (CFG.tbl_dir / "binary_screening.csv").write_text("\n".join(lines))
    print(f"\n[/] wrote {CFG.tbl_dir / 'binary_screening.csv'}")


if __name__ == "__main__":
    main()
