"""Ensemble prediction-diversity analysis — why the ensemble is robust.

The ensemble held ~72.6% even when a fine-tuned member degraded, which suggests it works as a
variance-reduction mechanism rather than averaging three lookalike classifiers. This quantifies
that with standard diversity measures over the base models' TEST predictions:

  - disagreement : fraction of samples where the pair predicts differently (higher = more diverse)
  - Q-statistic  : Yule's Q on correct/incorrect agreement (lower/→0 = more diverse)
  - rho          : correlation of the two models' correctness (lower = more diverse)
  - double_fault : fraction where BOTH are wrong (lower = safer to ensemble)

Output: results/tables/diversity.csv
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from trinet.config import CFG, ensure_dirs  # noqa: E402


def _preds(tag: str = ""):
    mid = f"{tag}_" if tag else ""
    y = np.load(CFG.features / "y_test.npy")
    preds = {
        bb: np.load(CFG.model_dir / f"prob_test_{mid}{bb}.npy").argmax(1) for bb in CFG.backbones
    }
    return y, preds


def _pair_stats(y, a, b) -> dict:
    ca, cb = (a == y), (b == y)
    n11 = int(np.sum(ca & cb))
    n10 = int(np.sum(ca & ~cb))
    n01 = int(np.sum(~ca & cb))
    n00 = int(np.sum(~ca & ~cb))
    n = len(y)
    disagree = (n10 + n01) / n
    denom = n11 * n00 + n01 * n10
    q = (n11 * n00 - n01 * n10) / denom if denom else 0.0
    # correlation of correctness
    va, vb = ca.astype(float), cb.astype(float)
    sd = va.std() * vb.std()
    rho = float(np.mean((va - va.mean()) * (vb - vb.mean())) / sd) if sd else 0.0
    return {
        "disagreement": round(disagree, 4),
        "Q": round(q, 4),
        "rho": round(rho, 4),
        "double_fault": round(n00 / n, 4),
    }


def main() -> None:
    ensure_dirs()
    y, preds = _preds()
    rows = []
    for a, b in combinations(CFG.backbones, 2):
        rows.append({"pair": f"{a}|{b}", **_pair_stats(y, preds[a], preds[b])})

    cols = ["pair", "disagreement", "Q", "rho", "double_fault"]
    mean_row = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in cols[1:]}
    lines = [",".join(cols)]
    lines += [",".join(str(r[c]) for c in cols) for r in rows]
    lines.append("MEAN," + ",".join(str(mean_row[k]) for k in cols[1:]))
    (CFG.tbl_dir / "diversity.csv").write_text("\n".join(lines))

    print("Pairwise diversity (test):")
    for r in rows:
        print(
            f"  {r['pair']:<36} disagree={r['disagreement']}  Q={r['Q']}  "
            f"rho={r['rho']}  double_fault={r['double_fault']}"
        )
    print(
        f"  {'MEAN':<36} disagree={mean_row['disagreement']}  Q={mean_row['Q']}  "
        f"rho={mean_row['rho']}  double_fault={mean_row['double_fault']}"
    )
    print(f"\n[✓] diversity -> {CFG.tbl_dir / 'diversity.csv'}")


if __name__ == "__main__":
    main()
