"""PSO-optimized weighted ensemble — the paper's headline contribution, implemented for real.

The three heads each output softmax probabilities. We combine them as
    P_final = w1·P_eff + w2·P_incres + w3·P_dense,   w >= 0,  Σw = 1,
and use Particle Swarm Optimization to choose the weights that MAXIMIZE validation accuracy.
Weights are fit on VAL only, then applied once to TEST (no test-set peeking).

We also run equal-weight / grid / random search baselines so the PSO benefit is reported
honestly (this is the paper's Table 4). Convergence history is saved for the PSO curve (Fig 11).

Outputs:
  results/models/pso_weights.json          (best weights + val accuracy + convergence history)
  results/tables/ensemble_scores.csv       (PSO test metrics + optimizer comparison)
"""

from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import numpy as np

from trinet.config import CFG, ensure_dirs  # noqa: E402
from trinet.evaluation.metrics import compute_scores  # noqa: E402


def _prob_path(split: str, bb: str, tag: str) -> Path:
    mid = f"{tag}_" if tag else ""
    return CFG.model_dir / f"prob_{split}_{mid}{bb}.npy"


def _load_probs(split: str, tag: str = ""):
    probs = [np.load(_prob_path(split, bb, tag)) for bb in CFG.backbones]
    y = np.load(CFG.features / f"y_{split}.npy")
    return probs, y


def _normalize(w: np.ndarray) -> np.ndarray:
    w = np.clip(np.asarray(w, dtype=float), 0, None)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)


def ensemble_prob(w, probs) -> np.ndarray:
    w = _normalize(w)
    return sum(wi * p for wi, p in zip(w, probs))


def val_accuracy(w, probs, y) -> float:
    return float((ensemble_prob(w, probs).argmax(1) == y).mean())


# ---- optimizers over the weight simplex ----
def opt_equal(probs, y):
    return _normalize(np.ones(len(CFG.backbones))), None


def opt_grid(probs, y, step=0.05):
    best_w, best_acc = None, -1.0
    grid = np.arange(0, 1 + 1e-9, step)
    for a, b in product(grid, grid):
        if a + b > 1 + 1e-9:
            continue
        w = np.array([a, b, 1 - a - b])
        acc = val_accuracy(w, probs, y)
        if acc > best_acc:
            best_acc, best_w = acc, w
    return best_w, None


def opt_random(probs, y, n=2000, seed=CFG.seed):
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(len(CFG.backbones)), size=n)
    accs = np.array([val_accuracy(w, probs, y) for w in W])
    return W[accs.argmax()], None


def opt_pso(probs, y):
    import pyswarms as ps

    d = len(CFG.backbones)

    def cost(X):
        return np.array([1.0 - val_accuracy(x, probs, y) for x in X])

    opt = ps.single.GlobalBestPSO(
        n_particles=CFG.pso_particles,
        dimensions=d,
        options={"c1": CFG.pso_c1, "c2": CFG.pso_c2, "w": CFG.pso_w},
        bounds=(np.zeros(d), np.ones(d)),
    )
    _, pos = opt.optimize(cost, iters=CFG.pso_iters, verbose=False)
    history = [1.0 - c for c in opt.cost_history]  # cost -> val accuracy
    return _normalize(pos), history


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="prob-file tag, e.g. 'ft' for fine-tuned models")
    tag = ap.parse_args().tag
    suffix = f"_{tag}" if tag else ""
    ensure_dirs()
    val_probs, y_val = _load_probs("val", tag)
    test_probs, y_test = _load_probs("test", tag)

    methods = {"Equal": opt_equal, "Grid": opt_grid, "Random": opt_random, "PSO": opt_pso}
    rows, pso_history, pso_weights = [], None, None
    for name, fn in methods.items():
        t0 = time.time()
        w, hist = fn(val_probs, y_val)
        dt = time.time() - t0
        v_acc = val_accuracy(w, val_probs, y_val)
        s = compute_scores(
            y_test, ensemble_prob(w, test_probs).argmax(1), ensemble_prob(w, test_probs)
        )
        rows.append(
            {
                "method": name,
                "weights": [round(float(x), 4) for x in _normalize(w)],
                "val_acc": round(100 * v_acc, 2),
                "time_s": round(dt, 2),
                **s.as_pct(),
            }
        )
        print(
            f"[{name:>6}] w={_normalize(w).round(3)} val_acc={100 * v_acc:.2f} "
            f"test_acc={s.as_pct()['accuracy']} time={dt:.2f}s"
        )
        if name == "PSO":
            pso_history, pso_weights = hist, _normalize(w).tolist()

    (CFG.model_dir / f"pso_weights{suffix}.json").write_text(
        json.dumps(
            {
                "backbones": CFG.backbones,
                "weights": pso_weights,
                "val_accuracy": val_accuracy(pso_weights, val_probs, y_val),
                "convergence": pso_history,
            },
            indent=2,
        )
    )

    cols = [
        "method",
        "weights",
        "val_acc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "kappa",
        "time_s",
    ]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(json.dumps(r[c]) if c == "weights" else str(r[c]) for c in cols))
    (CFG.tbl_dir / f"ensemble_scores{suffix}.csv").write_text("\n".join(lines))
    (CFG.tbl_dir / f"ensemble_scores{suffix}.json").write_text(json.dumps(rows, indent=2))
    print(f"\n[✓] ensemble scores -> {CFG.tbl_dir / f'ensemble_scores{suffix}.csv'}")
    print(f"    next: python -m src.eval.run_all{' --tag ' + tag if tag else ''}")


if __name__ == "__main__":
    main()
