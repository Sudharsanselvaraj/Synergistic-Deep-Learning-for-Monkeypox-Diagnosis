"""Generate the evaluation figures + tables for the lesion module — all from saved data.

Reads the per-model probabilities written by train_base + the PSO weights, then produces the
honest equivalents of the paper's figures/tables:
  results/figures/confusion_14class.png     (Fig. 7)
  results/figures/confusion_binary.png       (Fig. 6 — Mpox vs rest)
  results/figures/roc_curves.png             (Fig. 9)
  results/figures/kappa.png                  (Fig. 8)
  results/tables/evaluation.csv              (per-model + ensemble metrics)
  results/tables/mcnemar.csv                 (Table 7 — Tri-Net vs each base)

Nothing here is hand-typed; re-run after retraining to refresh every artifact.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc

from trinet.config import CFG, ensure_dirs                              # noqa: E402
from trinet.evaluation.metrics import compute_scores, confusion, mcnemar  # noqa: E402
from trinet.models.ensemble import ensemble_prob                # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

_SUFFIX = ""  # set from --tag in main(); keeps fine-tuned artifacts separate from Phase-1


def _load(tag: str = "", champion: bool = False):
    mid = f"{tag}_" if tag else ""
    suffix = f"_{tag}" if tag else ""
    y_test = np.load(CFG.features / "y_test.npy")
    base = {bb: np.load(CFG.model_dir / f"prob_test_{mid}{bb}.npy") for bb in CFG.backbones}
    if champion:
        # use the best learned fusion model (Concat+MLP) as the headline ensemble
        ens = np.load(CFG.model_dir / "prob_test_fusion.npy")
        return y_test, base, ens, None
    w = json.loads((CFG.model_dir / f"pso_weights{suffix}.json").read_text())["weights"]
    ens = ensemble_prob(w, [base[bb] for bb in CFG.backbones])
    return y_test, base, ens, w


def fig_confusion_14(y, ens):
    cm = confusion(y, ens.argmax(1), CFG.n_classes)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="viridis")
    ax.set(xticks=range(CFG.n_classes), yticks=range(CFG.n_classes),
           xlabel="Predicted", ylabel="True", title="Tri-Net — 14-class confusion (test)")
    ax.set_xticklabels(CFG.lesion_classes, rotation=90)
    ax.set_yticklabels(CFG.lesion_classes)
    for i in range(CFG.n_classes):
        for j in range(CFG.n_classes):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] < cm.max() / 2 else "black", fontsize=7)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(CFG.fig_dir / f"confusion_14class{_SUFFIX}.png")
    plt.close(fig)


def fig_confusion_binary(y, ens):
    mpox = CFG.lesion_classes.index(CFG.mpox_class)
    yb = (y == mpox).astype(int)
    pb = (ens.argmax(1) == mpox).astype(int)
    cm = np.array([[np.sum((yb == 0) & (pb == 0)), np.sum((yb == 0) & (pb == 1))],
                   [np.sum((yb == 1) & (pb == 0)), np.sum((yb == 1) & (pb == 1))]])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No Mpox", "Mpox"],
           yticklabels=["No Mpox", "Mpox"], xlabel="Predicted", ylabel="Actual",
           title="Tri-Net — Mpox vs rest (test)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(CFG.fig_dir / f"confusion_binary{_SUFFIX}.png")
    plt.close(fig)


def fig_roc(y, base, ens):
    mpox = CFG.lesion_classes.index(CFG.mpox_class)
    yb = (y == mpox).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for bb in CFG.backbones:
        fpr, tpr, _ = roc_curve(yb, base[bb][:, mpox])
        ax.plot(fpr, tpr, lw=1, label=f"{bb} (AUC={sk_auc(fpr, tpr):.3f})")
    fpr, tpr, _ = roc_curve(yb, ens[:, mpox])
    ax.plot(fpr, tpr, lw=2.2, color="k", label=f"Tri-Net (AUC={sk_auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC — Mpox detection")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(CFG.fig_dir / f"roc_curves{_SUFFIX}.png")
    plt.close(fig)


def fig_kappa(y, base, ens):
    names = CFG.backbones + ["Tri-Net"]
    kappas = [compute_scores(y, base[bb].argmax(1), base[bb]).kappa for bb in CFG.backbones]
    kappas.append(compute_scores(y, ens.argmax(1), ens).kappa)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(names, kappas, color=["#7aa" if n != "Tri-Net" else "#248" for n in names])
    ax.set(ylabel="Cohen's κ", title="Cohen's Kappa", ylim=(0, 1))
    for b, k in zip(bars, kappas):
        ax.text(b.get_x() + b.get_width() / 2, k + 0.01, f"{k:.3f}", ha="center", fontsize=8)
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(CFG.fig_dir / f"kappa{_SUFFIX}.png")
    plt.close(fig)


def table_eval(y, base, ens):
    rows = []
    for bb in CFG.backbones:
        rows.append({"model": bb, **compute_scores(y, base[bb].argmax(1), base[bb]).as_pct()})
    rows.append({"model": "Tri-Net (PSO)", **compute_scores(y, ens.argmax(1), ens).as_pct()})
    cols = ["model", "accuracy", "precision", "recall", "f1", "auc", "kappa", "n"]
    (CFG.tbl_dir / f"evaluation{_SUFFIX}.csv").write_text(
        "\n".join([",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in rows]))
    return rows


def table_mcnemar(y, base, ens):
    ens_pred = ens.argmax(1)
    lines = ["comparison,statistic,p_value,significant"]
    for bb in CFG.backbones:
        stat, p = mcnemar(y, ens_pred, base[bb].argmax(1))
        lines.append(f"Tri-Net vs {bb},{stat:.4f},{p:.4g},{'yes' if p < 0.05 else 'no'}")
    (CFG.tbl_dir / f"mcnemar{_SUFFIX}.csv").write_text("\n".join(lines))


def main() -> None:
    import argparse
    global _SUFFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="prob-file tag, e.g. 'ft' for fine-tuned models")
    ap.add_argument("--champion", action="store_true",
                    help="use the best learned fusion model (Concat+MLP) as the headline ensemble")
    args = ap.parse_args()
    tag = args.tag
    _SUFFIX = "_champion" if args.champion else (f"_{tag}" if tag else "")
    ensure_dirs()
    y, base, ens, w = _load(tag, champion=args.champion)
    fig_confusion_14(y, ens)
    fig_confusion_binary(y, ens)
    fig_roc(y, base, ens)
    fig_kappa(y, base, ens)
    rows = table_eval(y, base, ens)
    table_mcnemar(y, base, ens)
    print("Ensemble:", "Concat+MLP fusion (champion)" if w is None
          else f"weights {np.round(w, 3).tolist()}")
    for r in rows:
        print(f"  {r['model']:<22} acc={r['accuracy']:>6} f1={r['f1']:>6} "
              f"auc={r['auc']:>6} kappa={r['kappa']:>6}")
    print(f"\n[✓] figures -> {CFG.fig_dir}\n[✓] tables  -> {CFG.tbl_dir}")


if __name__ == "__main__":
    main()
