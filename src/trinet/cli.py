"""Tri-Net v2 command-line interface.

A single entry point for the whole pipeline — replaces the ad-hoc ``python -m ...`` calls:

    trinet download                       # fetch datasets from Kaggle
    trinet prepare                        # build the 14-class split + clean symptoms
    trinet features [--cpu]               # cache frozen-backbone features
    trinet train --backbones ConvNeXtTiny # train classification head(s)
    trinet ensemble                       # PSO ensemble + baselines
    trinet fusion                         # fusion-strategy ablation
    trinet evaluate [--champion]          # figures + tables
    trinet gradcam                        # Grad-CAM heatmaps
    trinet diversity | cross-val | efficiency
    trinet benchmark                      # the full image pipeline end to end

Backbone names: EfficientNetB4, InceptionResNetV2, DenseNet201, ConvNeXtTiny, EfficientNetV2S.
"""
from __future__ import annotations
import sys
from typing import List, Optional

import typer

app = typer.Typer(add_completion=False, help="Tri-Net v2 — reproducible Mpox diagnosis framework.")


def _run(entry, argv: List[str]) -> None:
    """Invoke a module's argparse ``main()`` with a synthesized argv."""
    old = sys.argv
    sys.argv = ["trinet"] + argv
    try:
        entry()
    finally:
        sys.argv = old


def _bb(backbones: Optional[List[str]]) -> List[str]:
    return (["--backbones"] + list(backbones)) if backbones else []


@app.command()
def download(only: Optional[List[str]] = typer.Option(None, help="subset of datasets")):
    """Download datasets from Kaggle (needs ~/.kaggle credentials)."""
    from trinet.datasets import download as m
    _run(m.main, (["--only"] + only) if only else [])


@app.command()
def prepare():
    """Build the leakage-free 14-class split and clean the symptom CSV."""
    from trinet.datasets import prepare_lesion, prepare_symptom
    _run(prepare_lesion.main, [])
    _run(prepare_symptom.main, [])


@app.command()
def features(backbones: Optional[List[str]] = typer.Option(None),
             cpu: bool = typer.Option(False, help="force CPU (Metal lacks ops for V2/ConvNeXt)")):
    """Cache frozen-backbone features to disk."""
    from trinet.training import features as m
    _run(m.main, _bb(backbones) + (["--cpu"] if cpu else []))


@app.command()
def train(backbones: Optional[List[str]] = typer.Option(None)):
    """Train the classification head(s) on cached features."""
    from trinet.training import base as m
    _run(m.main, _bb(backbones))


@app.command()
def finetune(backbones: Optional[List[str]] = typer.Option(None),
             epochs: Optional[int] = typer.Option(None)):
    """End-to-end backbone fine-tuning (Phase-2 experiment)."""
    from trinet.training import finetune as m
    _run(m.main, _bb(backbones) + (["--epochs", str(epochs)] if epochs else []))


@app.command()
def ensemble(tag: str = typer.Option("", help="prob-file tag, e.g. 'ft'")):
    """PSO ensemble weight optimization + grid/random/equal baselines."""
    from trinet.models import ensemble as m
    _run(m.main, (["--tag", tag] if tag else []))


@app.command()
def fusion(backbones: Optional[List[str]] = typer.Option(None),
           epochs: Optional[int] = typer.Option(None)):
    """Fusion-strategy ablation (Mean/PSO/Concat-MLP/Attention/Transformer)."""
    from trinet.models import fusion as m
    _run(m.main, _bb(backbones) + (["--epochs", str(epochs)] if epochs else []))


@app.command()
def evaluate(tag: str = typer.Option("", help="prob-file tag"),
             champion: bool = typer.Option(False, help="use the Concat-MLP fusion champion")):
    """Generate evaluation figures + tables (confusion, ROC, Kappa, McNemar)."""
    from trinet.evaluation import figures as m
    _run(m.main, (["--tag", tag] if tag else []) + (["--champion"] if champion else []))


@app.command(name="cross-val")
def cross_val():
    """5-fold cross-validation (leakage-free, original features)."""
    from trinet.evaluation import cross_val as m
    _run(m.main, [])


@app.command()
def diversity():
    """Ensemble prediction-diversity analysis (disagreement, Q, rho)."""
    from trinet.evaluation import diversity as m
    _run(m.main, [])


@app.command()
def efficiency(backbones: Optional[List[str]] = typer.Option(None)):
    """Efficiency profile (params, FLOPs, latency) per backbone."""
    from trinet.evaluation import efficiency as m
    _run(m.main, _bb(backbones))


@app.command()
def symptom():
    """Symptom module: CNN + LR/RF/XGBoost baselines + 'sum' ablation."""
    from trinet.training import symptom as m
    _run(m.main, [])


@app.command()
def gradcam():
    """Grad-CAM heatmaps for the base models (runs on CPU)."""
    from trinet.explainability import gradcam as m
    _run(m.main, [])


@app.command()
def benchmark():
    """Run the full image pipeline end to end: prepare→features→train→ensemble→fusion→evaluate."""
    from trinet.datasets import prepare_lesion
    from trinet.training import features as feat, base
    from trinet.models import ensemble, fusion as fus
    from trinet.evaluation import figures, diversity as div
    for entry in (prepare_lesion.main, feat.main, base.main, ensemble.main,
                  fus.main, figures.main, div.main):
        _run(entry, [])


if __name__ == "__main__":
    app()
