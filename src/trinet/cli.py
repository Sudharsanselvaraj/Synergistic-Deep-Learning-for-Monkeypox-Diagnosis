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

import typer

app = typer.Typer(add_completion=False, help="Tri-Net v2 — reproducible Mpox diagnosis framework.")


def _run(entry, argv: list[str]) -> None:
    """Invoke a module's argparse ``main()`` with a synthesized argv."""
    old = sys.argv
    sys.argv = ["trinet"] + argv
    try:
        entry()
    finally:
        sys.argv = old


def _bb(backbones: list[str] | None) -> list[str]:
    return (["--backbones"] + list(backbones)) if backbones else []


@app.command()
def download(only: list[str] | None = typer.Option(None, help="subset of datasets")):
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
def features(
    backbones: list[str] | None = typer.Option(None),
    cpu: bool = typer.Option(False, help="force CPU (Metal lacks ops for V2/ConvNeXt)"),
):
    """Cache frozen-backbone features to disk."""
    from trinet.training import features as m

    _run(m.main, _bb(backbones) + (["--cpu"] if cpu else []))


@app.command()
def train(backbones: list[str] | None = typer.Option(None)):
    """Train the classification head(s) on cached features."""
    from trinet.training import base as m

    _run(m.main, _bb(backbones))


@app.command()
def finetune(
    backbones: list[str] | None = typer.Option(None), epochs: int | None = typer.Option(None)
):
    """End-to-end backbone fine-tuning (Phase-2 experiment)."""
    from trinet.training import finetune as m

    _run(m.main, _bb(backbones) + (["--epochs", str(epochs)] if epochs else []))


@app.command()
def ensemble(tag: str = typer.Option("", help="prob-file tag, e.g. 'ft'")):
    """PSO ensemble weight optimization + grid/random/equal baselines."""
    from trinet.models import ensemble as m

    _run(m.main, (["--tag", tag] if tag else []))


@app.command()
def fusion(
    backbones: list[str] | None = typer.Option(None), epochs: int | None = typer.Option(None)
):
    """Fusion-strategy ablation (Mean/PSO/Concat-MLP/Attention/Transformer)."""
    from trinet.models import fusion as m

    _run(m.main, _bb(backbones) + (["--epochs", str(epochs)] if epochs else []))


@app.command()
def evaluate(
    tag: str = typer.Option("", help="prob-file tag"),
    champion: bool = typer.Option(False, help="use the Concat-MLP fusion champion"),
):
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
def efficiency(
    backbones: list[str] | None = typer.Option(None),
    gpu: bool = typer.Option(False, help="measure on GPU (Metal lacks ConvNeXt/EffV2 ops)"),
):
    """Efficiency profile (params, FLOPs, latency) per backbone."""
    from trinet.evaluation import efficiency as m

    _run(m.main, _bb(backbones) + (["--gpu"] if gpu else []))


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
def predict(
    image: str = typer.Argument(..., help="path to a skin-lesion image"),
    top_k: int = typer.Option(3, "--top-k"),
):
    """Predict the disease class (and Mpox screening) for a single image."""
    from trinet.inference import predict as m

    _run(m.main, [image, "--top-k", str(top_k)])


@app.command()
def explain():
    """Generate Grad-CAM explanations for the base models (alias of gradcam)."""
    from trinet.explainability import gradcam as m

    _run(m.main, [])


@app.command()
def export(
    fmt: str = typer.Option("savedmodel", "--format", help="savedmodel | onnx"),
    out: str | None = typer.Option(None),
):
    """Export the fusion champion (SavedModel / ONNX) for deployment."""
    from trinet.inference import export as m

    _run(m.main, ["--format", fmt] + (["--out", out] if out else []))


@app.command()
def doctor():
    """Check environment: Python, TensorFlow, GPU/Metal, datasets, checkpoints, credentials."""
    from trinet.utils import doctor as m

    m.run()


@app.command()
def info():
    """Print version, git commit, environment, backbones, and available checkpoints."""
    from trinet.utils import info as m

    m.run()


@app.command()
def reproduce():
    """Reproduce every artifact end to end: data → features → train → fusion → evaluate → symptom.

    Self-contained from a fresh install. The Kaggle download is skipped when the raw datasets are
    already present; feature caching runs on CPU so it works on every platform (some backbones
    lack Metal/GPU ops).
    """
    from trinet.config import CFG
    from trinet.datasets import download as dl
    from trinet.datasets import prepare_lesion, prepare_symptom
    from trinet.evaluation import cross_val, diversity, figures, gradcam
    from trinet.models import ensemble
    from trinet.models import fusion as fus
    from trinet.training import base, features
    from trinet.training import symptom as symp

    # 1) data: fetch only if the raw datasets aren't already downloaded
    if CFG.data_raw.exists() and any(CFG.data_raw.iterdir()):
        print("[i] raw datasets present — skipping download")
    else:
        _run(dl.main, [])
    _run(prepare_lesion.main, [])
    _run(prepare_symptom.main, [])
    _run(features.main, ["--cpu"])

    # 2) models + evaluation
    _run(base.main, [])
    _run(ensemble.main, [])
    _run(fus.main, [])
    _run(figures.main, ["--champion"])
    _run(diversity.main, [])
    _run(cross_val.main, [])
    _run(gradcam.main, [])
    _run(symp.main, [])
    print("\n[✓] reproduced all artifacts")


@app.command()
def benchmark():
    """Run the full image pipeline end to end: prepare→features→train→ensemble→fusion→evaluate."""
    from trinet.datasets import prepare_lesion
    from trinet.evaluation import diversity as div
    from trinet.evaluation import figures
    from trinet.models import ensemble
    from trinet.models import fusion as fus
    from trinet.training import base
    from trinet.training import features as feat

    for entry in (
        prepare_lesion.main,
        feat.main,
        base.main,
        ensemble.main,
        fus.main,
        figures.main,
        div.main,
    ):
        _run(entry, [])


if __name__ == "__main__":
    app()
