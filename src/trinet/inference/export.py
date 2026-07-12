"""Export the trained fusion champion for deployment.

Writes a TensorFlow SavedModel (always) and, if `tf2onnx` is installed, an ONNX graph. These
are the artifacts to attach to a GitHub Release so users can run inference without training.

    trinet export --format onnx --out outputs/export
"""

from __future__ import annotations

import sys
from pathlib import Path

from trinet.config import CFG
from trinet.models.fusion import load_fusion


def _mean_keras_model(bbs: list[str]):
    """A single Keras graph equivalent to the Mean ensemble (average the per-backbone heads).

    ``load_fusion`` returns a lightweight ``_MeanEnsemble`` for the Mean champion, which has no
    exportable graph. This reconstructs the same computation as a functional model so the Mean
    champion exports to SavedModel/ONNX like any learned fusion.
    """
    from tensorflow.keras import Input, Model, layers
    from tensorflow.keras.models import load_model

    from trinet.models.backbones import feature_dim

    heads = [load_model(CFG.model_dir / f"head_{bb}.keras") for bb in bbs]
    inputs = [Input(shape=(feature_dim(bb),), name=f"feat{i}") for i, bb in enumerate(bbs)]
    outs = [h(inp) for h, inp in zip(heads, inputs)]
    avg = outs[0] if len(outs) == 1 else layers.Average(name="mean")(outs)
    return Model(inputs, avg, name="mean_ensemble")


def export(fmt: str = "savedmodel", out: Path | None = None) -> Path:
    out = Path(out) if out else (CFG.outputs / "export")
    out.mkdir(parents=True, exist_ok=True)
    model, bbs = load_fusion()
    # The Mean champion is a plain averaging wrapper with no Keras graph; build an equivalent
    # functional model so `.export()` works for every deployed strategy.
    if not hasattr(model, "export"):
        model = _mean_keras_model(bbs)

    sm_path = out / "fusion_savedmodel"
    model.export(str(sm_path))
    print(f"[/] SavedModel -> {sm_path}")

    if fmt == "onnx":
        try:
            import subprocess

            import tf2onnx  # noqa: F401

            onnx_path = out / "fusion.onnx"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tf2onnx.convert",
                    "--saved-model",
                    str(sm_path),
                    "--output",
                    str(onnx_path),
                ],
                check=True,
            )
            print(f"[/] ONNX -> {onnx_path}")
        except ImportError:
            print("[!] tf2onnx not installed; skipping ONNX (pip install tf2onnx)")
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["savedmodel", "onnx"], default="savedmodel")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    export(args.format, args.out)


if __name__ == "__main__":
    main()
