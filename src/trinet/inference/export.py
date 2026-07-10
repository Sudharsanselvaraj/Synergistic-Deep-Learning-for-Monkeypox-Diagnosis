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


def export(fmt: str = "savedmodel", out: Path | None = None) -> Path:
    out = Path(out) if out else (CFG.outputs / "export")
    out.mkdir(parents=True, exist_ok=True)
    model, _ = load_fusion()

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
