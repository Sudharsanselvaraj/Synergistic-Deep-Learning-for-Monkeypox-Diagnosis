"""`trinet doctor` — environment and readiness diagnostics.

Checks the interpreter, TensorFlow, GPU/Metal availability, whether datasets, cached features
and checkpoints exist, and whether Kaggle credentials are configured — so users can tell at a
glance what still needs setting up.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

from trinet.config import CFG

OK, WARN, BAD = "✓", "!", "✗"


def _line(status: str, label: str, detail: str = "") -> str:
    return f"  [{status}] {label:<22} {detail}"


def run() -> None:
    rows = []
    rows.append(_line(OK, "Python", platform.python_version()))
    rows.append(_line(OK, "Platform", f"{platform.system()} {platform.machine()}"))

    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        rows.append(_line(OK, "TensorFlow", tf.__version__))
        rows.append(
            _line(
                OK if gpus else WARN,
                "GPU / Metal",
                f"{len(gpus)} device(s)" if gpus else "none (CPU only)",
            )
        )
    except Exception as e:  # pragma: no cover
        rows.append(_line(BAD, "TensorFlow", f"import failed: {e}"))

    def _exists(p: Path, label: str, hint: str):
        present = p.exists() and any(p.iterdir()) if p.is_dir() else p.exists()
        rows.append(
            _line(OK if present else WARN, label, str(p) if present else f"missing — {hint}")
        )

    _exists(CFG.data_proc, "Dataset split", "run `trinet download && trinet prepare`")
    _exists(CFG.features, "Cached features", "run `trinet features`")
    _exists(CFG.model_dir / "fusion.keras", "Champion checkpoint", "run `trinet fusion`")
    kaggle = (Path.home() / ".kaggle" / "kaggle.json").exists() or (
        Path.home() / ".kaggle" / "access_token"
    ).exists()
    rows.append(
        _line(
            OK if kaggle else WARN,
            "Kaggle credentials",
            "configured" if kaggle else "missing — needed for `trinet download`",
        )
    )

    print("Tri-Net v2 environment check\n" + "\n".join(rows))
    if any(BAD in r for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    run()
