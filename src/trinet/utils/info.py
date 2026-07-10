"""`trinet info` — a concise report of the installed version, environment, and artifacts.

Great for bug reports: paste the output and maintainers can see exactly what you're running.
"""

from __future__ import annotations

import platform
import subprocess

from trinet import __version__
from trinet.config import CFG


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=CFG.root, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def run() -> None:
    try:
        import tensorflow as tf

        tf_ver = tf.__version__
        gpus = len(tf.config.list_physical_devices("GPU"))
    except Exception:
        tf_ver, gpus = "not installed", 0

    ckpts = sorted(p.name for p in CFG.model_dir.glob("*")) if CFG.model_dir.exists() else []
    rows = [
        ("Tri-Net", __version__),
        ("Git commit", _git_commit()),
        ("Python", platform.python_version()),
        ("Platform", f"{platform.system()} {platform.machine()}"),
        ("TensorFlow", tf_ver),
        ("GPU / Metal", f"{gpus} device(s)"),
        ("Backbones", ", ".join(CFG.backbones)),
        ("Dataset split", "present" if CFG.data_proc.exists() else "missing"),
        ("Checkpoints", ", ".join(ckpts) if ckpts else "none"),
        ("Outputs dir", str(CFG.outputs)),
    ]
    width = max(len(k) for k, _ in rows)
    print("Tri-Net v2 — info")
    for k, v in rows:
        print(f"  {k:<{width}} : {v}")


if __name__ == "__main__":
    run()
