"""Efficiency profile for every backbone — accuracy is only half the story for deployable AI.

For each frozen backbone (+ its head) we record parameters, estimated FLOPs, inference latency
(ms/image) and model size. Combined with the accuracy tables, this justifies model choices on
cost as well as performance — what medical-AI reviewers increasingly expect.

Runs on CPU by default: the default backbones (ConvNeXt-Tiny, EfficientNetV2-S) have no
tensorflow-metal kernels, so profiling them on an Apple-Silicon GPU errors ("could not find
registered platform"). Pass ``--gpu`` to measure on-device latency for Metal-compatible
backbones (e.g. ``trinet efficiency --gpu --backbones DenseNet201``).

Output: results/tables/efficiency.csv
"""

from __future__ import annotations

import sys
import time

import tensorflow as tf

# Force CPU before the first TF op unless the user explicitly opts into the GPU. Must happen
# here (import time), before build_full_model runs any op that would initialise Metal.
if "--gpu" not in sys.argv:
    tf.config.set_visible_devices([], "GPU")

from trinet.config import CFG, ensure_dirs  # noqa: E402
from trinet.models.backbones import build_full_model  # noqa: E402


def _flops(model) -> float:
    """Best-effort FLOPs (GFLOPs) via the TF profiler; returns nan if unavailable."""
    try:
        from tensorflow.python.profiler.model_analyzer import profile
        from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

        forward = tf.function(lambda x: model(x))
        concrete = forward.get_concrete_function(tf.TensorSpec([1, *CFG.img_shape], tf.float32))
        frozen = concrete.graph
        opts = ProfileOptionBuilder(ProfileOptionBuilder.float_operation()).build()
        flops = profile(frozen, options=opts)
        return flops.total_float_ops / 1e9 if flops else float("nan")
    except Exception:
        return float("nan")


def _latency_ms(model, n_warmup=5, n_iter=30) -> float:
    x = tf.random.uniform([1, *CFG.img_shape])
    for _ in range(n_warmup):
        model(x, training=False)
    t0 = time.time()
    for _ in range(n_iter):
        model(x, training=False)
    return 1000.0 * (time.time() - t0) / n_iter


def profile_backbone(bb: str) -> dict:
    model = build_full_model(bb, finetune_unfreeze=0)
    params_m = model.count_params() / 1e6
    size_mb = model.count_params() * 4 / 1e6  # float32 weights
    row = {
        "backbone": bb,
        "params_M": round(params_m, 2),
        "gflops": round(_flops(model), 2),
        "latency_ms": round(_latency_ms(model), 2),
        "size_MB": round(size_mb, 1),
    }
    tf.keras.backend.clear_session()
    print(
        f"[✓] {bb}: {row['params_M']}M params, {row['gflops']} GFLOPs, {row['latency_ms']} ms/img"
    )
    return row


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="*", default=CFG.backbones)
    ap.add_argument(
        "--gpu",
        action="store_true",
        help="measure on the GPU (Metal-incompatible backbones like ConvNeXt/EffV2 will error)",
    )
    backbones = ap.parse_args().backbones
    device = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
    print(f"[i] profiling on {device}")
    ensure_dirs()
    rows = [profile_backbone(bb) for bb in backbones]
    cols = ["backbone", "params_M", "gflops", "latency_ms", "size_MB"]
    (CFG.tbl_dir / "efficiency.csv").write_text(
        "\n".join([",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in rows])
    )
    print(f"\n[✓] efficiency -> {CFG.tbl_dir / 'efficiency.csv'}")


if __name__ == "__main__":
    main()
