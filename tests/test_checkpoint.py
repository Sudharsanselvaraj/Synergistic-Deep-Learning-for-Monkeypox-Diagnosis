"""Checkpoint round-trip invariant: a saved model must reload to IDENTICAL logits.

This guards the serialization bug that (before the fix) produced 100%-confident wrong
predictions on reload while aggregate accuracy coincidentally matched. We assert logits, not
accuracy — the strict check the framework enforces at save time.
"""

import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

from trinet.models.fusion import (  # noqa: E402
    build_concat_mlp,
    build_gated_attention,
    build_transformer,
)

DIMS = [768, 1536, 1920]


def _reload_matches(build) -> float:
    # build other models first to offset the global layer-name counter (mimics a real run)
    _ = build_gated_attention(DIMS)
    _ = build_transformer(DIMS)
    model = build(DIMS)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    x = [np.random.rand(8, d).astype("float32") for d in DIMS]
    before = model.predict(x, verbose=0)

    # full-model .keras save/load is what the framework uses — it reconstructs the exact model
    path = Path(tempfile.mkdtemp()) / "m.keras"
    model.save(path)
    tf.keras.backend.clear_session()  # simulate a fresh process (name counter reset)
    from tensorflow.keras.models import load_model

    reloaded = load_model(path, safe_mode=False, compile=False)
    after = reloaded.predict(x, verbose=0)
    return float(np.abs(before - after).max())


def test_concat_mlp_roundtrip():
    assert _reload_matches(build_concat_mlp) < 1e-5


def test_gated_attention_roundtrip():
    assert _reload_matches(build_gated_attention) < 1e-5


def test_transformer_roundtrip():
    assert _reload_matches(build_transformer) < 1e-5
