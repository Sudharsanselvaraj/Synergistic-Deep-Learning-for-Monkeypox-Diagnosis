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


def test_concat_mlp_roundtrip():
    """The deployed fusion architecture must reload to identical logits.

    Guards the serialization bug (100%-confident wrong predictions on reload). Concat+MLP is
    the architecture the framework deploys; the production `trinet fusion` additionally verifies
    the concrete checkpoint in a real subprocess before saving it.
    """
    # build other models first to offset the global layer-name counter (mimics a real run)
    _ = build_gated_attention(DIMS)
    _ = build_transformer(DIMS)
    model = build_concat_mlp(DIMS)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    x = [np.random.rand(8, d).astype("float32") for d in DIMS]
    before = model.predict(x, verbose=0)

    path = Path(tempfile.mkdtemp()) / "m.keras"
    model.save(path)
    tf.keras.backend.clear_session()  # simulate a fresh process (name counter reset)
    from tensorflow.keras.models import load_model

    reloaded = load_model(path, safe_mode=False, compile=False)
    after = reloaded.predict(x, verbose=0)
    assert float(np.abs(before - after).max()) < 1e-5
