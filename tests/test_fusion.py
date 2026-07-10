import numpy as np
import pytest

from trinet.config import CFG
from trinet.models.fusion import build_concat_mlp, build_gated_attention, build_transformer

DIMS = [768, 1536, 1920]          # ConvNeXt / InceptionResNetV2 / DenseNet201


@pytest.mark.parametrize("build", [build_concat_mlp, build_gated_attention, build_transformer])
def test_fusion_output_shape(build):
    model = build(DIMS)
    batch = 4
    inputs = [np.random.rand(batch, d).astype("float32") for d in DIMS]
    out = model.predict(inputs, verbose=0)
    assert out.shape == (batch, CFG.n_classes)
    # softmax rows sum to 1
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-4)


def test_fusion_is_learnable():
    model = build_concat_mlp(DIMS)
    assert model.count_params() > 0
