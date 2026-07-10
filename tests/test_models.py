import numpy as np

from trinet.config import CFG
from trinet.models.backbones import build_head, feature_dim
from trinet.models.ensemble import ensemble_prob, val_accuracy


def test_head_output_shape():
    dim = 768
    head = build_head(dim)
    out = head.predict(np.random.rand(3, dim).astype("float32"), verbose=0)
    assert out.shape == (3, CFG.n_classes)


def test_feature_dims_known():
    assert feature_dim("ConvNeXtTiny") == 768
    assert feature_dim("DenseNet201") == 1920


def test_ensemble_prob_normalizes_weights():
    p1 = np.array([[0.7, 0.3], [0.2, 0.8]])
    p2 = np.array([[0.6, 0.4], [0.1, 0.9]])
    # unnormalized weights should be normalized internally
    out = ensemble_prob([2.0, 2.0], [p1, p2])
    expected = (p1 + p2) / 2
    assert np.allclose(out, expected)


def test_val_accuracy_range():
    probs = [np.array([[0.9, 0.1], [0.2, 0.8]])]
    y = np.array([0, 1])
    assert val_accuracy([1.0], probs, y) == 1.0
