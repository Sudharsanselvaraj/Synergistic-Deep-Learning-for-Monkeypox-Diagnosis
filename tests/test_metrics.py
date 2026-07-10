import numpy as np
from trinet.evaluation.metrics import compute_scores, mcnemar, bootstrap_ci


def test_perfect_prediction():
    y = np.array([0, 1, 2, 0, 1, 2])
    s = compute_scores(y, y)
    assert s.accuracy == 1.0
    assert s.f1 == 1.0
    assert s.kappa == 1.0
    assert s.n == 6


def test_partial_prediction():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    s = compute_scores(y, pred)
    assert 0.0 < s.accuracy < 1.0
    assert s.as_pct()["accuracy"] == 75.0


def test_mcnemar_symmetry():
    y = np.array([0, 1, 0, 1, 0, 1])
    a = np.array([0, 1, 0, 1, 0, 1])   # all correct
    b = np.array([1, 1, 1, 1, 1, 1])   # some wrong
    stat, p = mcnemar(y, a, b)
    assert 0.0 <= p <= 1.0
    assert stat >= 0.0


def test_bootstrap_ci_bounds():
    vals = np.random.default_rng(0).normal(0.8, 0.05, 200)
    lo, hi = bootstrap_ci(vals, n_boot=500)
    assert lo < vals.mean() < hi
