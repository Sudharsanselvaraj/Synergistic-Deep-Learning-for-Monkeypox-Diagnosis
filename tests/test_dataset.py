from pathlib import Path

import numpy as np

from trinet.config import CFG
from trinet.datasets.prepare_lesion import _split


def _files(n):
    return [Path(f"img_{i}.jpg") for i in range(n)]


def test_fourteen_classes():
    assert CFG.n_classes == 14
    assert len(CFG.lesion_classes) == 14
    assert CFG.mpox_class in CFG.lesion_classes


def test_split_fractions_sum_to_one():
    assert abs(CFG.train_frac + CFG.val_frac + CFG.test_frac - 1.0) < 1e-9


def test_output_paths_under_outputs():
    assert CFG.model_dir.name == "checkpoints"
    assert CFG.tbl_dir.name == "reports"
    assert CFG.outputs.name == "outputs"


def test_split_normal_class_populates_all_three():
    s = _split(_files(100), np.random.default_rng(0))
    assert len(s["train"]) + len(s["val"]) + len(s["test"]) == 100
    assert s["val"] and s["test"]


def test_split_small_class_never_empties_val_or_test():
    for n in (3, 4, 5, 8):
        s = _split(_files(n), np.random.default_rng(0))
        assert len(s["val"]) >= 1 and len(s["test"]) >= 1
        assert len(s["train"]) + len(s["val"]) + len(s["test"]) == n
        assert len(s["train"]) >= 1


def test_split_degenerate_class_goes_all_to_train():
    s = _split(_files(2), np.random.default_rng(0))
    assert len(s["train"]) == 2 and not s["val"] and not s["test"]
