from trinet.config import CFG


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
