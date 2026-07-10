"""Cache frozen-backbone features to disk — the key to training a triple ensemble on a Mac.

For each backbone we run every image through `preprocess -> backbone -> global-avg-pool` ONCE
and save the pooled vectors. Head training then reads these `.npy` files and runs in
seconds/epoch instead of forwarding through three heavy CNNs every epoch.

  - TRAIN: `aug_multiplier` variants per image (1 original + K-1 augmented) — augmentation
    happens here, before the frozen backbone, so heads still benefit from it.
  - VAL / TEST: cached clean (no augmentation), so evaluation is honest.

Output (per backbone bb):
  data/features/X_train_<bb>.npy, X_val_<bb>.npy, X_test_<bb>.npy
  data/features/y_train.npy, y_val.npy, y_test.npy   (shared across backbones)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
# --cpu forces CPU BEFORE any op initialises Metal (tensorflow-metal lacks ops for
# EfficientNetV2/ConvNeXt: V2S hangs, ConvNeXt errors "could not find registered platform").
# Must run here, before build_feature_extractor is imported and before the first GPU op.
if "--cpu" in sys.argv:
    tf.config.set_visible_devices([], "GPU")
from tensorflow.keras import layers
from tensorflow.keras.utils import load_img, img_to_array

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CFG, ensure_dirs           # noqa: E402
from src.models.backbones import build_feature_extractor  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# augmentation applied to 0-255 float images before the backbone's own preprocessing
_AUG = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical", seed=CFG.seed),
    layers.RandomRotation(0.15, seed=CFG.seed),
    layers.RandomZoom(0.15, seed=CFG.seed),
    layers.RandomTranslation(0.1, 0.1, seed=CFG.seed),
    layers.RandomContrast(0.15, seed=CFG.seed),
], name="augment")


def _list_split(split: str) -> tuple[list[Path], np.ndarray]:
    files, labels = [], []
    for ci, cls in enumerate(CFG.lesion_classes):
        for f in sorted((CFG.data_proc / split / cls).glob("*")):
            if f.suffix.lower() in IMG_EXT:
                files.append(f)
                labels.append(ci)
    return files, np.asarray(labels, dtype=np.int64)


def _load_raw(files: list[Path]) -> np.ndarray:
    arr = np.empty((len(files), *CFG.img_shape), dtype=np.float32)
    for i, f in enumerate(files):
        arr[i] = img_to_array(load_img(f, target_size=CFG.img_size))
    return arr


def _extract(extractor, x: np.ndarray) -> np.ndarray:
    return extractor.predict(x, batch_size=CFG.batch_size, verbose=0)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="*", default=CFG.backbones,
                    help="which backbones to cache features for")
    ap.add_argument("--cpu", action="store_true",
                    help="force CPU (tensorflow-metal lacks ops for EfficientNetV2/ConvNeXt)")
    args = ap.parse_args()
    backbones = args.backbones
    if args.cpu:
        print("[i] running on CPU (GPU disabled at import for Metal-incompatible backbones)")
    ensure_dirs()
    CFG.features.mkdir(parents=True, exist_ok=True)

    splits = {s: _list_split(s) for s in ("train", "val", "test")}
    for s, (f, y) in splits.items():
        print(f"[i] {s}: {len(f)} images")

    # shared labels (train labels tiled by the augmentation multiplier)
    K = CFG.aug_multiplier
    np.save(CFG.features / "y_train.npy", np.tile(splits["train"][1], K))
    np.save(CFG.features / "y_val.npy", splits["val"][1])
    np.save(CFG.features / "y_test.npy", splits["test"][1])

    raw = {s: _load_raw(files) for s, (files, _) in splits.items()}
    print(f"[i] raw images loaded (train {raw['train'].shape}, "
          f"val {raw['val'].shape}, test {raw['test'].shape})")

    for bb in backbones:
        print(f"\n[>] {bb}: building frozen extractor ...")
        extractor = build_feature_extractor(bb)

        # train — original + (K-1) augmented variants, stacked (labels tiled to match)
        chunks = [_extract(extractor, raw["train"])]
        for k in range(1, K):
            aug = _AUG(raw["train"], training=True).numpy()
            chunks.append(_extract(extractor, aug))
            print(f"    train aug pass {k}/{K-1} done")
        np.save(CFG.features / f"X_train_{bb}.npy", np.concatenate(chunks, axis=0))

        np.save(CFG.features / f"X_val_{bb}.npy", _extract(extractor, raw["val"]))
        np.save(CFG.features / f"X_test_{bb}.npy", _extract(extractor, raw["test"]))
        print(f"[✓] {bb}: features cached "
              f"(train {CFG.aug_multiplier}× = {len(splits['train'][0]) * K})")
        del extractor
        tf.keras.backend.clear_session()

    print(f"\n[✓] all features in {CFG.features}\n    next: python -m src.models.train_base")


if __name__ == "__main__":
    main()
