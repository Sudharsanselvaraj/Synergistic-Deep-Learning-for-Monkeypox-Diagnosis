"""Single-image inference with the Tri-Net v2 champion (Concat+MLP fusion).

Runs a lesion image through the frozen backbones, fuses their features with the trained fusion
model, and returns the 14-class diagnosis plus the binary Mpox-screening decision. Runs on CPU
(some backbones lack Metal ops), which is instant for a single image.

    from trinet.inference.predict import predict_image
    result = predict_image("lesion.jpg")
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from tensorflow.keras.utils import img_to_array, load_img

from trinet.config import CFG
from trinet.models.backbones import build_feature_extractor
from trinet.models.fusion import load_fusion


def _force_cpu() -> None:
    """Run inference on CPU (single image is instant; sidesteps Metal op gaps).

    Done lazily here rather than at import so that importing this module never has the global
    side effect of disabling the GPU for the whole process. Opt out with ``TRINET_USE_GPU=1``.
    """
    if os.environ.get("TRINET_USE_GPU"):
        return
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass  # GPU already initialised elsewhere; op placement still falls back gracefully


@lru_cache(maxsize=1)
def _load():
    _force_cpu()
    meta = CFG.model_dir / "fusion_meta.json"
    if not meta.exists():
        raise FileNotFoundError(
            f"No fusion champion in {CFG.model_dir}. Train it (`trinet fusion`) or download "
            "released weights (fusion_meta.json + fusion.keras or head_*.keras) into "
            "outputs/checkpoints/."
        )
    # The Mean champion is served from the per-backbone heads (fusion.keras is intentionally
    # absent); every learned strategy is served from fusion.keras. Check the file the deployed
    # strategy actually needs, so a valid Mean deployment doesn't look like a missing checkpoint.
    strategy = json.loads(meta.read_text()).get("strategy")
    if strategy != "Mean" and not (CFG.model_dir / "fusion.keras").exists():
        raise FileNotFoundError(
            f"fusion_meta.json reports strategy={strategy!r} but fusion.keras is missing in "
            f"{CFG.model_dir}. Re-run `trinet fusion`."
        )
    fusion, bbs = load_fusion()
    extractors = {bb: build_feature_extractor(bb) for bb in bbs}
    return extractors, fusion, bbs


def predict_image(path: str | Path, top_k: int = 3) -> dict:
    """Return {diagnosis, confidence, top_k, mpox_screening} for one image."""
    extractors, fusion, bbs = _load()
    img = img_to_array(load_img(path, target_size=CFG.img_size))[None]
    feats = [extractors[bb].predict(img, verbose=0) for bb in bbs]
    prob = fusion.predict(feats, verbose=0)[0]

    order = prob.argsort()[::-1]
    mpox_i = CFG.lesion_classes.index(CFG.mpox_class)
    return {
        "diagnosis": CFG.lesion_classes[int(order[0])],
        "confidence": round(float(prob[order[0]]), 4),
        "top_k": [(CFG.lesion_classes[int(i)], round(float(prob[i]), 4)) for i in order[:top_k]],
        "mpox_screening": {
            "label": "Mpox" if int(order[0]) == mpox_i else "Non-Mpox",
            "mpox_probability": round(float(prob[mpox_i]), 4),
        },
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Predict the disease class for a lesion image.")
    ap.add_argument("image", help="path to a skin-lesion image")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()
    r = predict_image(args.image, top_k=args.top_k)
    print(f"Diagnosis : {r['diagnosis']}  ({100 * r['confidence']:.1f}%)")
    print(
        f"Mpox screen: {r['mpox_screening']['label']}  "
        f"(p={r['mpox_screening']['mpox_probability']:.3f})"
    )
    print("Top-k:")
    for name, p in r["top_k"]:
        print(f"  {name:<22} {100 * p:5.1f}%")


if __name__ == "__main__":
    main()
