"""Grad-CAM heatmaps for the Tri-Net base models (paper Fig. 10 — done for real).

Heads were trained on cached pooled features, so here we reattach each trained head onto its
frozen backbone's last conv feature map to get an end-to-end model we can differentiate
through. We render a grid: sample images x [original | EfficientNetB4 | InceptionResNetV2 |
DenseNet201], each overlaid with the class-activation heatmap for the predicted class.

Output: results/figures/gradcam.png
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
# tensorflow-metal can't compile the Grad-CAM gradient graph ("MLIR pass manager failed"),
# so run this on CPU — it's only a handful of images and gradients, instant either way.
tf.config.set_visible_devices([], "GPU")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CFG                                  # noqa: E402
from src.models.backbones import build_backbone, _SPEC   # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _last_conv_name(backbone) -> str:
    for layer in reversed(backbone.layers):
        if len(layer.output.shape) == 4:
            return layer.name
    raise RuntimeError("no 4D conv layer found")


def _build_gradcam_model(bb: str):
    """frozen backbone last-conv -> GAP -> trained head; returns (grad_model, preprocess).

    We rebuild the head with fresh uniquely-named layers (reusing the loaded head's layer
    objects collides with the backbone's BatchNorm names) and copy the trained weights in.
    """
    _, preprocess, _ = _SPEC[bb]
    backbone = build_backbone(bb, trainable=False)
    last_conv = backbone.get_layer(_last_conv_name(backbone)).output
    g = layers.GlobalAveragePooling2D()(last_conv)

    head = load_model(CFG.model_dir / f"head_{bb}.keras")
    hw = [l.get_weights() for l in head.layers if l.get_weights()]  # Dense256,BN,Dense128,softmax
    d256 = layers.Dense(256, activation="relu", name="gc_d256")
    bn = layers.BatchNormalization(name="gc_bn")
    d128 = layers.Dense(128, activation="relu", name="gc_d128")
    out = layers.Dense(CFG.n_classes, activation="softmax", name="gc_out")
    x = out(d128(bn(d256(g))))
    model = Model(backbone.input, [last_conv, x])
    for layer, w in zip((d256, bn, d128, out), hw):
        layer.set_weights(w)
    return model, preprocess


def _heatmap(grad_model, preprocess, raw_img: np.ndarray) -> tuple[np.ndarray, int]:
    x = preprocess(raw_img[None].astype("float32"))
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        cls = int(tf.argmax(preds[0]))
        channel = preds[:, cls]
    grads = tape.gradient(channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    hm = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
    hm = tf.nn.relu(hm)
    hm = hm / (tf.reduce_max(hm) + 1e-8)
    return hm.numpy(), cls


def _overlay(raw_img: np.ndarray, hm: np.ndarray, alpha=0.4) -> np.ndarray:
    hm_r = tf.image.resize(hm[..., None], CFG.img_size).numpy()[..., 0]
    color = cm.jet(hm_r)[..., :3]
    return np.clip((raw_img / 255.0) * (1 - alpha) + color * alpha, 0, 1)


def _sample_images(n_per_class=1, classes=("Mpox", "Melanoma", "HFMD")) -> list[Path]:
    out = []
    for c in classes:
        files = sorted((CFG.data_proc / "test" / c).glob("*"))
        out += [f for f in files if f.suffix.lower() in IMG_EXT][:n_per_class]
    return out


def main() -> None:
    samples = _sample_images()
    grad_models = {bb: _build_gradcam_model(bb) for bb in CFG.backbones}

    ncol = 1 + len(CFG.backbones)
    fig, axes = plt.subplots(len(samples), ncol, figsize=(2.4 * ncol, 2.4 * len(samples)))
    if len(samples) == 1:
        axes = axes[None, :]
    for r, path in enumerate(samples):
        raw = img_to_array(load_img(path, target_size=CFG.img_size))
        axes[r, 0].imshow(raw.astype("uint8"))
        axes[r, 0].set_ylabel(path.parent.name, fontsize=9)
        axes[r, 0].set_title("input" if r == 0 else "")
        for c, bb in enumerate(CFG.backbones, start=1):
            gm, prep = grad_models[bb]
            hm, cls = _heatmap(gm, prep, raw)
            axes[r, c].imshow(_overlay(raw, hm))
            axes[r, c].set_title(bb if r == 0 else "", fontsize=8)
            axes[r, c].text(4, 18, CFG.lesion_classes[cls], color="white", fontsize=7,
                            bbox=dict(facecolor="black", alpha=0.5, pad=1))
        for c in range(ncol):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    fig.suptitle("Grad-CAM — Tri-Net base models", fontsize=11)
    fig.tight_layout()
    fig.savefig(CFG.fig_dir / "gradcam.png")
    plt.close(fig)
    print(f"[✓] Grad-CAM -> {CFG.fig_dir/'gradcam.png'}")


if __name__ == "__main__":
    main()
