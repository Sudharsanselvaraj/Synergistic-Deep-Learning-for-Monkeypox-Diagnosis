"""Backbone + classification-head builders for the Tri-Net ensemble.

Three ImageNet-pretrained backbones — EfficientNetB4, InceptionResNetV2, DenseNet201 — each
with its OWN `preprocess_input` (they disagree: EfficientNet wants raw 0-255, InceptionResNetV2
wants [-1,1], DenseNet wants caffe-style). Getting this wrong silently wrecks accuracy, so the
preprocessing is baked into each model here.

Two things are built from the same spec:
  - `build_feature_extractor(name)` : input -> preprocess -> backbone -> GAP  (frozen; for caching)
  - `build_head(input_dim)`         : the trainable classifier on top of cached features
  - `build_full_model(name)`        : end-to-end (used for fine-tuning + Grad-CAM)
"""
from __future__ import annotations
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import applications as apps

from trinet.config import CFG  # noqa: E402

# name -> (application_ctor, preprocess_input, pooled_feature_dim)
_SPEC = {
    # Phase-1 backbones (the paper's three)
    "EfficientNetB4": (apps.EfficientNetB4, apps.efficientnet.preprocess_input, 1792),
    "InceptionResNetV2": (apps.InceptionResNetV2, apps.inception_resnet_v2.preprocess_input, 1536),
    "DenseNet201": (apps.DenseNet201, apps.densenet.preprocess_input, 1920),
    # Phase-2 stronger modern backbones (frozen-feature track)
    "EfficientNetV2S": (apps.EfficientNetV2S, apps.efficientnet_v2.preprocess_input, 1280),
    "ConvNeXtTiny": (apps.ConvNeXtTiny, apps.convnext.preprocess_input, 768),
}


def feature_dim(name: str) -> int:
    return _SPEC[name][2]


def build_backbone(name: str, trainable: bool = False) -> Model:
    ctor, _, _ = _SPEC[name]
    backbone = ctor(weights="imagenet", include_top=False, input_shape=CFG.img_shape)
    backbone.trainable = trainable
    return backbone


def build_feature_extractor(name: str) -> Model:
    """Frozen input -> preprocess -> backbone -> global-avg-pool. Used to cache features."""
    ctor, preprocess, _ = _SPEC[name]
    inp = layers.Input(shape=CFG.img_shape)
    x = preprocess(inp)
    backbone = ctor(weights="imagenet", include_top=False, input_shape=CFG.img_shape)
    backbone.trainable = False
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    return Model(inp, x, name=f"{name}_feat")


def build_head(input_dim: int, n_classes: int = CFG.n_classes) -> Model:
    """Trainable classifier head applied to a pooled feature vector.

    GAP is already done during caching, so this is Dense(256)->BN->Dense(128)->Dropout->Softmax,
    matching the paper's 'improved custom layers' (Fig. 2).
    """
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(CFG.dropout)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return Model(inp, out, name="head")


def build_full_model(name: str, n_classes: int = CFG.n_classes,
                     finetune_unfreeze: int = 0) -> Model:
    """End-to-end model (preprocess -> backbone -> head). For fine-tuning and Grad-CAM.

    finetune_unfreeze>0 unfreezes the last N backbone layers.
    """
    ctor, preprocess, dim = _SPEC[name]
    inp = layers.Input(shape=CFG.img_shape)
    x = preprocess(inp)
    backbone = ctor(weights="imagenet", include_top=False, input_shape=CFG.img_shape)
    backbone.trainable = finetune_unfreeze > 0
    if finetune_unfreeze > 0:
        for layer in backbone.layers[:-finetune_unfreeze]:
            layer.trainable = False
    x = backbone(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(CFG.dropout)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return Model(inp, out, name=f"{name}_full")


if __name__ == "__main__":
    for n in CFG.backbones:
        fe = build_feature_extractor(n)
        print(f"{n}: feature_dim={feature_dim(n)}, params={fe.count_params():,}")
