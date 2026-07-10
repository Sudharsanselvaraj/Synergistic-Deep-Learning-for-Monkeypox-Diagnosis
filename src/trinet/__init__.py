"""Tri-Net v2 — reproducible deep-learning framework for Mpox skin-lesion diagnosis.

A leakage-free benchmark and library: modern CNN backbones, learned feature fusion, an honest
evaluation suite (metrics, diversity, Grad-CAM), and a symptom-based screening module.

Quick start:
    from trinet.config import CFG
    from trinet.models.backbones import build_feature_extractor
    from trinet.models.fusion import build_concat_mlp
"""

from trinet.config import CFG

__version__ = "2.0.0"
__all__ = ["CFG", "__version__"]
