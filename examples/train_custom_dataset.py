"""Train a Tri-Net head on your own class-folder dataset.

Lay out your data as class folders Keras can read::

    <CFG.data_proc>/
      train/<class_name>/*.jpg
      val/<class_name>/*.jpg
      test/<class_name>/*.jpg

Configure the run by editing ``src/trinet/config.py`` (NOT by reassigning ``CFG`` at runtime):
set ``lesion_classes`` to your class names and, if needed, ``data_proc`` to your data root.
``n_classes`` is derived from ``lesion_classes`` at import, and every head's output size defaults
to it, so editing the source is the supported way to re-target the pipeline — a late
``CFG.lesion_classes = [...]`` will not resize models already built. Then cache features and train:

    trinet features --backbones ConvNeXtTiny --cpu
    trinet train    --backbones ConvNeXtTiny
    trinet fusion   --backbones ConvNeXtTiny InceptionResNetV2 DenseNet201

This example just shows the programmatic entry points; the CLI is the usual path.
"""

from trinet.config import CFG
from trinet.models import fusion
from trinet.training import base, features


def main() -> None:
    print(f"Classes ({CFG.n_classes}): {CFG.lesion_classes}")
    print(f"Data at: {CFG.data_proc}")
    features.main()  # cache frozen-backbone features
    base.main()  # train per-backbone heads
    fusion.main()  # fusion ablation + champion


if __name__ == "__main__":
    main()
