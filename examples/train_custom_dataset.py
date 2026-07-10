"""Train a Tri-Net head on your own class-folder dataset.

Point CFG.data_proc at a directory laid out as::

    my_data/
      train/<class_name>/*.jpg
      val/<class_name>/*.jpg
      test/<class_name>/*.jpg

and set CFG.lesion_classes to your class names. Then cache features and train:

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
