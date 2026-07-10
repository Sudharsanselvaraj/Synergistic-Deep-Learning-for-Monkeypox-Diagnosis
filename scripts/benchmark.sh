#!/usr/bin/env bash
# Full image pipeline end to end. Assumes `pip install -e .` and downloaded data.
set -euo pipefail
trinet prepare
trinet features
trinet features --backbones ConvNeXtTiny EfficientNetV2S --cpu   # modern backbones need CPU
trinet train --backbones ConvNeXtTiny InceptionResNetV2 DenseNet201
trinet ensemble
trinet fusion
trinet evaluate --champion
trinet cross-val
trinet diversity
