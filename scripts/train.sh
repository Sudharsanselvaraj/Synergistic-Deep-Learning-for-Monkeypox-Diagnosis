#!/usr/bin/env bash
# Train base heads and the fusion champion.
set -euo pipefail
trinet train --backbones ConvNeXtTiny InceptionResNetV2 DenseNet201
trinet fusion
