#!/usr/bin/env bash
# Regenerate all evaluation artifacts from trained models.
set -euo pipefail
trinet evaluate --champion
trinet diversity
trinet gradcam
python experiments/binary_screening.py
