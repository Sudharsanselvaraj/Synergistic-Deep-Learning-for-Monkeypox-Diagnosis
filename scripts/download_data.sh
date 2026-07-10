#!/usr/bin/env bash
# Download datasets from Kaggle and build the split. Needs ~/.kaggle credentials.
set -euo pipefail
trinet download
trinet prepare
