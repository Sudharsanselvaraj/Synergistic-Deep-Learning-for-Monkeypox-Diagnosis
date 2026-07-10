#!/usr/bin/env bash
# Create the Tri-Net v0.9.0 GitHub Release with the pretrained checkpoint attached,
# so `trinet predict` works for anyone without training.
#
# Prerequisites:
#   1. A trained checkpoint  ->  run `trinet fusion` first
#   2. GitHub CLI, authenticated  ->  `brew install gh && gh auth login`
#
# Usage:  bash scripts/make_release.sh
set -euo pipefail

VERSION="v0.9.0"
CKPT="outputs/checkpoints"
BUNDLE="/tmp/trinet-${VERSION}-weights.zip"

[ -f "${CKPT}/fusion_meta.json" ] || { echo "No checkpoint found — run 'trinet fusion' first."; exit 1; }

# Bundle exactly what inference needs: the meta + the deployed model's files.
# (Mean strategy uses head_*.keras; learned strategies use fusion.keras.)
rm -f "${BUNDLE}"
( cd "${CKPT}" && zip -q "${BUNDLE}" fusion_meta.json $(ls fusion.keras head_*.keras 2>/dev/null) )
echo "Bundled checkpoint -> ${BUNDLE}"
echo "Strategy: $(python -c 'import json;print(json.load(open("'"${CKPT}"'/fusion_meta.json"))["strategy"])')"

gh release create "${VERSION}" "${BUNDLE}" \
  --title "Tri-Net v2 — Initial Open-Source Release (${VERSION})" \
  --notes-file docs/release_notes.md

echo
echo "Release ${VERSION} created. Users can now:"
echo "  1. Download trinet-${VERSION}-weights.zip from the release"
echo "  2. Unzip it into outputs/checkpoints/"
echo "  3. Run: trinet predict lesion.jpg"
