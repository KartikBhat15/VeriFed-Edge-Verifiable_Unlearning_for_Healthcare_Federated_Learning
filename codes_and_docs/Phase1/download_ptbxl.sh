#!/bin/bash
# Run this in a Colab or Kaggle notebook cell (prefix with !) or a Colab terminal.
# Downloads PTB-XL v1.0.3 (~1.7GB) from PhysioNet.
# Docs: https://physionet.org/content/ptb-xl/

set -e

PTBXL_DIR="ptbxl"
mkdir -p "$PTBXL_DIR"
cd "$PTBXL_DIR"

echo "Downloading PTB-XL (this takes a few minutes)..."
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/

echo "Done. Point --ptbxl_root at:"
echo "  $(pwd)/physionet.org/files/ptb-xl/1.0.3"
