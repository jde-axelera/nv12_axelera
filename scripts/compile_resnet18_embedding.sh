#!/usr/bin/env bash
# Compile resnet18_embedding.onnx for Axelera Metis AIPU.
#
# Run from the evs repo root on the SSH machine:
#   bash scripts/compile_resnet18_embedding.sh
#
# Outputs compiled model.json to:
#   /home/ubuntu/1.6/voyager-sdk/build/resnet18-embedding/resnet18-embedding/1/
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SDK_VENV="/home/ubuntu/1.6/voyager-sdk/venv"
SDK_BUILD="/home/ubuntu/1.6/voyager-sdk/build"
MODELS_DIR="$REPO_ROOT/models"
OUT_DIR="$SDK_BUILD/resnet18-embedding/resnet18-embedding/1"
IMAGESET="$REPO_ROOT/output_images"   # existing JPEG outputs used as calibration
TRANSFORM="$SCRIPT_DIR/imagenet_transform.py"
ONNX="$MODELS_DIR/resnet18_embedding.onnx"

source "$SDK_VENV/bin/activate"

# Step 1 — export headless ONNX if not already present
mkdir -p "$MODELS_DIR"
if [ ! -f "$ONNX" ]; then
    echo "[1/2] Exporting headless ResNet18 ONNX..."
    python "$SCRIPT_DIR/export_resnet18_headless.py" "$ONNX"
else
    echo "[1/2] ONNX already exists: $ONNX"
fi

# Step 2 — compile for Metis AIPU
echo "[2/2] Compiling with axcompile..."
axcompile \
    --input         "$ONNX" \
    --transform     "$TRANSFORM" \
    --imageset      "$IMAGESET" \
    --output        "$OUT_DIR" \
    --overwrite \
    --color-format  RGB \
    --input-shape   1,3,224,224 \
    --log-level     INFO

echo
echo "Done.  Compiled model: $OUT_DIR/model.json"
echo "Run with:"
echo "  export LD_LIBRARY_PATH=/opt/axelera/runtime-1.6.0-1/lib:\$LD_LIBRARY_PATH"
echo "  ./build/feature_extraction \\"
echo "    --model=$OUT_DIR/model.json \\"
echo "    input_images/dog_bike_768x576.rgba \\"
echo "    --size=768x576 --warmup=5 --runs=30 \\"
echo "    --output-emb=aipu_embedding.txt"
