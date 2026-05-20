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
IMAGESET="$REPO_ROOT/output_images"   # JPEG calibration images
TRANSFORM="$SCRIPT_DIR/pixel255_transform.py"   # pixel/255 only — matches C++ rgba_to_tensor
ONNX="$MODELS_DIR/resnet18_embedding.onnx"

# Preprocessing: pixel/255 (no ImageNet mean/std).  Matches C++ rgba_to_tensor.
# The transform file self-registers in sys.modules to work around axcompile's
# multiprocessing pickle issue (spec_from_file_location without sys.modules registration).

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
NIMAGES=$(ls "$IMAGESET"/*.jpg "$IMAGESET"/*.jpeg "$IMAGESET"/*.png 2>/dev/null | wc -l)
echo "[2/2] Compiling with axcompile (${NIMAGES} calibration images, pixel/255)..."
# PYTHONPATH must include scripts/ so the spawned worker process can import
# pixel255_transform by name when deserialising the cloudpickle payload.
PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}" axcompile \
    --input         "$ONNX" \
    --transform     "$TRANSFORM" \
    --imageset      "$IMAGESET" \
    --dataset-len   "$NIMAGES" \
    --output        "$OUT_DIR" \
    --overwrite \
    --color-format  RGB \
    --input-shape   1,3,224,224 \
    --log-level     INFO

echo
MODEL_JSON="$OUT_DIR/compiled_model/model.json"
echo "Done.  Compiled model: $MODEL_JSON"
echo "Run with:"
echo "  export LD_LIBRARY_PATH=/opt/axelera/runtime-1.6.0-1/lib:\$LD_LIBRARY_PATH"
echo "  ./build/feature_extraction \\"
echo "    --model=$MODEL_JSON \\"
echo "    input_images/dog_bike_768x576.rgba \\"
echo "    --size=768x576 --warmup=5 --runs=30 \\"
echo "    --output-emb=aipu_embedding.txt"
echo ""
echo "Verify vs ONNX (pixel/255, no ImageNet norm):"
echo "  source /home/ubuntu/1.6/voyager-sdk/venv/bin/activate"
echo "  python scripts/verify_onnx_embedding.py \\"
echo "    --onnx models/resnet18_embedding.onnx \\"
echo "    --image input_images/dog_bike_768x576.rgba \\"
echo "    --size 768x576 \\"
echo "    --aipu aipu_embedding.txt"
