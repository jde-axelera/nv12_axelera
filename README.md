# Axelera EVS — NV12 Detection + Embedding Extraction

Two C++ inference pipelines for the Axelera Metis AIPU:

| Binary | Model | Input | Output |
|---|---|---|---|
| `yolov5s_nv12` | YOLOv5s COCO | Raw NV12 frames or JPEG/PNG | Bounding boxes + annotated JPEG |
| `feature_extraction` | Any embedding model (e.g. ResNet18) | RGBA frames or JPEG/PNG | Float embedding vector |

Both use a **double-buffered DMA-BUF pipeline**: two pinned input buffers alternate each frame so CPU preprocessing overlaps AIPU inference.

---

## Pipeline — yolov5s_nv12

```
Camera / File
    │
    ▼
NV12 Buffer (raw YUV420 semi-planar)
    │
    ├─── [Thread: preprocess buf[nxt]] ──────────────────────────┐
    │     NV12→BGR, resize+letterbox, quantise → int8 NHWC       │ (~2 ms)
    │                                                             │
    ▼   [Inference — AIPU, buf[cur]]  (~6 ms)                    │
    │    YOLOv5s int8 (sigmoid fused on-chip)                     │
    ▼                                                             │
3 × Output Tensors (host memory, int8) ◄────────────────────────┘
    │                  (thread done; buf[nxt] ready for next iter)
    ▼   [Decode + NMS — CPU]
    │    decode_head × 3 strides → Det list → greedy NMS (IoU 0.45)
    ▼
Annotated JPEG
```

## Pipeline — feature_extraction

```
RGBA / JPEG / PNG
    │
    ├─── [Thread: preprocess buf[nxt]] ──────────────────────────┐
    │     RGBA→BGR, resize to 224×224, pixel/255 → int8 NHWC     │ (~0.6 ms)
    │                                                             │
    ▼   [Inference — AIPU, buf[cur]]  (~1.2 ms)                  │
    │    ResNet18 (or any embedding model)                         │
    ▼                                                             │
NHWC Output [1, H, W, C] (host memory, int8) ◄──────────────────┘
    │
    ▼   [Global avg pool + dequantise — CPU]
    │    average over H×W spatial dims → float[C]
    ▼
Embedding vector  (512-dim for ResNet18)
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Axelera Voyager SDK 1.6 | `source venv/bin/activate` |
| axruntime pkg-config | `$AXELERA_RUNTIME_DIR/lib/pkgconfig` |
| OpenCV 4 | `pkg-config --modversion opencv4` |
| CMake >= 3.12, Ninja | `apt install cmake ninja-build` |
| GCC >= 11 (C++20) | `gcc --version` |

---

## Download models

```bash
source $VOYAGER_SDK/venv/bin/activate

# YOLOv5s COCO
axdownloadmodel --model yolov5s-v7-coco
# → build/yolov5s-v7-coco/yolov5s-v7-coco/1/model.json

# ResNet18 embedding (compile from ONNX — see scripts/)
bash scripts/compile_resnet18_embedding.sh
# → $VOYAGER_SDK/build/resnet18-embedding/resnet18-embedding/1/compiled_model/model.json
```

---

## Build

```bash
cd /path/to/nv12_axelera

source $VOYAGER_SDK/venv/bin/activate
export AXELERA_RUNTIME_DIR=$(python -c 'from axelera.runtime.configs import runtime_dir; print(runtime_dir)')

PKG_CONFIG_PATH=$AXELERA_RUNTIME_DIR/lib/pkgconfig \
    cmake -Bbuild -GNinja . -DCMAKE_BUILD_TYPE=Release

ninja -C build          # builds both yolov5s_nv12 and feature_extraction
```

---

## Usage — yolov5s_nv12

```
./build/yolov5s_nv12  model.json  [image]  [labels.names]
                     [--size=WxH]  [--output=path.jpg]
                     [--warmup=N]  [--runs=N]
```

| Flag | Default | Description |
|---|---|---|
| `model.json` | required | Axelera model descriptor |
| `image` | synthetic grey | JPEG/PNG **or** raw `.nv12`/`.yuv` |
| `labels.names` | (no labels) | One class name per line |
| `--size=WxH` | 1920×1080 or filename-parsed | NV12/YUV frame dimensions |
| `--output=path` | `<image>_detections.jpg` | Output JPEG path |
| `--warmup=N` | 5 | Warmup iterations |
| `--runs=N` | 20 | Benchmark iterations |

### Example

```bash
export LD_LIBRARY_PATH=/opt/axelera/runtime-1.6.0-1/lib:$LD_LIBRARY_PATH

./build/yolov5s_nv12 \
    $VOYAGER_SDK/build/yolov5s-v7-coco/yolov5s-v7-coco/1/model.json \
    input_images/dog_bike_768x576.nv12 \
    $VOYAGER_SDK/ax_datasets/labels/coco.names \
    --size=768x576 --warmup=5 --runs=30 \
    --output=output_images/dog_bike_result.jpg
```

Expected: **dog 89 %**, **bicycle 45 %**, **car 65 %**

### Latency (768×576 NV12, Metis SDK 1.6)

```
+--------------------------------------------------------------------+
| LATENCY BREAKDOWN  (30 runs, DMA-BUF input, double-buffered pipeline)
+------------------+----------+----------+----------+----------+
| Section          |   avg ms |   min ms |   max ms |   p95 ms |
+------------------+----------+----------+----------+----------+
| Preprocess NV12  |    1.925 |    1.843 |    2.289 |    2.027 |
| Inference (AIPU) |    6.072 |    5.651 |    6.423 |    6.360 |
| Decode + NMS     |    0.060 |    0.055 |    0.076 |    0.074 |
| Frame wall time  |    6.153 |    5.725 |    6.549 |    6.437 |
+------------------+----------+----------+----------+----------+
| Throughput (pipelined):  162.5 FPS
| Sequential latency:      8.057 ms  (pre+inf+dec, non-overlapped)
+--------------------------------------------------------------------+
```

---

## Usage — feature_extraction

```
./build/feature_extraction  --model=model.json  [image]
                            [--size=WxH]  [--output-emb=emb.txt]
                            [--warmup=N]  [--runs=N]
```

| Flag | Default | Description |
|---|---|---|
| `--model=model.json` | required | Axelera model descriptor |
| `image` | synthetic grey | JPEG/PNG **or** raw `.rgba` |
| `--size=WxH` | 640×640 or filename-parsed | RGBA frame dimensions |
| `--output-emb=path` | (not saved) | Save float embedding for ONNX comparison |
| `--warmup=N` | 5 | Warmup iterations |
| `--runs=N` | 30 | Benchmark iterations |

### Example

```bash
export LD_LIBRARY_PATH=/opt/axelera/runtime-1.6.0-1/lib:$LD_LIBRARY_PATH
MODEL=$VOYAGER_SDK/build/resnet18-embedding/resnet18-embedding/1/compiled_model/model.json

./build/feature_extraction \
    --model=$MODEL \
    input_images/dog_bike_768x576.rgba \
    --size=768x576 --warmup=5 --runs=30 \
    --output-emb=aipu_embedding.txt
```

### Verify against ONNX reference

```bash
source $VOYAGER_SDK/venv/bin/activate

python scripts/verify_onnx_embedding.py \
    --onnx models/resnet18_embedding.onnx \
    --image input_images/dog_bike_768x576.rgba \
    --size 768x576 \
    --aipu aipu_embedding.txt
```

Expected output:
```
[ONNX]  dim=512  norm=29.64
[AIPU]  dim=512  norm=29.45
[COMPARE]  cosine_similarity=0.9983  L2_error=1.73
  ✓ Excellent match (cosine > 0.99)
```

### Latency (ResNet18, 768×576 RGBA input, Metis SDK 1.6)

```
+--------------------------------------------------------------------+
| LATENCY BREAKDOWN  (30 runs, DMA-BUF input, double-buffered pipeline)
| Embedding dim: 512    AIPU: 1 core(s)
+------------------+----------+----------+----------+----------+
| Section          |   avg ms |   min ms |   max ms |   p95 ms |
+------------------+----------+----------+----------+----------+
| Preprocess       |    0.656 |    0.375 |    0.873 |    0.851 |
| Inference (AIPU) |    1.219 |    1.115 |    1.835 |    1.512 |
| Frame wall time  |    1.243 |    1.145 |    1.859 |    1.612 |
+------------------+----------+----------+----------+----------+
| Throughput (pipelined):  804.3 FPS
| Sequential latency:      1.875 ms  (pre+inf, non-overlapped)
+--------------------------------------------------------------------+
```

---

## Source layout

```
/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── annotate.hpp       — save_annotated(): draw boxes, save JPEG
│   ├── dmabuf.hpp         — DmaBuf: alloc/release via /dev/dma_heap/system
│   ├── preprocess.hpp     — nv12_to_tensor, rgba_to_tensor, rgba_to_tensor_imagenet
│   ├── timer.hpp          — SectionTimer + ScopeTimer RAII helpers
│   └── yolo_decode.hpp    — Det struct, decode_head(), nms()
├── src/
│   ├── main.cpp                   — yolov5s_nv12: NV12 detection pipeline
│   ├── main_feature_extraction.cpp — feature_extraction: generic embedding runner
│   ├── annotate.cpp
│   ├── preprocess.cpp             — TensorLayout helper; NV12/RGBA→int8 NHWC
│   └── yolo_decode.cpp
├── scripts/
│   ├── compile_resnet18_embedding.sh — end-to-end: export ONNX + axcompile
│   ├── export_resnet18_headless.py   — remove FC head, export [1,512] ONNX
│   ├── pixel255_transform.py         — axcompile calibration transform (pixel/255)
│   ├── imagenet_transform.py         — axcompile calibration transform (ImageNet norm)
│   └── verify_onnx_embedding.py      — compare AIPU embedding vs ONNX reference
├── input_images/
│   ├── dog_bike_768x576.nv12
│   ├── dog_bike_768x576.rgba
│   └── tulips_nv12_prog_qcif.yuv
└── output_images/
    └── *.jpg
```

---

## Implementation notes

### Double-buffer DMA-BUF pipeline
Two DMA-BUF allocations (`buf[0]`, `buf[1]`) are made at startup.  The benchmark
loop alternates using `cur = i & 1` / `nxt = cur ^ 1`.  A `std::async` thread
writes the next preprocessed tensor into `buf[nxt]` while `axr_run_model_instance`
blocks on `buf[cur]`.  Because preprocess finishes well before inference, the
`future.get()` call never stalls and wall time approaches bare-metal AIPU time.

### DMA-BUF zero-copy input
When `/dev/dma_heap/system` is available the AIPU reads directly from the
DMA-BUF allocation (`input_dmabuf=1`).  Falls back to host memory with a warning.

### Why `output_dmabuf=0`
Metis MMIO outputs are not DMA-BUF capable.  Attempting `output_dmabuf=1` fails
at runtime.

### Post-sigmoid outputs (yolov5s_nv12)
YOLOv5s is compiled with sigmoid fused into the AIPU graph.  `decode_head`
dequantises raw int8 values directly — **do not apply sigmoid again**.

### Global average pool (feature_extraction)
axcompile splits avgpool+flatten to a CPU postprocess graph, so the AIPU outputs
`[1, H, W, C]` (e.g. `[1, 7, 7, 512]` for ResNet18), not a flat `[1, 512]`.
The binary applies global average pooling over the H×W spatial dimensions in C++
after dequantisation.

### Preprocessing and calibration (feature_extraction)
`rgba_to_tensor` (pixel/255, no mean/std) matches the default axcompile
calibration range: `scale ≈ 1/255`, `zp = -128`.  If you recompile with
`--transform imagenet_transform.py`, switch to `rgba_to_tensor_imagenet`.

`pixel255_transform.py` bundles two workarounds for axcompile's multiprocessing:
1. Self-registration in `sys.modules` — fixes the "not the same object" pickle error.
2. cloudpickle `ForkingPickler` patch — fixes axcompile's own internal lambda error.
Run with `PYTHONPATH=scripts/` so the spawned worker can import the module by name.

### NV12 dimension requirements
`cv::COLOR_YUV2BGR_NV12` requires even width and height.  `bgr_to_nv12()` rounds
up odd dimensions before conversion.
