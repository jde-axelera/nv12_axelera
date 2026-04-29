# YOLOv5s — NV12 Object Detection on Axelera Metis

Real-time object detection pipeline that accepts raw NV12 camera frames (or
JPEG/PNG via NV12 simulation), preprocesses them on the CPU, runs YOLOv5s on
the Axelera Metis AIPU, decodes the three detection heads, applies NMS, and
saves an annotated JPEG.  Per-section latency breakdown is printed after each
benchmark run.

---

## Pipeline

```
Camera / File
    |
    v
NV12 Buffer (raw YUV420 semi-planar)
    |
    +-------[Thread: preprocess next frame into buf[nxt]]
    |                                                    |
    v  [Preprocess — CPU, buf[cur]]                      | (overlaps)
    |   cvtColor NV12→BGR, resize letterbox,             |
    |   quantise → int8 NHWC                             |
    v                                                    |
DMA-BUF Input Tensor  (zero-copy to AIPU)                |
    |                                                    |
    v  [Inference — AIPU, buf[cur]] (~6 ms)              |
    |   YOLOv5s int8 (post-sigmoid outputs on chip)      |
    v                                                    |
3 x Output Tensors  (host memory, int8)  <---------------+
    |                   (thread done, buf[nxt] ready for next iter)
    v  [Decode + NMS — CPU]
    |   decode_head × 3 strides  →  candidate Det list
    |   greedy per-class NMS (IoU 0.45)
    v
Annotated JPEG  (bounding boxes + labels)
```

The benchmark uses a **double-buffer pipeline**: two DMA-BUF input allocations
(`buf[0]` and `buf[1]`) alternate each frame.  A `std::async` thread preprocesses
frame N+1 into `buf[nxt]` while the AIPU runs frame N on `buf[cur]`.  Because
preprocess (~2 ms) completes well before inference (~6 ms), the CPU thread is
already done by the time the main thread calls `future.get()` — so the per-frame
wall time approaches the inference time alone.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Axelera Voyager SDK 1.6 | `source venv/bin/activate` |
| axruntime pkg-config | `$AXELERA_RUNTIME_DIR/lib/pkgconfig` |
| OpenCV 4 | `pkg-config --modversion opencv4` |
| CMake >= 3.12, Ninja | `apt install cmake ninja-build` |
| GCC >= 11 (C++20) | `gcc --version` |
| YOLOv5s COCO model | see Download section |

---

## Download the model

```bash
source $VOYAGER_SDK/venv/bin/activate
axdownloadmodel --model yolov5s-v7-coco
# model lands in:  build/yolov5s-v7-coco/yolov5s-v7-coco/1/model.json
```

---

## Build

```bash
cd /path/to/nv12_axelera

source $VOYAGER_SDK/venv/bin/activate
export AXELERA_RUNTIME_DIR=$(python -c 'from axelera.runtime.configs import runtime_dir; print(runtime_dir)')

PKG_CONFIG_PATH=$AXELERA_RUNTIME_DIR/lib/pkgconfig \
    cmake -Bbuild -GNinja . -DCMAKE_BUILD_TYPE=Release

ninja -C build
```

The binary is `build/yolov5s_nv12`.

---

## Usage

```
./build/yolov5s_nv12  model.json  [image]  [labels.names]
                     [--size=WxH]  [--output=path.jpg]
                     [--warmup=N]  [--runs=N]
```

| Flag | Default | Description |
|---|---|---|
| `model.json` | required | Axelera model descriptor |
| `image` | synthetic grey | JPEG / PNG **or** raw `.nv12` / `.yuv` |
| `labels.names` | (no labels) | One class name per line (COCO: 80 lines) |
| `--size=WxH` | 1920x1080 or filename-parsed | NV12/YUV frame dimensions |
| `--output=path` | `<image>_detections.jpg` | Output JPEG path |
| `--warmup=N` | 5 | Warmup iterations (not timed) |
| `--runs=N` | 20 | Benchmark iterations |

---

## Example commands

### Dog + bicycle (768 x 576 NV12)

```bash
export LD_LIBRARY_PATH=/opt/axelera/runtime-1.6.0-1/lib:$LD_LIBRARY_PATH

./build/yolov5s_nv12 \
    $VOYAGER_SDK/build/yolov5s-v7-coco/yolov5s-v7-coco/1/model.json \
    input_images/dog_bike_768x576.nv12 \
    $VOYAGER_SDK/ax_datasets/labels/coco.names \
    --size=768x576 --warmup=5 --runs=30 \
    --output=output_images/dog_bike_result.jpg
```

Expected detections: **dog 89 %**, **bicycle 45 %**, **car 65 %**

### Tulips (QCIF 176 x 144 NV12)

```bash
./build/yolov5s_nv12 \
    $VOYAGER_SDK/build/yolov5s-v7-coco/yolov5s-v7-coco/1/model.json \
    input_images/tulips_nv12_prog_qcif.yuv \
    $VOYAGER_SDK/ax_datasets/labels/coco.names \
    --size=176x144 --warmup=5 --runs=30 \
    --output=output_images/tulips_result.jpg
```

Expected detection: **vase 38 %**

---

## Per-section latency table

After benchmarking the tool prints a breakdown of each pipeline stage.  The
benchmark uses the double-buffer pipeline, so **Frame wall time** reflects the
actual steady-state per-frame time (preprocess overlaps inference).
**Sequential latency** is the sum of all stages as if run back-to-back and
represents the minimum end-to-end latency for a single frame.

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

Measured on dog_bike_768x576.nv12, Axelera Metis, SDK 1.6.

### Why pipelined FPS > sequential FPS

| Mode | Formula | FPS |
|---|---|---|
| Sequential (old) | pre + inf + dec = 8.1 ms | ~124 FPS |
| Pipelined (double-buffer) | max(pre, inf) + dec = 6.1 ms | ~163 FPS |

The bottleneck is always the AIPU inference (~6 ms).  By hiding the 2 ms
preprocess behind inference, the pipeline runs at close to bare-metal AIPU
throughput.

---

## Source layout

```
/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── annotate.hpp      — save_annotated(): draw boxes, save JPEG
│   ├── dmabuf.hpp        — DmaBuf struct: alloc/release via /dev/dma_heap/system
│   ├── preprocess.hpp    — nv12_to_tensor(): NV12 -> int8 NHWC tensor
│   ├── timer.hpp         — SectionTimer + ScopeTimer RAII
│   └── yolo_decode.hpp   — Det struct, decode_head(), nms()
├── src/
│   ├── main.cpp          — arg parsing, runtime setup, double-buffer pipeline, latency table
│   ├── annotate.cpp
│   ├── preprocess.cpp
│   └── yolo_decode.cpp
├── input_images/
│   ├── dog_bike_768x576.nv12
│   └── tulips_nv12_prog_qcif.yuv
└── output_images/
    ├── dog_bike_result.jpg
    └── tulips_result.jpg
```

---

## Implementation notes

### Double-buffer DMA-BUF pipeline
Two DMA-BUF allocations (`buf[0]`, `buf[1]`) are made at startup.  The
benchmark loop alternates between them using `cur = i & 1` and `nxt = cur ^ 1`.
A `std::async(std::launch::async, ...)` thread writes the preprocessed tensor
into `buf[nxt]` while `axr_run_model_instance` blocks on `buf[cur]`.  The
future's return value carries the thread's measured preprocess time.

```cpp
auto pre_fut = std::async(std::launch::async,
    [src, w, h, nxt_ptr, &info]() -> double {
        auto t0 = Clock::now();
        nv12_to_tensor(src, w, h, nxt_ptr, info);
        return Ms(Clock::now() - t0).count();
    });

{ ScopeTimer st(t_inf);
  axr_run_model_instance(instance, in_args[cur].data(), ...); }

t_pre.record(pre_fut.get());   // already done; no stall
```

### DMA-BUF zero-copy input
When `/dev/dma_heap/system` is available, the preprocessed input tensor is
written directly into a DMA-BUF-backed allocation.  The AIPU reads it without
any extra copy (`input_dmabuf=1`).  If the heap is unavailable the code falls
back to host memory with a warning.

### Why `output_dmabuf=0`
Output tensors are MMIO-mapped registers on the Metis device.  The axruntime
does not support DMA-BUF for MMIO outputs; attempting to set `output_dmabuf=1`
returns an error at runtime.  Host memory pointers are used for all outputs.

### Post-sigmoid outputs
YOLOv5s is compiled with the sigmoid activations fused into the AIPU graph.
`decode_head` therefore dequantises the raw int8 values directly — **do not
apply sigmoid again**.  Objectness and class scores are already in [0, 1]
after dequantisation.

### Coordinate space
All bounding boxes produced by `decode_head` are in the model's 640 x 640
input coordinate space.  `save_annotated` scales them back to the original
image resolution before drawing.

### NV12 format requirements
The NV12 decoder (`cv::COLOR_YUV2BGR_NV12`) requires even width and height.
If the source image has odd dimensions, `main.cpp` rounds up to the nearest
even size before conversion.  The extra pixel column/row is filled by OpenCV
and does not affect detection accuracy for typical images.
