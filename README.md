# YOLOv5s — NV12 / RGBA Object Detection on Axelera Metis

Real-time YOLOv5s object detection pipeline for the Axelera Metis AIPU.
Three entry points cover different camera integration scenarios — raw NV12
from a standard camera, raw RGBA from an FPGA/ISP, and a full FPGA DMA
simulation with physical-address-aware double-buffering.

---

## Pipeline overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 THREE SOURCE PATHS — converge at the AIPU                   │
└─────────────────────────────────────────────────────────────────────────────┘

  [1] yolov5s_nv12          [2] yolov5s_rgba          [3] yolov5s_rgba_fpga
  ──────────────────        ─────────────────          ─────────────────────
  .nv12 / .yuv file         .rgba file                 /dev/dma_heap/system
         │                        │                           │
         │ fread                  │ fread               DMA_HEAP_IOCTL_ALLOC
         ▼                        ▼                           ▼
  ┌─────────────┐          ┌─────────────┐          ┌───────────────────────┐
  │ NV12 bytes  │          │ RGBA bytes  │          │  rgba_dma[0]  (1.7 MB)│
  │ heap vector │          │ heap vector │          │  rgba_dma[1]  (1.7 MB)│
  └──────┬──────┘          └──────┬──────┘          └──────────┬────────────┘
         │                        │                            │
         │                        │                   virt_to_phys()
         │                        │                   → phys_addr[0/1]
         │                        │                   → fpga_set_dma_target()
         │                        │                            │
         │                        │                     ┌──────┴──────┐
         │                        │                     │ FPGA DMA    │
         │                        │                     │ engine      │
         │                        │                     │ writes RGBA │
         │                        │                     └──────┬──────┘
         │                        │                            │ (hardware,
         │                        │                            │  concurrent
         │                        │                            │  with AIPU)
         │                        │                            ▼
         │                        │                   rgba_dma[nxt].ptr
         │                        │                   (CPU reads via mmap)
         │                        │                            │
         │ nv12_to_tensor()       │ rgba_to_tensor()           │ rgba_to_tensor()
         │  cvtColor NV12→BGR     │  R,G,B direct              │  R,G,B direct
         │  (OpenCV)              │  (no cvtColor)             │  (no cvtColor)
         │  resize → uH×uW        │  resize → uH×uW            │  resize → uH×uW
         │  scale/zp → int8       │  scale/zp → int8           │  scale/zp → int8
         │  pad to [644,656,4]    │  pad to [644,656,4]        │  pad to [644,656,4]
         └────────────────────────┴────────────────────────────┘
                                          │
                                          ▼
                               ┌──────────────────────┐
                               │     in_dma[b]         │  ← DMA-BUF (1.6 MB)
                               │  int8  NHWC tensor    │
                               │  shape [1,644,656,4]  │
                               │  .fd → axruntime      │
                               └──────────┬────────────┘
                                          │  input_dmabuf=1  (zero-copy)
                                          ▼
                               ┌──────────────────────┐
                               │   Axelera Metis AIPU  │
                               │   YOLOv5s-v7-coco     │
                               │   int8, sigmoid fused │
                               └──────────┬────────────┘
                                          │  3 × MMIO outputs (host memory)
                              ┌───────────┼───────────┐
                              ▼           ▼           ▼
                        [1,80,80,256] [1,40,40,256] [1,20,20,256]
                         stride 8      stride 16     stride 32
                              │           │           │
                              └───────────┴───────────┘
                                          │
                                          │  decode_head() × 3
                                          │   dequantise int8 → float
                                          │   apply anchors → [x,y,w,h]
                                          │   conf filter (0.25)
                                          ▼
                                  greedy per-class NMS
                                  IoU threshold 0.45
                                          │
                                          ▼
                            Detections  [x1,y1,x2,y2,conf,cls]
                                          │
                                          ▼
                                  save_annotated()
                                  annotated output JPEG
```

---

## Double-buffer timing

Both the preprocess step and AIPU inference run every frame. The pipeline
overlaps them by using two input buffer slots (`buf[0]` / `buf[1]`):

```
  i = 0          i = 1          i = 2          i = 3
  cur=0, nxt=1   cur=1, nxt=0   cur=0, nxt=1   cur=1, nxt=0
  ─────────────────────────────────────────────────────────────────
  Async thread   │ pre(buf[1])│ pre(buf[0])│ pre(buf[1])│ ...
  (CPU)          └────────────┘└────────────┘└────────────┘
  ─────────────────────────────────────────────────────────────────
  Main thread    │  inf(buf[0])│  inf(buf[1])│  inf(buf[0])│ ...
  (AIPU)         └─────────────┘└─────────────┘└─────────────┘
  ─────────────────────────────────────────────────────────────────
  Wall time per  │◄── ~6.9 ms ─►│
  frame                  ↑
                  max(pre, inf)   ← preprocess (~6 ms) hidden
                                    behind inference (~6.5 ms)

  Sequential (no overlap):  pre + inf + dec  ≈ 12.8 ms  →  ~78 FPS
  Pipelined (double-buffer): max(pre, inf)   ≈  6.9 ms  → ~145 FPS
```

The async lambda returns `{dma_ms, pre_ms}` via `std::future`; `future.get()`
is called after `axr_run_model_instance` returns, so it never stalls the main
thread in steady state.

---

## FPGA DMA buffer chain (yolov5s_rgba_fpga)

```
  STARTUP — allocate and expose physical addresses
  ─────────────────────────────────────────────────────────────────────────
  /dev/dma_heap/system
       │
       │ DMA_HEAP_IOCTL_ALLOC × 4   (2 RGBA source + 2 tensor)
       │
  ┌────┴──────────────────────────────────────────────────────────────────┐
  │  rgba_dma[0]  W×H×4 B   .ptr = 0x7f…  .fd   ← FPGA writes here     │
  │  rgba_dma[1]  W×H×4 B   .ptr = 0x7f…  .fd   ← FPGA writes here     │
  │                                                                       │
  │  in_dma[0]    644×656×4 B  .ptr  .fd         ← axruntime reads this │
  │  in_dma[1]    644×656×4 B  .ptr  .fd         ← axruntime reads this │
  └───────────────────────────────────────────────────────────────────────┘
       │
       │ virt_to_phys(rgba_dma[b].ptr)  → phys_addr[b]
       │   reads /proc/self/pagemap  (needs root / CAP_SYS_ADMIN)
       │
       ▼
  fpga_set_dma_target(slot=0, phys=0x…, stride=W×4)   ← one-time ioctl
  fpga_set_dma_target(slot=1, phys=0x…, stride=W×4)

  PER FRAME — double-buffered, cur = i&1, nxt = cur^1
  ─────────────────────────────────────────────────────────────────────────

   ┌─ Async thread (CPU) ─────────────────────────────┐
   │                                                   │
   │  ① fpga_trigger_frame(nxt)    ← non-blocking     │
   │     fpga_wait_frame_done(nxt) ← blocks on IRQ    │
   │     (simulated: memcpy into rgba_dma[nxt].ptr)   │
   │                                                   │
   │  ② rgba_to_tensor(                               │
   │        rgba_dma[nxt].ptr,  ← reads DMA-BUF       │
   │        W, H,                                      │
   │        in_dma[nxt].ptr,    ← writes tensor        │
   │        in_info[0])                                │
   └──────────────────────────────┬────────────────────┘
                                  │  runs concurrently ↕
   ┌─ Main thread (AIPU) ─────────┴────────────────────┐
   │                                                   │
   │  axr_run_model_instance(                          │
   │      in_args[cur],   ← in_dma[cur].fd             │
   │      out_args)       ← host memory                │
   │                                                   │
   └───────────────────────────────────────────────────┘

   future.get()  →  decode + NMS  →  next iteration
```

---

## Measured latency

All three pipelines, 30-run benchmark, Axelera Metis, SDK 1.6, 768×576 source:

```
  [1] yolov5s_nv12  (NV12 source, DMA-BUF tensor, double-buffered)

  +------------------+----------+----------+----------+----------+
  | Section          |   avg ms |   min ms |   max ms |   p95 ms |
  +------------------+----------+----------+----------+----------+
  | Preprocess NV12  |    5.372 |    3.127 |    7.276 |    7.196 |
  | Inference (AIPU) |    6.691 |    6.111 |    7.883 |    7.462 |
  | Decode + NMS     |    0.140 |    0.080 |    0.227 |    0.195 |
  | Frame wall time  |    6.931 |    6.221 |    8.098 |    7.695 |
  +------------------+----------+----------+----------+----------+
  | Throughput:  144.3 FPS   Sequential: 12.2 ms

  [2] yolov5s_rgba  (RGBA file source, same tensor path)

  +------------------+----------+----------+----------+----------+
  | Section          |   avg ms |   min ms |   max ms |   p95 ms |
  +------------------+----------+----------+----------+----------+
  | Preprocess RGBA  |    5.962 |    4.024 |    6.559 |    6.548 |
  | Inference (AIPU) |    6.699 |    6.306 |    7.096 |    6.980 |
  | Decode + NMS     |    0.163 |    0.099 |    0.193 |    0.191 |
  | Frame wall time  |    6.935 |    6.641 |    7.331 |    7.202 |
  +------------------+----------+----------+----------+----------+
  | Throughput:  144.2 FPS   Sequential: 12.8 ms

  [3] yolov5s_rgba_fpga  (FPGA DMA simulation, DMA-BUF source + tensor)

  +------------------+----------+----------+----------+----------+
  | Section          |   avg ms |   min ms |   max ms |   p95 ms |
  +------------------+----------+----------+----------+----------+
  | FPGA DMA sim     |    0.207 |    0.133 |    0.253 |    0.249 |  ← → ~0 ms real
  | Preprocess RGBA  |    5.915 |    3.474 |    6.886 |    6.682 |
  | Inference (AIPU) |    6.514 |    6.109 |    6.955 |    6.945 |
  | Decode + NMS     |    0.153 |    0.072 |    0.213 |    0.200 |
  | Frame wall time  |    6.892 |    6.262 |    7.557 |    7.296 |
  +------------------+----------+----------+----------+----------+
  | Throughput:  145.1 FPS   Async thread: 6.1 ms (hidden behind AIPU)
```

The **FPGA DMA sim** row (0.2 ms memcpy) collapses to ~0 ms when a real FPGA
DMA engine writes the frame concurrently with the previous inference.

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
# lands in: build/yolov5s-v7-coco/yolov5s-v7-coco/1/model.json
```

---

## Build

```bash
source $VOYAGER_SDK/venv/bin/activate
export AXELERA_RUNTIME_DIR=$(python -c \
  'from axelera.runtime.configs import runtime_dir; print(runtime_dir)')

PKG_CONFIG_PATH=$AXELERA_RUNTIME_DIR/lib/pkgconfig \
    cmake -Bbuild -GNinja . -DCMAKE_BUILD_TYPE=Release

ninja -C build
```

Produces three binaries in `build/`:

| Binary | Source | Description |
|---|---|---|
| `yolov5s_nv12` | `src/main.cpp` | NV12 / YUV semi-planar input |
| `yolov5s_rgba` | `src/main_rgba.cpp` | Raw RGBA (4 B/px) input |
| `yolov5s_rgba_fpga` | `src/main_rgba_fpga.cpp` | RGBA with FPGA DMA simulation |

---

## Usage

All three binaries share the same CLI:

```
./build/<binary>  model.json  [image]  [labels.names]
                  [--size=WxH]  [--output=path.jpg]
                  [--warmup=N]  [--runs=N]
```

| Argument | Description |
|---|---|
| `model.json` | Axelera model descriptor (required) |
| `image` | `yolov5s_nv12`: `.nv12` / `.yuv` or JPEG/PNG |
| | `yolov5s_rgba` / `yolov5s_rgba_fpga`: `.rgba` or JPEG/PNG |
| `labels.names` | One class name per line |
| `--size=WxH` | Raw frame dimensions (default 1920×1080) |
| `--output=path` | Output JPEG (default `<image>_detections.jpg`) |
| `--warmup=N` | Warmup iterations, default 5 |
| `--runs=N` | Benchmark iterations, default 20 |

---

## Example commands

```bash
export LD_LIBRARY_PATH=/opt/axelera/runtime-1.6.0-1/lib:$LD_LIBRARY_PATH
MODEL=/home/ubuntu/1.6/voyager-sdk/build/yolov5s-v7-coco/yolov5s-v7-coco/1/model.json
LABELS=/home/ubuntu/1.6/voyager-sdk/ax_datasets/labels/coco.names
```

### NV12 source

```bash
./build/yolov5s_nv12 $MODEL input_images/dog_bike_768x576.nv12 $LABELS \
    --size=768x576 --warmup=5 --runs=30 --output=output_images/nv12_result.jpg
```

### RGBA source (file)

```bash
./build/yolov5s_rgba $MODEL input_images/dog_bike_768x576.rgba $LABELS \
    --size=768x576 --warmup=5 --runs=30 --output=output_images/rgba_result.jpg
```

### RGBA source with FPGA DMA simulation

```bash
# Run as root to print physical DMA addresses for the FPGA driver:
sudo ./build/yolov5s_rgba_fpga $MODEL input_images/dog_bike_768x576.rgba $LABELS \
    --size=768x576 --warmup=5 --runs=30 --output=output_images/rgba_fpga_result.jpg
```

Expected detections (all three): **dog 88 %**, **car 65 %**, **bicycle 44 %**

---

## FPGA integration guide

`yolov5s_rgba_fpga` is structured so that replacing the simulation with a real
FPGA driver is a two-line change inside the async lambda in `main_rgba_fpga.cpp`:

```cpp
// Current (simulation — memcpy into DMA-BUF):
const double dma_ms = sim.write_frame(nxt_rgba);

// Production (real FPGA DMA engine):
fpga_trigger_frame(nxt);           // tell FPGA: "fill slot nxt"
fpga_wait_frame_done(nxt);         // block until DMA-complete IRQ fires
```

Everything else — DMA-BUF allocation, physical address query, `rgba_to_tensor`,
tensor buffers, axruntime — is production-ready as-is.

### Physical address flow

```
  sudo ./build/yolov5s_rgba_fpga ...

  [FPGA-SIM] Slot 0:  virt=0x7fd33c3da000  phys=0x1a3f4000
  [FPGA-SIM]          → production: fpga_set_dma_target(slot=0, phys=0x1a3f4000, stride=3072)
  [FPGA-SIM] Slot 1:  virt=0x7fd33c22a000  phys=0x1b5e0000
  [FPGA-SIM]          → production: fpga_set_dma_target(slot=1, phys=0x1b5e0000, stride=3072)
```

The stride value (`W × 4` bytes) is the line pitch for the FPGA DMA descriptor.
Physical addresses are stable for the process lifetime once the DMA-BUF pages
are faulted in (done via `memset` before `virt_to_phys`).

> **Note:** `/proc/self/pagemap` PFN access requires `CAP_SYS_ADMIN` on Linux
> kernel ≥ 4.0. Run as root or grant the capability. The rest of the pipeline
> runs without elevated privileges.

---

## Source layout

```
.
├── CMakeLists.txt
├── README.md
├── include/
│   ├── annotate.hpp       save_annotated(): draw boxes, save JPEG
│   ├── dmabuf.hpp         DmaBuf: alloc/release via /dev/dma_heap/system
│   ├── preprocess.hpp     nv12_to_tensor(), rgba_to_tensor()
│   ├── timer.hpp          SectionTimer + ScopeTimer RAII
│   └── yolo_decode.hpp    Det struct, decode_head(), nms()
├── src/
│   ├── main.cpp               NV12 pipeline  → yolov5s_nv12
│   ├── main_rgba.cpp          RGBA pipeline  → yolov5s_rgba
│   ├── main_rgba_fpga.cpp     FPGA DMA sim   → yolov5s_rgba_fpga
│   ├── preprocess.cpp         nv12_to_tensor + rgba_to_tensor
│   ├── yolo_decode.cpp        anchor decode + NMS
│   └── annotate.cpp           bounding box drawing
├── input_images/
└── output_images/
```

---

## Implementation notes

### Preprocessing: NV12 vs RGBA

`nv12_to_tensor` calls `cv::cvtColor(YUV2BGR_NV12)` then quantises. The colour
conversion is the dominant cost (~2–3 ms at 768×576).

`rgba_to_tensor` skips `cvtColor` entirely. RGBA pixels are `[R, G, B, A]`; the
model expects `[R, G, B]` per channel, so `in_rgba[c]` for `c = 0,1,2` with a
stride of 4 (skipping A) is sufficient. This eliminates the intermediate BGR
buffer and the `cvtColor` call.

### DMA-BUF zero-copy input

`in_dma[b]` is allocated from `/dev/dma_heap/system`. The preprocessed int8
tensor is written directly into its mmap'd pointer. axruntime receives the fd
via `axrArgument.fd` (`input_dmabuf=1`), letting the AIPU DMA engine fetch it
without a CPU copy.

### Why `output_dmabuf=0`

Metis outputs are MMIO-mapped registers, not DMA-capable memory. Setting
`output_dmabuf=1` returns an error; all three output tensors use host memory.

### Post-sigmoid outputs

Sigmoid is fused into the AIPU graph. `decode_head` dequantises the raw int8
values directly — do not re-apply sigmoid. Objectness and class scores are
already in [0, 1] after `value * scale + zero_point`.

### Coordinate space

`decode_head` outputs boxes in the model's 640×640 input space. `save_annotated`
scales them back to the original image resolution before drawing.
