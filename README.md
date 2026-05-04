# YOLOv5s Object Detection on Axelera Metis

Real-time YOLOv5s-v7-coco detection pipeline for the Axelera Metis AIPU.
Accepts raw camera frames (NV12 or RGBA), runs int8 inference on the AIPU,
and returns bounding boxes with class labels and confidence scores.

Three binaries are provided to match different camera integration scenarios:

| Binary | Input source | Use when |
|---|---|---|
| `yolov5s_nv12` | `.nv12` / `.yuv` file or JPEG/PNG | Standard camera, NV12 output |
| `yolov5s_rgba` | `.rgba` file or JPEG/PNG | FPGA/ISP that outputs raw RGBA |
| `yolov5s_rgba_fpga` | `.rgba` file + DMA-BUF simulation | Testing FPGA DMA integration |

---

## Quick start

### 1. Build

```bash
source $VOYAGER_SDK/venv/bin/activate
export AXELERA_RUNTIME_DIR=$(python -c \
  'from axelera.runtime.configs import runtime_dir; print(runtime_dir)')

PKG_CONFIG_PATH=$AXELERA_RUNTIME_DIR/lib/pkgconfig \
    cmake -Bbuild -GNinja . -DCMAKE_BUILD_TYPE=Release

ninja -C build
# → build/yolov5s_nv12, build/yolov5s_rgba, build/yolov5s_rgba_fpga
```

### 2. Download the model

```bash
axdownloadmodel --model yolov5s-v7-coco
# lands in: build/yolov5s-v7-coco/yolov5s-v7-coco/1/model.json
```

### 3. Run

```bash
export LD_LIBRARY_PATH=/opt/axelera/runtime-1.6.0-1/lib:$LD_LIBRARY_PATH
MODEL=build/yolov5s-v7-coco/yolov5s-v7-coco/1/model.json
LABELS=$VOYAGER_SDK/ax_datasets/labels/coco.names

# NV12 source
./build/yolov5s_nv12 $MODEL input_images/dog_bike_768x576.nv12 $LABELS \
    --size=768x576 --warmup=5 --runs=30 --output=output_images/result.jpg

# RGBA source
./build/yolov5s_rgba $MODEL input_images/dog_bike_768x576.rgba $LABELS \
    --size=768x576 --warmup=5 --runs=30 --output=output_images/result.jpg

# FPGA DMA simulation (run as root to print physical addresses)
sudo ./build/yolov5s_rgba_fpga $MODEL input_images/dog_bike_768x576.rgba $LABELS \
    --size=768x576 --warmup=5 --runs=30 --output=output_images/result.jpg
```

Expected detections on `dog_bike_768x576`: **dog 88 %**, **car 65 %**, **bicycle 44 %**

### CLI reference

| Argument | Default | Description |
|---|---|---|
| `model.json` | required | Axelera model descriptor |
| `image` | synthetic grey | `.nv12`/`.yuv`/`.rgba` or JPEG/PNG |
| `labels.names` | (no labels) | One class name per line |
| `--size=WxH` | 1920×1080 | Raw frame dimensions |
| `--output=path` | `<image>_detections.jpg` | Output JPEG |
| `--warmup=N` | 5 | Warmup iterations (not timed) |
| `--runs=N` | 20 | Benchmark iterations |

---

## How a frame becomes a detection

The pipeline has four stages. Each stage is explained below with the relevant
code function.

```
  Camera frame (NV12 or RGBA)
          │
          │  Stage 1 — Preprocess
          │  Convert colour format, resize, quantise to int8
          ▼
  int8 input tensor  [1, 644, 656, 4]
          │
          │  Stage 2 — AIPU Inference
          │  YOLOv5s forward pass, sigmoid fused on-chip
          ▼
  3 × output tensors  (stride 8 / 16 / 32)
          │
          │  Stage 3 — Decode
          │  Dequantise, apply anchors, filter by confidence
          ▼
  Candidate detections  [x, y, w, h, conf, cls]
          │
          │  Stage 4 — NMS
          │  Remove overlapping boxes, keep best per class
          ▼
  Final detections  [x1, y1, x2, y2, conf, cls]
```

### Stage 1 — Preprocess

The model expects a 640×640 RGB image quantised to int8 with shape
`[1, 644, 656, 4]` (the extra rows/columns are zero-padding added by the
compiler; channel count 4 includes 1 pad channel).

**NV12 path** (`nv12_to_tensor`):
```
NV12 bytes  →  cv::cvtColor(YUV2BGR_NV12)  →  BGR uint8
            →  cv::resize(→ 640×640)
            →  BGR→RGB reorder + scale/zp → int8
```

**RGBA path** (`rgba_to_tensor`):
```
RGBA bytes  →  cv::resize(→ 640×640)  (4-channel resize, no colour conversion)
            →  read R,G,B channels directly, drop A + scale/zp → int8
```

The RGBA path skips `cvtColor` entirely. Since RGBA pixels are already
`[R, G, B, A]` and the model expects R, G, B order, the three channels are
read directly during quantisation with a stride of 4 to skip A. No
intermediate buffer is needed.

### Stage 2 — AIPU inference

`axr_run_model_instance` submits the int8 tensor to the Axelera Metis AIPU
and blocks until inference completes (~6.5 ms). The sigmoid activation is
fused into the compiled AIPU graph — do not apply it again in software.

### Stage 3 — Decode

`decode_head` is called once per output stride. It:
1. Dequantises int8 values: `float = value × scale + zero_point`
2. Applies the YOLOv5 anchor grid to convert raw offsets → `[x, y, w, h]`
3. Filters candidates below the confidence threshold (0.25)

Output boxes are in the model's 640×640 coordinate space. `save_annotated`
scales them back to the original frame resolution before drawing.

### Stage 4 — NMS

Greedy per-class non-maximum suppression. Boxes are sorted by confidence,
then iteratively kept or discarded based on IoU > 0.45 with already-kept boxes.

---

## Understanding the buffers

This is the most important section if you are integrating an FPGA or custom
camera source.

### Why buffers matter: three parties, three languages

The FPGA, CPU, and AIPU all work with memory differently:

| Party | How it refers to memory | Why |
|---|---|---|
| **FPGA DMA engine** | Physical address (e.g. `0x1a3f4000`) | Hardware DMA controllers work with real RAM addresses as seen by the memory controller |
| **CPU (your code)** | Virtual address (e.g. `0x7fd33c3da000`, the `ptr` from `mmap`) | Linux user-space programs never see physical addresses directly |
| **AIPU (axruntime)** | DMA-BUF file descriptor (`fd`) | The kernel tracks buffer ownership and cache coherency through fd objects |

Physical address, virtual address, and DMA-BUF fd can all refer to **the same
bytes in RAM** — they are just three different ways to name them.

```
  Physical RAM
  ┌─────────────────────────────────┐
  │  0x1a3f4000  (1.7 MB)           │  ← actual bytes on the memory bus
  └─────────────────────────────────┘
         ▲                   ▲
         │ DMA write         │ mmap read
         │ (hardware)        │ (software)
    FPGA engine           CPU ptr
    sees 0x1a3f4000       sees 0x7fd33c3da000
```

### Why you need two buffers for the FPGA path

The FPGA outputs raw RGBA pixels. The AIPU expects a quantised int8 tensor.
These are different sizes and different formats — you cannot pass one directly
as the other. A CPU preprocessing step converts between them, so two separate
buffers are required:

```
  Buffer A — rgba_dma  (FPGA → CPU)
  ┌──────────────────────────────────────────────────────┐
  │  Size:  W × H × 4 bytes  (raw RGBA pixels)           │
  │  e.g.   768 × 576 × 4 = 1.7 MB                      │
  │                                                      │
  │  .phys_addr ──────────► FPGA DMA engine              │
  │              "write your frame to THIS address"      │
  │                                                      │
  │  .ptr (mmap) ──────────► CPU reads raw pixels here   │
  │              (same physical RAM, virtual window)     │
  │                                                      │
  │  .fd         ✗  NOT passed to axruntime              │
  └──────────────────────────────────────────────────────┘
                           │
                           │  rgba_to_tensor()
                           │  ① cv::resize  RGBA → 640×640
                           │  ② read R,G,B, skip A
                           │  ③ scale/zero_point → int8
                           │  CPU bridges the two buffers
                           ▼
  Buffer B — in_dma  (CPU → AIPU)
  ┌──────────────────────────────────────────────────────┐
  │  Size:  644 × 656 × 4 bytes  (int8 tensor)           │
  │         ≈ 1.6 MB                                     │
  │                                                      │
  │  .ptr (mmap) ◄──────── CPU writes quantised data     │
  │                                                      │
  │  .fd  ─────────────► axruntime (input_dmabuf=1)      │
  │        "AIPU, read your input tensor from THIS fd"   │
  └──────────────────────────────────────────────────────┘
```

**The FPGA never touches Buffer B. axruntime never touches Buffer A.**
The CPU's `rgba_to_tensor()` call is the only bridge between them.

### Step-by-step: startup and per-frame flow

**Startup (done once):**

```
Step 1 — Allocate Buffer A (rgba_dma) from /dev/dma_heap/system
         DMA_HEAP_IOCTL_ALLOC(size = W×H×4)
           → .fd   (kernel DMA-BUF object)
           → .ptr  (virtual address, CPU can read/write via mmap)

Step 2 — Translate .ptr to a physical address
         read /proc/self/pagemap for the page containing .ptr
           → .phys  (e.g. 0x1a3f4000)
         Requires root / CAP_SYS_ADMIN.
         Only needed once — physical address is stable for process lifetime.

Step 3 — Tell the FPGA where to write
         fpga_set_dma_target(phys=0x1a3f4000, stride=W×4)
         From this point on, the FPGA DMA engine writes each frame
         directly to that physical address. No CPU involvement.

Step 4 — Allocate Buffer B (in_dma) from /dev/dma_heap/system
         DMA_HEAP_IOCTL_ALLOC(size = 644×656×4)
           → .fd   ← this fd is what you pass to axruntime
           → .ptr  ← CPU writes the tensor here
```

**Per frame (repeated every frame):**

```
Step 5 — FPGA writes the new frame
         FPGA DMA engine copies W×H×4 RGBA bytes → phys address of Buffer A
         Hardware operation. CPU does nothing. Runs concurrently with AIPU.

Step 6 — CPU reads from Buffer A, writes to Buffer B
         rgba_to_tensor(rgba_dma.ptr,  ← reads from Buffer A via virtual ptr
                        W, H,
                        in_dma.ptr,    ← writes into Buffer B via virtual ptr
                        tensor_info)

Step 7 — AIPU runs inference on Buffer B
         axr_run_model_instance(in_dma.fd, ...)
         axruntime passes .fd to the AIPU DMA engine.
         AIPU fetches the tensor directly — no CPU copy.
```

### What the program prints at startup (run as root)

```
[FPGA-SIM] Slot 0:  virt=0x7fd33c3da000  phys=0x1a3f4000
[FPGA-SIM]          → production: fpga_set_dma_target(slot=0, phys=0x1a3f4000, stride=3072)
[FPGA-SIM] Slot 1:  virt=0x7fd33c22a000  phys=0x1b5e0000
[FPGA-SIM]          → production: fpga_set_dma_target(slot=1, phys=0x1b5e0000, stride=3072)

[INFO] Buffer layout:
  rgba_dma[0] ptr=0x7fd33c3da000  ← FPGA DMA target, slot 0
  rgba_dma[1] ptr=0x7fd33c22a000  ← FPGA DMA target, slot 1
  in_dma[0]   ptr=0x7fd327e63000  ← tensor input to axruntime, slot 0
  in_dma[1]   ptr=0x7fd327cc6000  ← tensor input to axruntime, slot 1
```

The `stride` value is `W × 4` bytes — the line pitch for the FPGA DMA
descriptor (number of bytes from the start of one row to the start of the next).

---

## Double-buffer pipeline

Preprocessing and AIPU inference happen every frame. By running them on
alternating buffer slots they overlap, hiding the preprocess time behind inference.

```
  Two slots:  buf[0] and buf[1]
  Each slot has its own rgba_dma and in_dma pair.

  Iteration:    i=0            i=1            i=2            i=3
  cur slot:     0              1              0              1
  nxt slot:     1              0              1              0
  ──────────────────────────────────────────────────────────────────
  Async thread  │pre+DMA(buf1)│ pre+DMA(buf0)│ pre+DMA(buf1)│ ...
  (CPU)         └─────────────┘└─────────────┘└─────────────┘
  ──────────────────────────────────────────────────────────────────
  Main thread   │ inference(buf0) │ inference(buf1) │ inference(buf0) │ ...
  (AIPU)        └─────────────────┘└─────────────────┘└─────────────────┘
  ──────────────────────────────────────────────────────────────────
  Frame time    │◄─── ~6.9 ms ───►│

  Sequential (no overlap):  preprocess + inference + decode ≈ 12.8 ms →  ~78 FPS
  Pipelined (double-buffer): max(preprocess, inference)     ≈  6.9 ms → ~145 FPS
```

Because preprocessing (~6 ms) completes before inference (~6.5 ms) finishes,
`future.get()` in the main thread never stalls — the async thread is already
done by the time it is called.

---

## FPGA integration: from simulation to production

`yolov5s_rgba_fpga` uses a `memcpy` to simulate what the FPGA DMA engine does.
Replacing the simulation with a real FPGA driver requires changing **two lines**
in the async thread inside `src/main_rgba_fpga.cpp`:

```cpp
// ── Current code (simulation) ──────────────────────────────────────
const double dma_ms = sim.write_frame(nxt_rgba);
//   ↑ this is a memcpy into the DMA-BUF to stand in for the FPGA

// ── Replace with (production) ──────────────────────────────────────
fpga_trigger_frame(nxt);           // non-blocking ioctl to FPGA driver
fpga_wait_frame_done(nxt);         // blocks until DMA-complete IRQ fires
//   ↑ at this point rgba_dma[nxt].ptr contains the new RGBA frame
//     written by the FPGA DMA engine to the physical address from Step 3
```

Everything else in the file is production-ready: DMA-BUF allocation, physical
address query, `rgba_to_tensor`, tensor buffers, axruntime properties, the
double-buffer loop, decode, NMS.

> **Root requirement:** `/proc/self/pagemap` PFN access requires
> `CAP_SYS_ADMIN` (kernel ≥ 4.0). Run as root once at startup to obtain
> physical addresses for the FPGA driver. The rest of the pipeline runs
> without elevated privileges.

---

## Measured latency

30-run benchmark, Axelera Metis PCIe, SDK 1.6, 768×576 source frame.

```
  yolov5s_nv12 — NV12 source
  +------------------+----------+----------+----------+----------+
  | Section          |   avg ms |   min ms |   max ms |   p95 ms |
  +------------------+----------+----------+----------+----------+
  | Preprocess NV12  |    5.372 |    3.127 |    7.276 |    7.196 |
  | Inference (AIPU) |    6.691 |    6.111 |    7.883 |    7.462 |
  | Decode + NMS     |    0.140 |    0.080 |    0.227 |    0.195 |
  | Frame wall time  |    6.931 |    6.221 |    8.098 |    7.695 |
  +------------------+----------+----------+----------+----------+
  Throughput: 144.3 FPS   Sequential: 12.2 ms

  yolov5s_rgba — RGBA source
  +------------------+----------+----------+----------+----------+
  | Section          |   avg ms |   min ms |   max ms |   p95 ms |
  +------------------+----------+----------+----------+----------+
  | Preprocess RGBA  |    5.962 |    4.024 |    6.559 |    6.548 |
  | Inference (AIPU) |    6.699 |    6.306 |    7.096 |    6.980 |
  | Decode + NMS     |    0.163 |    0.099 |    0.193 |    0.191 |
  | Frame wall time  |    6.935 |    6.641 |    7.331 |    7.202 |
  +------------------+----------+----------+----------+----------+
  Throughput: 144.2 FPS   Sequential: 12.8 ms

  yolov5s_rgba_fpga — FPGA DMA simulation
  +------------------+----------+----------+----------+----------+
  | Section          |   avg ms |   min ms |   max ms |   p95 ms |
  +------------------+----------+----------+----------+----------+
  | FPGA DMA sim (†) |    0.207 |    0.133 |    0.253 |    0.249 |
  | Preprocess RGBA  |    5.915 |    3.474 |    6.886 |    6.682 |
  | Inference (AIPU) |    6.514 |    6.109 |    6.955 |    6.945 |
  | Decode + NMS     |    0.153 |    0.072 |    0.213 |    0.200 |
  | Frame wall time  |    6.892 |    6.262 |    7.557 |    7.296 |
  +------------------+----------+----------+----------+----------+
  Throughput: 145.1 FPS   Async thread total: 6.1 ms

  (†) The FPGA DMA sim row is a memcpy (1.7 MB). With a real FPGA the
      DMA write runs concurrently with the AIPU — this row becomes ~0 ms.
```

---

## Source layout

```
.
├── CMakeLists.txt
├── README.md
├── include/
│   ├── annotate.hpp       save_annotated() — draw boxes, write JPEG
│   ├── dmabuf.hpp         DmaBuf struct — alloc/mmap/release via dma_heap
│   ├── preprocess.hpp     nv12_to_tensor(), rgba_to_tensor()
│   ├── timer.hpp          SectionTimer, ScopeTimer RAII helpers
│   └── yolo_decode.hpp    Det struct, decode_head(), nms()
├── src/
│   ├── main.cpp               NV12 pipeline        → yolov5s_nv12
│   ├── main_rgba.cpp          RGBA file pipeline   → yolov5s_rgba
│   ├── main_rgba_fpga.cpp     FPGA DMA simulation  → yolov5s_rgba_fpga
│   ├── preprocess.cpp         nv12_to_tensor + rgba_to_tensor implementations
│   ├── yolo_decode.cpp        anchor decode + NMS
│   └── annotate.cpp           bounding box drawing
├── input_images/
└── output_images/
```

---

## Implementation notes

### Why `output_dmabuf=0`

The three output tensors on Metis are MMIO-mapped registers on the device,
not regular DMA-capable memory. Passing `output_dmabuf=1` returns a runtime
error. All outputs use host memory pointers.

### Post-sigmoid outputs

YOLOv5s is compiled with sigmoid fused into the AIPU graph. `decode_head`
dequantises raw int8 values with `value × scale + zero_point` — this already
gives values in [0, 1]. Do not apply sigmoid again.

### NV12 dimension requirement

`cv::COLOR_YUV2BGR_NV12` requires even width and height. `main.cpp` rounds
up odd source dimensions by one pixel before conversion.

### Physical address stability

`virt_to_phys` calls `memset` on the DMA-BUF before reading `/proc/self/pagemap`
to ensure all pages are faulted into physical RAM. Once faulted in, DMA-BUF
pages are pinned for the lifetime of the fd — the physical address does not
change and only needs to be read once at startup.
