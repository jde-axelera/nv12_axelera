# Basics of AIPU Execution — Axelera Metis

A mental-picture guide to everything from silicon to GStreamer.

---

## 1. The Hardware Stack — What's Actually in the Box

```
┌─────────────────────────────────────────────────────────────┐
│                    HOST CPU  (x86)                          │
│  Your C++ program / GStreamer pipeline runs here            │
└────────────────────────┬────────────────────────────────────┘
                         │  PCIe bus
┌────────────────────────▼────────────────────────────────────┐
│                  Axelera Metis PCIe Card                    │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  AIPU 0  │  │  AIPU 1  │  │  AIPU 2  │  │  AIPU 3  │   │
│  │  ~8MB L2 │  │  ~8MB L2 │  │  ~8MB L2 │  │  ~8MB L2 │   │
│  │  MAC arr │  │  MAC arr │  │  MAC arr │  │  MAC arr │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  Shared DDR  (GDDR6)                 │   │
│  │  model weights (elf_in_ddr=1), activations, I/O      │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Key facts to burn in:**
- 4 independent AIPU cores, each with its own ~8 MB L2 SRAM (constant memory).
  Two models on the same core → they fight for that 8 MB → load fails.
- L2 is where model weights live during inference (fast path).
  DDR is where they come from (slow path, ~10× higher latency).
- MMIO output registers: inference results are memory-mapped I/O registers,
  not a DMA-capable buffer. That's why `output_dmabuf=0` is mandatory forever.

---

## 2. Level Zero (L0) — The Driver Language

**Mental model: L0 is the "assembly language" of GPU/accelerator compute.**

```
Your code (C++)
    │
    ▼
axruntime API       ← what you call: axr_run_model_instance()
    │
    ▼
Level Zero (L0)     ← Intel oneAPI low-level driver interface
    │                  (same layer as CUDA for NVIDIA, Metal for Apple)
    ▼
PCIe driver
    │
    ▼
Metis card hardware
```

L0 concepts that matter:

```
ze_context_handle_t   ─── the "session" — owns all allocations, modules, queues
                          one context = one coherent view of the device
                          NOT thread-safe for concurrent submission

ze_command_queue_t    ─── ordered stream of work submitted to the device
                          axruntime creates ONE queue per connection

ze_command_list_t     ─── a recorded batch of operations (like a GPU command buffer)
                          axr_run_model_instance records one, submits it to the queue

ze_fence_t            ─── CPU waits on this until the GPU/AIPU signals "done"
```

**Why axr_run_model_instance is not thread-safe:**

```
Thread A:                        Thread B:
  submit command_list_A  ─┐        submit command_list_B  ─┐
  wait on fence_A         │        wait on fence_B         │
                          │                                │
              ┌───────────▼───────────────────────────────▼──┐
              │      SINGLE ze_command_queue                  │
              │      SINGLE ze_fence (shared state)           │
              └───────────────────────────────────────────────┘
                          ↓
             Race: Thread B resets the fence before
             Thread A's wait resolves → L0::Exception
             "Wait kernel failed with return code -1"
```

axruntime allocates **one fence per connection**, not one per instance.
Even though AIPU cores are independent silicon, the host-side bookkeeping
is shared → one thread wins, the other sees a broken fence.

---

## 3. A Single axr_run_model_instance Call — What Happens

```
Host CPU                              Metis Card
─────────────────────────────────────────────────────────────
axr_run_model_instance(instance, in_args, n_in, out_args, n_out)
  │
  ├─ L0: build command list
  │    ├─ DMA transfer: input tensor → DDR   (if DMA-BUF: near-zero, already there)
  │    ├─ Signal AIPU core N to start
  │    └─ Fence: notify CPU when done
  │
  ├─ L0: submit command list to queue ────────────────────────────────────────►
  │                                                                            │
  │  [CPU blocks here on fence]              AIPU core N executes operators:  │
  │                                          [conv2d] → [bn] → [relu] → ...   │
  │                                          reads weights from L2 or DDR     │
  │                                          writes activations to DDR        │
  │                                          writes outputs to MMIO registers  │
  │                                                                            │
  ◄────────────────────────────────────── fence signalled ─────────────────────
  │
  └─ copy MMIO output registers into out_host[] buffers
        (this is why output_dmabuf=0: can't DMA from MMIO)

Returns to your code.  Outputs are in model.out_host[].
```

**The key insight:** the CPU is *synchronously blocked* for the entire AIPU execution.
`axr_run_model_instance` does not return until inference is complete.
This makes it easy to reason about but means idle CPU time = wasted time.

---

## 4. The Operator Pipeline Inside One AIPU Call

Each neural network layer is an "operator". The AIPU runs them in sequence:

```
Model weights in L2 (pre-loaded at axr_load_model_instance time):
┌─────────────────────────────────────────────────────────┐
│  W_conv1 │ W_conv2 │ W_bn1 │ W_conv3 │ ... │ W_fcN     │
└─────────────────────────────────────────────────────────┘

AIPU execution timeline (double_buffer=0 — NO internal prefetch):
─────────────────────────────────────────────────────────────────
time →
[load W_conv1 → L2]  [execute conv1]  [load W_conv2 → L2]  [execute conv2]  ...
 ←── DDR latency ──►  ←── MAC arr ──►  ←── DDR latency ──►  ←── MAC arr ──►

AIPU execution timeline (double_buffer=1 — WITH internal prefetch):
─────────────────────────────────────────────────────────────────
time →
[load W_conv1 → L2]  [execute conv1 + load W_conv2]  [execute conv2 + load W_conv3] ...
 ←── DDR latency ──►  ←── MAC arr ──── DDR hidden ──►  ←── MAC arr ──── DDR hidden ──►
                       ^ overlap! DDR load hides behind MAC execution
```

**double_buffer=1 is a micro-architecture trick.** It hides DDR weight-load latency
behind MAC array execution. It's the AIPU equivalent of CPU instruction pipelining.

**Why it had no effect in our benchmark:**
Our models use `elf_in_ddr=1` (weights stay in DDR, not L2 code memory).
With small models (YOLOv5s ~7MB, ResNet50 ~8MB), the AIPU's own prefetch
hardware and L2 caching absorbs most DDR latency even without the flag.
`double_buffer=1` helps most with very large models that thrash L2.

---

## 5. Frame-Level Double Buffering — The Producer/Consumer Picture

**Problem:** Preprocessing (CPU) takes ~6ms. AIPU takes ~9ms.
Without double buffering:

```
Frame timeline (SEQUENTIAL — no overlap):
─────────────────────────────────────────────────────────────────
Frame 0:  [preprocess 6ms][AIPU 9ms]
Frame 1:                              [preprocess 6ms][AIPU 9ms]
Frame 2:                                                          [preprocess 6ms][AIPU 9ms]

Wall time per frame = 6 + 9 = 15ms → 67 FPS   ← leaving 6ms idle on CPU
```

**Solution: Double buffer — two input slots, CPU and AIPU run on different slots.**

```
Buffer slots:   slot[0] ████████████████ (DMA-BUF pinned memory)
                slot[1] ████████████████ (DMA-BUF pinned memory)

Frame timeline (DOUBLE BUFFERED — overlap):
─────────────────────────────────────────────────────────────────
         slot[0]:  [preprocess F0]
         slot[1]:                  [preprocess F1]   [preprocess F2]
                                          ↑ hidden!         ↑ hidden!
AIPU:              .........  [AIPU F0]  [  AIPU F1  ]  [  AIPU F2  ]
                   ^ first                ^ uses slot[0]   ^ uses slot[1]
                   frame has
                   no overlap

Wall time per frame = max(6ms preprocess, 9ms AIPU) = 9ms → 111 FPS ← ideal
                      (actual: ~9ms wall once pipeline is full)
```

**In code (our implementation):**

```
slot[0]: yolo_dma[0] ──► yolo_ptrs[0] ──► DMA-BUF fd 0
slot[1]: yolo_dma[1] ──► yolo_ptrs[1] ──► DMA-BUF fd 1

Frame N loop:
  cur = N & 1          // which slot is ready for AIPU
  nxt = cur ^ 1        // which slot CPU will write next

  async thread ──► preprocess_yolo(yolo_ptrs[nxt])    // CPU writes nxt
  main thread  ──► axr_run_model_instance(yolo_args[cur])  // AIPU reads cur
  wait for async
```

**Why DMA-BUF?**
Without DMA-BUF, you'd need to copy: `host_malloc → DMA-BUF → AIPU`.
With DMA-BUF, preprocess writes *directly* into the pinned buffer the AIPU reads from.
Zero extra copy. The `input_dmabuf=1` prop tells axruntime "the fd I'm passing IS
the input, no staging needed."

```
Without DMA-BUF:
  preprocess → heap buffer → [memcpy into DMA-BUF] → AIPU
               ↑ your buffer   ↑ axruntime does this  ↑ reads directly

With DMA-BUF:
  preprocess → DMA-BUF ──────────────────────────── → AIPU
               ↑ IS the DMA-BUF, no copy
```

---

## 6. ResNet50 Batch-Level Double Buffering

The same principle applies *within* a single frame, across crop batches:

```
Frame has 3 detections. ResNet50 batch=2 → need 2 calls (batches of 2, 1+pad).

Timeline WITHOUT batch double-buffering:
  [fill crops 0-1: 2ms] [AIPU batch0: 4ms] [fill crops 2: 2ms] [AIPU batch1: 4ms]
  Total = 2+4+2+4 = 12ms

Timeline WITH batch double-buffering:
  [fill crops 0-1: 2ms]
                         [AIPU batch0: 4ms]
                          [fill crops 2: 2ms]  ← hidden behind AIPU!
                                               [AIPU batch1: 4ms]
  Total = 2ms prime + 4ms + 4ms = 10ms   (saved 2ms)
```

Two ResNet50 DMA-BUF slots: `resnet_dma[0]` and `resnet_dma[1]`.
While AIPU runs batch b (slot cur), async thread fills batch b+1 (slot nxt).
This is structurally identical to the frame-level YOLO double buffer.

---

## 7. GStreamer — The Same Thing, Automatically

GStreamer achieves frame-level double-buffering through its **threading model**,
not through any special code you write.

```
GStreamer pipeline:

┌──────────┐     ┌────────────┐     ┌───────┐     ┌────────────────┐
│  v4l2src │────►│ preprocess │────►│ queue │────►│ axinferencenet │
│ (camera) │     │  element   │     │ depth=│     │  (AIPU call)   │
└──────────┘     └────────────┘     │  2-4  │     └────────────────┘
                                    └───────┘
thread A ──────────────────────────────────┐  thread B ─────────────
(produces + preprocesses frames)          │  (pulls + runs AIPU)
                                           │
                          GstBufferPool ◄──┘
                    (pre-allocated DMA-BUF slots)
```

**The queue element IS the double buffer:**
- It holds a ring of `GstBuffer` objects backed by DMA-BUF.
- Thread A (preprocess) writes into buffer slot N+1.
- Thread B (axinferencenet) is blocked in `axr_run_model_instance` on slot N.
- When AIPU finishes, thread B returns slot N to the pool, picks up slot N+1.
- Thread A gets slot N from the pool to write slot N+2.

```
GstBufferPool (min-buffers=2):

slot[0]: GstBuffer ──► DMA-BUF fd 0 ──► mmap ptr ──► preprocess writes here
slot[1]: GstBuffer ──► DMA-BUF fd 1 ──► mmap ptr ──► AIPU reads from here

                  ↕ slots rotate through the pool automatically
```

**axinferencenet internally:**
When it receives a `GstBuffer` in its `chain()` function, it:
1. Extracts the DMA-BUF fd from the buffer's memory.
2. Calls `axr_run_model_instance` with `input_dmabuf=1` and that fd.
3. Copies MMIO outputs into downstream buffers.
4. Releases the input buffer back to the pool.

The GStreamer element doesn't need to manage two slots manually — the pool
and queue handle that. Our C++ code had to implement it explicitly because
we bypassed GStreamer entirely.

---

## 8. Full Mental Picture — Everything Together

```
                        FRAME N-1          FRAME N           FRAME N+1
                     ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
CAMERA/SOURCE        │ read frame  │   │ read frame  │   │ read frame  │
                     └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
                            │                 │                 │
                     ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
PREPROCESS           │ RGBA→tensor │   │ RGBA→tensor │   │ RGBA→tensor │
(CPU thread A)       │ slot[1]     │   │ slot[0]     │   │ slot[1]     │
                     └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
                            │ ←─ hidden ──►   │ ←─ hidden ──►   │
                     ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
YOLO AIPU            │ AIPU slot[1]│   │ AIPU slot[0]│   │ AIPU slot[1]│
(main thread B)      │   9ms       │   │   9ms       │   │   9ms       │
  ─── WALL ────      └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
                            │ decode           │ decode           │ decode
                     ┌──────▼──────────────────▼─────────────────▼──────┐
RESNET50             │ embed batch0 │ embed batch1 │ embed batch0 │ ...  │
(main thread B)      │  rslot[0]   │  rslot[1]   │  rslot[0]   │      │
  ─── WALL ────      │  AIPU+fill  │  AIPU+fill  │  AIPU+fill  │      │
                     └─────────────────────────────────────────────────┘

Wall time per frame = max(preprocess, YOLO AIPU) + decode + ResNet section
                    = max(6.4ms, 9ms)            + 0.2ms  + 9.6ms
                    = 9ms + 0.2ms + 9.6ms ≈ 18.8ms → ~53 FPS
```

---

## 9. The Model Version = Core Count = Batch Size Connection

```
/build/resnet50-imagenet/resnet50-imagenet/
                                          ├── 1/model.json  → batch=1, 1 core
                                          ├── 2/model.json  → batch=2, 2 cores
                                          ├── 3/model.json  → batch=3, 3 cores
                                          └── 4/model.json  → batch=4, 4 cores
```

Why does batch size match core count?

```
Batch=2 inference layout:
┌──────────────────────────────────────────────┐
│             AIPU cores 1 + 2                 │
│  core 1: processes sample 0 of the batch     │
│  core 2: processes sample 1 of the batch     │
│  sync barrier at end → combined output       │
└──────────────────────────────────────────────┘
```

Each sample in a batch runs on its own core. More cores = higher batch throughput,
but per-call latency grows (sync overhead, weight duplication). That's why
going from v1 to v3 raises per-call time (4.1ms → 5.4ms) even though you use 3x
the silicon — synchronisation overhead and weight broadcast cost is non-trivial.

**Why v1+v3 beats v1+v2 for 3 detections:**
- v2: ceil(3/2)=2 calls × 4.1ms = 8.2ms AIPU total
- v3: ceil(3/3)=1 call  × 5.4ms = 5.4ms AIPU total  ← 2.8ms saved

**Why v2+v2 for YOLO is slower than v1:**
2-core YOLO adds sync overhead and weight broadcast but runs the *same frame*
(you can't batch separate frames because each frame has its own data).
Single core → no sync overhead → faster for batch=1 frames.

---

## 10. L2 Memory Budget — Why You Must Connect to Enough Sub-Devices

```
axr_device_connect(ctx, nullptr, N_sub_devices, props)
                                 ↑
                   This controls how many cores you claim.
                   Each sub-device has its own ~8MB L2.

If N=1:
  ┌──────────────────────────────────┐
  │  Sub-device 0 (core 0, 8MB L2)  │
  │  ← YOLOv5s loads here (~7.8MB)  │
  │  ← ResNet50 tries to load here  │  ← FAILS: only 0.2MB free!
  └──────────────────────────────────┘

If N=3 (yolo_cores=1 + resnet_cores=2):
  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
  │ core 0 8MB  │  │ core 1 8MB  │  │ core 2 8MB  │
  │ YOLOv5s v1  │  │ ResNet50    │  │ ResNet50    │
  │ (~7.8MB)    │  │ v2 half     │  │ v2 half     │
  └─────────────┘  └─────────────┘  └─────────────┘
  Independent pools — no competition. Both models load successfully.
```

---

## Quick Reference Cheatsheet

| Concept | What it is | Where it lives |
|---------|-----------|---------------|
| L0 / Level Zero | Intel's low-level accelerator driver API | Host OS / PCIe driver |
| `axrContext` | L0 session, owns all state | Host CPU |
| `axrConnection` | L0 command queue + fence, single submission path | Host CPU |
| `axrModelInstance` | One loaded model on N cores, with its L2 weights | Metis card |
| `axr_run_model_instance` | Synchronous blocking call: submit → wait on fence → return | CPU→AIPU |
| L2 SRAM | Fast weight memory, ~8MB per core, exclusive per core | Metis card |
| DDR | Slow weight storage, shared; used when elf_in_ddr=1 | Metis card |
| MMIO outputs | Results land in memory-mapped I/O registers, not DMA-capable | Metis card |
| DMA-BUF | Pinned host memory the AIPU DMA engine can read directly (no copy) | Host RAM / PCIe |
| Frame-level double-buffer | Two input DMA-BUF slots; CPU fills slot N+1 while AIPU reads slot N | Host CPU + DMA |
| Batch-level double-buffer | Same idea across crop batches within one frame (ResNet50 stage) | Host CPU + DMA |
| `double_buffer=1` prop | AIPU-internal: prefetch next operator's weights behind current MAC | AIPU microarch |
| GStreamer queue | Achieves frame-level double-buffer automatically via thread separation | Host CPU |
| axinferencenet | GStreamer element wrapping axr_run_model_instance with GstBufferPool | Host CPU |
| Thread safety | axruntime is single-threaded — one fence per connection crashes if two threads submit concurrently | axruntime / L0 |
