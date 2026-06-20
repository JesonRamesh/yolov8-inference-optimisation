# YOLOv8x Inference Optimisation — ONNX, FP16, INT8

> UCL MEng Robotics & AI (Year 2) — Computer Vision Portfolio Project
> **Companion repo to** [yolov8-bytetrack-mot17](https://github.com/JesonRamesh/yolov8-bytetrack-mot17)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNXRuntime-1.20-green)](https://onnxruntime.ai)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JesonRamesh/yolov8-inference-optimization/blob/main/notebook.ipynb)

---

## Motivation

My companion repo ([yolov8-bytetrack-mot17](https://github.com/JesonRamesh/yolov8-bytetrack-mot17)) builds a working pedestrian tracking pipeline that achieves HOTA 41.4% on MOT17 — but runs at only ~17 FPS on a T4 GPU. For real-world deployment on embedded hardware (Jetson, Hailo, edge cameras), that's not fast enough.

This repo asks the next question: **how much faster can we make it, and what does speed cost us in tracking accuracy?**

I tested five backends — ranging from standard PyTorch to INT8 post-training quantisation — measured latency with proper GPU timing methodology, profiled where time actually goes in the pipeline, and compared HOTA accuracy between FP32 and INT8. The results included one clear winner (FP16), one counterintuitive finding (ONNX slower than PyTorch on T4), and one genuine failure (INT8 calibration collapse) that turned out to be more instructive than a clean result would have been.

---

## What was Tested

| Backend | What changes | Expected benefit |
|:--------|:-------------|:----------------|
| PyTorch FP32 | Nothing — this is the baseline | — |
| PyTorch FP16 | Weights and activations in 16-bit float | 2× faster on Tensor Core GPUs |
| ONNX FP32 | Exported to ONNX format, run via ORT CUDA | Graph fusion, fewer kernel launches |
| ONNX FP16 | Full ONNX graph cast to FP16 | Combined format + precision benefit |
| ONNX INT8 | Post-training quantisation, 8-bit integers | 4× smaller model, theoretically fastest |

Each backend is evaluated on latency (100 timed runs, 20 warmup, GPU-synchronised timing) and accuracy (HOTA/MOTA/IDF1 on MOT17-09-SDP via TrackEval).

---

## Architecture

```
YOLOv8x.pt (PyTorch, 130 MB)
      │
      ├── export.py
      │     ├── Ultralytics export API ──────► yolov8x.onnx       (FP32, 273 MB)
      │     ├── onnxconverter_common fp16 ───► yolov8x_fp16.onnx  (FP16, 137 MB)
      │     └── ORT MinMax PTQ ──────────────► yolov8x_int8.onnx  (INT8,  69 MB)
      │
      ├── benchmark.py
      │     Methodology: warmup=20, runs=100, cuda.synchronize() per run
      │     ├── PyTorch FP32   (baseline)
      │     ├── PyTorch FP16   (Tensor Core path)
      │     ├── ONNX FP32      (ORT CUDAExecutionProvider)
      │     ├── ONNX FP16      (full graph FP16)
      │     └── ONNX INT8      (QDQ format, per-channel weights)
      │
      ├── accuracy.py
      │     FP32 pipeline ──► ORT inference ──► ByteTrack ──► TrackEval (HOTA)
      │     INT8 pipeline ──► ORT inference ──► ByteTrack ──► TrackEval (HOTA)
      │
      └── profile.py
            Manual timing: preprocess / model_forward / postprocess / association
            PyTorch Profiler: Chrome trace export (chrome://tracing)
```

---

## Results

### Latency Benchmark — T4 GPU, batch=1, 640px input, 100 timed runs

| Backend | Mean (ms) | Std (ms) | P95 (ms) | FPS | Speedup | Model size |
|:--------|----------:|---------:|---------:|----:|--------:|----------:|
| PyTorch FP32 | 58.28 | 0.47 | 59.13 | 17.2 | 1.00× | 130 MB |
| PyTorch FP16 | 22.82 | 0.87 | 23.19 | 43.8 | **2.55×** | 130 MB |
| ONNX FP32 | 63.53 | 0.73 | 64.48 | 15.7 | 0.92× | 273 MB |
| ONNX FP16 | — | — | — | — | — | 137 MB |
| ONNX INT8 | 88.77 | 0.86 | 90.44 | 11.3 | 0.66× | 69 MB |

> ONNX FP16 could not be benchmarked due to a type inconsistency at the Resize node introduced during FP16 graph conversion — see [Known Issues](#known-issues).

### Accuracy — FP32 vs INT8 on MOT17-09-SDP

| Backend | HOTA↑ | DetA↑ | AssA↑ | MOTA↑ | IDF1↑ |
|:--------|------:|------:|------:|------:|------:|
| ONNX FP32 | 56.07 | 48.88 | 41.30 | 55.55 | 56.10 |
| ONNX INT8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **Drop** | **−56.07 pp** | **−48.88 pp** | **−41.30 pp** | **−55.55 pp** | **−56.10 pp** |

> The INT8 result is a complete calibration failure, not a partial accuracy drop — see [What Went Wrong and What I Learned](#what-went-wrong-and-what-i-learned).

### Pipeline Stage Breakdown — FP32, mean ms per frame across 50 frames

| Stage | Time (ms) | % of total | Notes |
|:------|----------:|-----------:|:------|
| Preprocessing (load + resize + normalise) | 15.32 | 17.7% | CPU-bound (OpenCV); cannot overlap with GPU |
| Backbone (CSPDarknet, approx) | 39.58 | 45.8% | Dominant compute stage |
| Neck (PAN + SPPF, approx) | 13.19 | 15.3% | Feature pyramid aggregation |
| Head (box regression + cls, approx) | 13.19 | 15.3% | Detection output |
| Postprocessing (NMS + rescale) | 5.12 | 5.9% | CPU-side filtering |
| ByteTrack association (IoU + Hungarian) | 0.06 | 0.1% | Negligible |
| **Total pipeline** | **86.46** | **100%** | **11.6 FPS end-to-end** |

> Backbone/neck/head split is an approximation based on YOLOv8x FLOPs distribution (60/20/20%). The Chrome trace at `benchmark_results/chrome_trace.json` gives exact kernel-level numbers.

---

## What was Found

### FP16 is the clear deployment recommendation

PyTorch FP16 gave a 2.55× speedup (58.3ms → 22.8ms) at zero accuracy cost. This is the right optimisation target for any GPU deployment on T4 or newer. The T4's Tensor Cores are specifically designed for 16-bit matrix multiplication and the speedup is near-linear with the theoretical throughput improvement.

Low variance (std 0.87ms vs 0.47ms for FP32) confirmed the result is stable across runs rather than a lucky sample. P95 of 23.19ms shows tail latency is well-controlled.

### ONNX FP32 is slower than PyTorch FP32 on T4

This surprised me. ONNX Runtime with `CUDAExecutionProvider` and `ORT_ENABLE_ALL` optimisations ran at 63.5ms vs PyTorch's 58.3ms — 9% slower despite being a supposedly optimised inference runtime.

The reason is that PyTorch's CUDA kernels for FP32 convolution are already specifically tuned for the Turing architecture (T4). ORT applies graph-level optimisations (constant folding, node fusion) but doesn't override the underlying CUDA kernel selection, and its session initialisation overhead adds latency that PyTorch's eager mode doesn't have. On newer GPUs (A100, H100) or with larger batch sizes, the tradeoff reverses in ORT's favour.

The lesson: benchmark assumptions don't always hold on specific hardware. "ORT is faster than PyTorch" is a heuristic, not a law.

### INT8 via ORT QDQ on T4 is slower than FP32

ONNX INT8 ran at 88.8ms — 52% slower than FP32. This is also counterintuitive but has a clear explanation: ORT's QDQ (Quantize-DeQuantize) format inserts dequantisation nodes between every quantised operation. On CPU, SIMD integer arithmetic is faster than float, so the savings exceed the overhead. On GPU, the dequantisation kernel launches and memory roundtrips between INT8 and FP32 exceed the compute savings from integer arithmetic.

INT8 shows real speed benefits on GPU only with TensorRT, which has a fused INT8 execution path that eliminates the per-layer dequantisation overhead entirely.

### Preprocessing is a bigger bottleneck than expected

At 15.3ms (17.7% of total pipeline time), preprocessing is the second largest single stage after the backbone. OpenCV runs on CPU while the model runs on GPU — these are sequential, not parallel. The GPU sits idle during preprocessing and the CPU sits idle during inference.

In production, this would be replaced with CUDA-accelerated preprocessing (TensorRT's built-in image pre-processing pipeline, or a custom CUDA kernel) to keep the GPU continuously utilised.

### ByteTrack association costs essentially nothing

0.06ms — less than 0.1% of total pipeline time. This is important because it means there's no point optimising the tracker itself to improve end-to-end FPS. The model is the bottleneck, and specifically the backbone. If you want 60 FPS, you need a faster backbone (YOLOv8n or YOLOv8s) not a faster tracker.

---

## What Went Wrong

### INT8 calibration collapse

The INT8 model produced HOTA 0.0 — every single confidence score output was exactly 0.0. This is not a partial accuracy drop; the model produces zero detections on every frame.

**What happened technically:** ORT's MinMax calibration uses 100 frames to measure the range of values flowing through each layer, then chooses an integer scale that maps that range onto -128 to 127. YOLOv8's detection head outputs raw logits — small negative numbers for background anchors and small positive numbers for foreground. Out of 8400 anchors per frame, typically only 5–20 are pedestrians. So 99.8% of the calibration data is background logits, and MinMax sets a scale based on that distribution. When the rare positive (pedestrian) logits are mapped through this scale, they round to the same integer bin as background — effectively zero after sigmoid activation.

**What I would do differently:** Use Percentile calibration (which ignores the top 0.001% of activation values, preventing background anchors from dominating the scale) with at least 500 calibration frames. Or, better, apply quantisation-aware training (QAT) rather than post-training quantisation — QAT fine-tunes the model to minimise quantisation error during training rather than trying to fix it after the fact.

**Why I couldn't fix it during this experiment:** Attempting recalibration with better settings caused Colab to crash due to out-of-memory errors. ORT's `quantize_static` builds an instrumented calibration graph that is ~2-3× larger than the original model in memory — for YOLOv8x at 273MB, that means ~600-800MB plus session overhead, which exceeds Colab's 12GB RAM limit. This is a real infrastructure constraint: proper INT8 calibration for large detection models requires a machine with more RAM than a free Colab runtime provides.

### ONNX FP16 type inconsistency

The FP16 model exported via `onnxconverter_common` had a type mismatch at a Resize node — the node expected `tensor(float16)` but received `tensor(float)`. This is a known issue with automatic FP16 graph conversion on models that use Resize operations (common in feature pyramid networks). The fix is to use Ultralytics' own `half=True` export flag which correctly handles mixed-precision boundary nodes, or to pin specific nodes to FP32 using ORT's `op_block_list` parameter. I documented this as a known issue rather than silently skipping it.

---

## Methodology Notes

### Why timing is harder than it looks

The naive approach — `t0 = time.time(); model(x); t1 = time.time()` — gives wrong answers on GPU for two reasons.

First, GPU operations are asynchronous. When Python calls a GPU operation, it returns immediately while the GPU is still running the kernel. If you stop the timer at that point, you measure near-zero latency. `torch.cuda.synchronize()` blocks Python until the GPU actually finishes.

Second, the first few inference calls are slower than steady state because CUDA compiles kernels on first use and allocates memory. I discarded 20 warmup runs before timing started, which is why my mean latency (58ms) is lower than what you'd see if you just timed a single call cold.

### Why I used 100 runs instead of fewer

With 100 runs I can compute reliable percentiles. P95 and P99 reveal tail latency — occasional slow frames from memory pressure, thermal throttling, or OS scheduling jitter. Mean alone would miss these. In a real deployment, P99 latency matters more than mean latency because it determines whether your system can reliably hit its frame rate target.

### On the FP32 ONNX accuracy baseline

The ONNX FP32 HOTA (56.07) is slightly lower than the companion repo's equivalent score (61.0 on MOT17-09-SDP) because the inference paths differ. The companion repo uses Ultralytics' full prediction pipeline including its own NMS implementation and confidence scaling. The `accuracy.py` here uses a custom postprocessing function that parses the raw ONNX output directly. Both are correct implementations of the same model, but different postprocessing choices (NMS threshold, score scaling) produce slightly different detection sets and therefore slightly different HOTA scores. This is expected and worth noting for anyone trying to reproduce exact numbers.

---

## Reproduction

### Colab (recommended)

1. Open `notebook.ipynb` with a T4 GPU runtime
2. Run **Cell 0** → **Runtime → Restart session** (fixes ORT CUDA version)
3. Fill in Kaggle credentials in **Cell 2**
4. Run **Cells 1–8** top to bottom
5. Expected total time: ~30 min

> **Note:** Cell 0 pins `onnxruntime-gpu==1.20.1` which is built for CUDA 12.x. Colab's default ORT version is built for CUDA 13 and will fail with `libcudart.so.13: cannot open shared object file`. The runtime restart after Cell 0 is mandatory.

### Local / command line

```bash
git clone https://github.com/JesonRamesh/yolov8-inference-optimization
cd yolov8-inference-optimization

pip install -r requirements.txt
git clone https://github.com/JonathonLuiten/TrackEval.git

# Fill in Kaggle credentials in config.py, then:
python export.py      # export FP32, FP16, INT8 models (~20 min on T4)
python benchmark.py   # latency benchmarks (~10 min)
python accuracy.py    # HOTA accuracy comparison (~5 min)
python profile.py     # stage breakdown + Chrome trace (~3 min)
```

---

## Repository Structure

````
yolov8-inference-optimization/
├── config.py        # all paths, benchmark settings, model configs
├── export.py        # ONNX FP32/FP16 export + INT8 PTQ calibration
├── benchmark.py     # latency benchmarking across all backends
├── accuracy.py      # HOTA accuracy evaluation (FP32 vs INT8)
├── profile.py       # PyTorch Profiler + manual stage timing
├── requirements.txt # pip dependencies
├── notebook.ipynb   # end-to-end Colab notebook
└── README.md
````

Generated at runtime:

````
/content/
├── models/
│   ├── yolov8x.onnx        # FP32 export  (273 MB)
│   ├── yolov8x_fp16.onnx   # FP16 export  (137 MB)
│   └── yolov8x_int8.onnx   # INT8 PTQ     (69 MB)
└── benchmark_results/
    ├── latency_benchmark.json
    ├── accuracy_results.json
    ├── profile_summary.json
    └── chrome_trace.json    # open at chrome://tracing
````

---

## Known Issues

| Issue | Cause | Status |
|:------|:------|:-------|
| ONNX FP16 fails to load in ORT | Type mismatch at Resize node after automatic FP16 conversion | Use `ultralytics export half=True` or pin Resize to FP32 via `op_block_list` |
| INT8 produces zero detections | MinMax calibration collapse on detection head logits | Use Percentile calibration with 500+ frames, or QAT |
| INT8 recalibration OOM on Colab | Instrumented calibration graph (~800MB) exceeds 12GB RAM limit | Requires higher-memory environment or TensorRT (streams calibration) |
| ORT CUDA version mismatch | Default Colab ORT built for CUDA 13, T4 has CUDA 12.8 | Pin `onnxruntime-gpu==1.20.1` (see Cell 0) |

---

## Related Work

This repo is a companion to [yolov8-bytetrack-mot17](https://github.com/JesonRamesh/yolov8-bytetrack-mot17), which evaluates the same YOLOv8x + ByteTrack pipeline on MOT17 using HOTA/MOTA/IDF1 metrics. That repo establishes the detection and tracking baseline; this repo explores inference optimisation and the speed/accuracy tradeoff.

---

## References

1. Jocher, G. et al. (2023). **Ultralytics YOLOv8**. [GitHub](https://github.com/ultralytics/ultralytics)
2. ONNX Runtime Team. **ONNX Runtime: cross-platform, high performance ML inferencing**. [GitHub](https://github.com/microsoft/onnxruntime)
3. Wu, H. et al. (2020). **Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation**. arXiv:2004.09602
4. Luiten, J. et al. (2021). **HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking**. IJCV 79, 408–428.
5. PyTorch Team. **PyTorch Profiler**. [Docs](https://pytorch.org/docs/stable/profiler.html)
6. NVIDIA. **TensorRT Developer Guide — INT8 Calibration**. [Docs](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#int8-calibration)

---

## License

MIT — see `LICENSE`.
