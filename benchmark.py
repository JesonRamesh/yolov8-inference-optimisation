"""
benchmark.py — Latency benchmarking across PyTorch, ONNX FP32, FP16, and INT8.

Benchmarking methodology:
  - Warmup runs:     20 forward passes discarded (GPU cache warmup, JIT compilation)
  - Timed runs:      100 forward passes measured individually
  - Metric:          mean ± std latency in ms, P50, P95, P99 percentiles
  - Input:           single real MOT17 frame (batch=1) at 640×640
  - Synchronisation: torch.cuda.synchronize() before each timer stop on GPU
  - Timer:           time.perf_counter() (nanosecond resolution)

Why this matters:
  - Without warmup, first runs include CUDA kernel compilation and memory allocation
  - P95/P99 percentiles reveal tail latency spikes from memory pressure
  - Per-run timing (not total/N) captures variance correctly

Usage:
    python benchmark.py
"""

import time
import json
import numpy as np
from pathlib import Path

import config


# ── Image preprocessing ───────────────────────────────────────────────────

def load_test_frame() -> np.ndarray:
    """
    Load one real MOT17 frame and preprocess to model input format.

    Returns np.float32 array of shape (1, 3, 640, 640), values in [0, 1].
    Using a real frame (not random noise) matters — real images trigger
    realistic activation patterns and memory access patterns.
    """
    import cv2

    frame_path = (config.MOT17_ROOT / config.CALIBRATION_SEQ
                  / "img1" / "000001.jpg")
    img = cv2.imread(str(frame_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (config.INPUT_SIZE, config.INPUT_SIZE))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)       # HWC → CHW
    img = np.expand_dims(img, axis=0)  # → NCHW
    return img


# ── PyTorch baseline ──────────────────────────────────────────────────────

def benchmark_pytorch(img_np: np.ndarray) -> dict:
    """
    Benchmark YOLOv8x PyTorch inference (FP32 and FP16).

    Uses torch.inference_mode() and cuda.synchronize() for accurate timing.
    """
    import torch
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    for precision, half in [("fp32", False), ("fp16", True)]:
        print(f"\nBenchmarking PyTorch {precision.upper()}...")
        model = YOLO(config.DETECTOR_MODEL)
        model.model.to(device)
        if half:
            model.model.half()
        model.model.eval()

        # Convert input to correct dtype and device
        dtype  = torch.float16 if half else torch.float32
        tensor = torch.from_numpy(img_np).to(device=device, dtype=dtype)

        # Warmup
        print(f"  Warmup ({config.WARMUP_RUNS} runs)...")
        with torch.inference_mode():
            for _ in range(config.WARMUP_RUNS):
                _ = model.model(tensor)
                if device == "cuda":
                    torch.cuda.synchronize()

        # Timed runs
        print(f"  Timing ({config.BENCHMARK_RUNS} runs)...")
        latencies = []
        with torch.inference_mode():
            for _ in range(config.BENCHMARK_RUNS):
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model.model(tensor)
                if device == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)  # ms

        stats = _compute_stats(latencies, f"PyTorch {precision.upper()}")
        results[f"pytorch_{precision}"] = stats

        # Free memory before next backend
        del model, tensor
        if device == "cuda":
            torch.cuda.empty_cache()

    return results


# ── ONNX Runtime ──────────────────────────────────────────────────────────

def _create_ort_session(model_path: str, use_cuda: bool = True):
    """
    Create an ONNX Runtime InferenceSession with optimal provider settings.

    Provider priority:
      CUDAExecutionProvider  — GPU (TensorRT kernels where possible)
      CPUExecutionProvider   — CPU fallback for unsupported ops

    Session options:
      graph_optimization_level = ORT_ENABLE_ALL
        Enables all graph optimisations: constant folding, common
        subexpression elimination, node fusion (Conv+BN, etc.)
      execution_mode = ORT_SEQUENTIAL
        Sequential execution is faster than parallel for single-batch inference
        because it avoids thread synchronisation overhead.
    """
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_opts.intra_op_num_threads = 1   # single-threaded for fair latency measurement

    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if use_cuda else ["CPUExecutionProvider"])

    session = ort.InferenceSession(
        model_path,
        sess_options=sess_opts,
        providers=providers,
    )

    # Log which provider is actually being used
    active = session.get_providers()
    print(f"  Active providers: {active}")
    return session


def _run_ort_benchmark(session, img_np: np.ndarray,
                       input_name: str, label: str) -> dict:
    """Run warmup + timed benchmark for one ORT session."""
    # Warmup
    print(f"  Warmup ({config.WARMUP_RUNS} runs)...")
    for _ in range(config.WARMUP_RUNS):
        session.run(None, {input_name: img_np})

    # Timed runs
    print(f"  Timing ({config.BENCHMARK_RUNS} runs)...")
    latencies = []
    for _ in range(config.BENCHMARK_RUNS):
        t0 = time.perf_counter()
        session.run(None, {input_name: img_np})
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    return _compute_stats(latencies, label)


def benchmark_onnx(img_np: np.ndarray) -> dict:
    """Benchmark ONNX Runtime across FP32, FP16, and INT8 models."""
    import onnxruntime as ort
    import onnx

    results = {}
    backends = [
        ("fp32", config.ONNX_MODEL,      "ONNX FP32"),
        ("fp16", config.ONNX_FP16_MODEL, "ONNX FP16"),
        ("int8", config.ONNX_INT8_MODEL, "ONNX INT8"),
    ]

    for key, model_path, label in backends:
        if not model_path.exists():
            print(f"\nSkipping {label} — model not found at {model_path}")
            continue

        print(f"\nBenchmarking {label}...")
        session    = _create_ort_session(str(model_path))
        input_name = session.get_inputs()[0].name

        # INT8 model uses float32 inputs (cast nodes handle the conversion)
        stats = _run_ort_benchmark(session, img_np, input_name, label)
        results[f"onnx_{key}"] = stats
        del session

    return results


# ── Statistics ────────────────────────────────────────────────────────────

def _compute_stats(latencies: list, label: str) -> dict:
    """Compute and print latency statistics from a list of ms values."""
    arr  = np.array(latencies)
    stats = {
        "label":    label,
        "mean_ms":  float(np.mean(arr)),
        "std_ms":   float(np.std(arr)),
        "p50_ms":   float(np.percentile(arr, 50)),
        "p95_ms":   float(np.percentile(arr, 95)),
        "p99_ms":   float(np.percentile(arr, 99)),
        "min_ms":   float(np.min(arr)),
        "max_ms":   float(np.max(arr)),
        "fps":      float(1000.0 / np.mean(arr)),
        "n_runs":   len(latencies),
    }
    print(f"  {label}:")
    print(f"    Mean ± Std: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")
    print(f"    P50 / P95 / P99: "
          f"{stats['p50_ms']:.2f} / {stats['p95_ms']:.2f} / {stats['p99_ms']:.2f} ms")
    print(f"    FPS: {stats['fps']:.1f}")
    return stats


# ── Results table ─────────────────────────────────────────────────────────

def print_results_table(all_results: dict):
    """Print a formatted comparison table of all backends."""
    # Use PyTorch FP32 as baseline for speedup calculation
    baseline = all_results.get("pytorch_fp32", {}).get("mean_ms", None)

    print("\n" + "=" * 75)
    print(f"{'Backend':<22} {'Mean ms':>9} {'Std ms':>8} "
          f"{'P95 ms':>8} {'FPS':>8} {'Speedup':>9}")
    print("─" * 75)

    order = ["pytorch_fp32", "pytorch_fp16",
             "onnx_fp32", "onnx_fp16", "onnx_int8"]

    for key in order:
        if key not in all_results:
            continue
        r = all_results[key]
        speedup = (f"{baseline / r['mean_ms']:.2f}×"
                   if baseline else "—")
        print(f"  {r['label']:<20} {r['mean_ms']:>9.2f} {r['std_ms']:>8.2f} "
              f"{r['p95_ms']:>8.2f} {r['fps']:>8.1f} {speedup:>9}")

    print("─" * 75)
    print(f"  Warmup: {config.WARMUP_RUNS} runs discarded  |  "
          f"Timed: {config.BENCHMARK_RUNS} runs  |  "
          f"Batch size: {config.BATCH_SIZE}  |  Input: {config.INPUT_SIZE}px")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test frame...")
    img_np = load_test_frame()
    print(f"  Input shape: {img_np.shape}  dtype: {img_np.dtype}")

    all_results = {}

    # PyTorch baselines
    all_results.update(benchmark_pytorch(img_np))

    # ONNX Runtime backends
    all_results.update(benchmark_onnx(img_np))

    # Summary table
    print_results_table(all_results)

    # Save JSON
    out_path = config.RESULTS_DIR / "latency_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
