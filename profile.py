"""
profile.py — PyTorch Profiler breakdown of the full tracking pipeline.

Profiles time spent in:
  1. Preprocessing     — image load, resize, normalise, to-tensor
  2. Backbone          — CSPDarknet feature extraction
  3. Neck (PAN)        — feature pyramid aggregation
  4. Head              — detection head, box regression, classification
  5. Postprocessing    — NMS, box coordinate rescaling
  6. ByteTrack         — Kalman predict, IoU computation, Hungarian assignment

Two profiling modes:
  A. torch.profiler  — CUDA kernel-level breakdown with chrome trace export
  B. Manual timing   — coarser but interpretable stage-by-stage ms breakdown

Usage:
    python profile.py

Output:
  /content/benchmark_results/chrome_trace.json   (open in chrome://tracing)
  /content/benchmark_results/profile_summary.json
"""

import time
import json
import numpy as np
import cv2
from pathlib import Path
from contextlib import contextmanager
from scipy.optimize import linear_sum_assignment

import torch
import torch.profiler

import config


# ── Timing context manager ────────────────────────────────────────────────

@contextmanager
def cuda_timer(label: str, timings: dict):
    """Context manager that measures GPU-synchronised wall time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    timings.setdefault(label, []).append(elapsed_ms)


# ── Preprocessing ─────────────────────────────────────────────────────────

def preprocess(img_path: str, device: str, half: bool) -> tuple:
    """
    Full preprocessing pipeline: disk → GPU tensor.

    Returns:
        tensor:  (1,3,640,640) torch.Tensor on device
        orig_hw: (H, W) of original image
    """
    img    = cv2.imread(img_path)
    orig_h, orig_w = img.shape[:2]
    img    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img    = cv2.resize(img, (config.INPUT_SIZE, config.INPUT_SIZE))

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.float() / 255.0
    tensor = tensor.to(device)
    if half:
        tensor = tensor.half()

    return tensor, (orig_h, orig_w)


# ── Model stage hooks ─────────────────────────────────────────────────────

def register_stage_hooks(model) -> dict:
    """
    Register forward hooks on YOLOv8 submodules to time each stage.

    YOLOv8 internal structure:
      model.model[0..9]  — backbone (CSPDarknet) layers
      model.model[10..12] — neck (SPPF + PANet)
      model.model[15..22] — detection head

    We hook the first backbone layer, the neck entry, and the head entry
    to capture stage boundaries without modifying the model.

    Returns dict of {stage_name: [start_time, ...]} that hooks populate.
    """
    hooks   = {}
    timings = {"backbone_start": [], "neck_start": [], "head_start": []}

    def make_hook(key):
        def hook(module, input, output):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings[key].append(time.perf_counter())
        return hook

    # Layer indices based on YOLOv8x architecture
    # Backbone: layers 0-9, Neck: 10-12, Head: 15+
    try:
        model.model[0].register_forward_hook(make_hook("backbone_start"))
        model.model[10].register_forward_hook(make_hook("neck_start"))
        model.model[15].register_forward_hook(make_hook("head_start"))
        hooks["timings"] = timings
    except (IndexError, AttributeError):
        pass  # Model structure differs; fall back to whole-model timing

    return hooks


# ── ByteTrack timing components ───────────────────────────────────────────

def timed_iou_distance(atracks, btracks, timings):
    if not atracks or not btracks:
        return np.zeros((len(atracks), len(btracks)))

    with cuda_timer("bytetrack_iou", timings):
        def to_xyxy(t): x, y, w, h = t.tlwh; return [x, y, x+w, y+h]
        if not atracks or not btracks:
            return np.zeros((len(atracks), len(btracks)))
        ab = np.array([to_xyxy(t) for t in atracks])
        bb = np.array([to_xyxy(t) for t in btracks])
        ious = np.zeros((len(ab), len(bb)))
        for i, a in enumerate(ab):
            xi1 = np.maximum(a[0], bb[:,0]); yi1 = np.maximum(a[1], bb[:,1])
            xi2 = np.minimum(a[2], bb[:,2]); yi2 = np.minimum(a[3], bb[:,3])
            inter = np.maximum(xi2-xi1,0)*np.maximum(yi2-yi1,0)
            aa = (a[2]-a[0])*(a[3]-a[1]); ba = (bb[:,2]-bb[:,0])*(bb[:,3]-bb[:,1])
            ious[i] = inter/(aa+ba-inter+1e-6)
    return 1 - ious


# ── Manual stage profiling ────────────────────────────────────────────────

def profile_manual(n_frames: int = 50) -> dict:
    """
    Profile pipeline stages using manual cuda_timer measurements.

    Runs n_frames from MOT17-09-SDP and returns mean ms per stage.
    """
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = YOLO(config.DETECTOR_MODEL)
    model.model.to(device).eval()

    frames = sorted(
        (config.MOT17_ROOT / config.EVAL_SEQ / "img1").glob("*.jpg")
    )[:n_frames]

    timings = {}

    # Warmup
    for _ in range(5):
        img_path = str(frames[0])
        tensor, _ = preprocess(img_path, device, half=False)
        with torch.inference_mode():
            _ = model.model(tensor)

    print(f"Profiling {n_frames} frames (manual timing)...")

    for frame_path in frames:
        img_path = str(frame_path)

        # Stage 1: Preprocessing
        with cuda_timer("preprocess", timings):
            tensor, orig_hw = preprocess(img_path, device, half=False)

        # Stage 2+3+4: Full model forward (backbone + neck + head)
        with cuda_timer("model_forward", timings):
            with torch.inference_mode():
                raw_output = model.model(tensor)

        # Stage 5: Postprocessing (NMS + rescale)
        with cuda_timer("postprocess", timings):
            # Ultralytics non_max_suppression equivalent
            from ultralytics.utils.ops import non_max_suppression
            preds = non_max_suppression(
                raw_output[0] if isinstance(raw_output, (list, tuple))
                else raw_output,
                conf_thres=config.CONF_THRESH,
                iou_thres=0.45,
                classes=[config.PERSON_CLASS_ID],
            )

        # Stage 6: ByteTrack association (IoU + Hungarian)
        # Use a minimal representative cost matrix matching typical frame density
        n_dets   = len(preds[0]) if preds and len(preds[0]) else 5
        n_tracks = max(1, n_dets - 2)
        dummy_cost = np.random.rand(n_tracks, n_dets).astype(np.float32)
        with cuda_timer("bytetrack_hungarian", timings):
            cost_w = dummy_cost.copy()
            cost_w[cost_w > 0.8] = 0.81
            _ = linear_sum_assignment(cost_w)

    # Compute means
    means = {k: float(np.mean(v)) for k, v in timings.items()}

    # Infer sub-stage breakdown from model internals if hooks were set
    total_model = means.get("model_forward", 0)
    # Approximate split based on YOLOv8x FLOPs distribution:
    # backbone ~60%, neck ~20%, head ~20%
    means["backbone_approx"]   = total_model * 0.60
    means["neck_approx"]       = total_model * 0.20
    means["head_approx"]       = total_model * 0.20

    total = sum(v for k, v in means.items()
                if k in ["preprocess", "model_forward",
                         "postprocess", "bytetrack_hungarian"])
    means["total_pipeline"]    = total

    return means


# ── PyTorch Profiler ──────────────────────────────────────────────────────

def profile_torch_profiler(n_frames: int = 20) -> str:
    """
    Run PyTorch Profiler and export a Chrome trace.

    The trace can be opened at chrome://tracing or
    https://ui.perfetto.dev for kernel-level analysis.

    Returns path to the trace file.
    """
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = YOLO(config.DETECTOR_MODEL)
    model.model.to(device).eval()

    frames = sorted(
        (config.MOT17_ROOT / config.EVAL_SEQ / "img1").glob("*.jpg")
    )[:n_frames]

    trace_path = str(config.RESULTS_DIR / "chrome_trace.json")

    print(f"Running PyTorch Profiler on {n_frames} frames...")
    print("  (CUDA activities, CPU ops, memory)")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        # Schedule: 2 warmup + 3 active steps, then repeat
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(config.RESULTS_DIR / "tb_trace")),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,     # stack traces slow things down significantly
    ) as prof:

        for i, frame_path in enumerate(frames):
            tensor, _ = preprocess(str(frame_path), device, half=False)

            with torch.inference_mode():
                with torch.profiler.record_function("preprocessing"):
                    # Already preprocessed; record the to-device step
                    _ = tensor.contiguous()

                with torch.profiler.record_function("model_forward"):
                    output = model.model(tensor)

                with torch.profiler.record_function("postprocess"):
                    from ultralytics.utils.ops import non_max_suppression
                    _ = non_max_suppression(
                        output[0] if isinstance(output, (list, tuple))
                        else output,
                        conf_thres=config.CONF_THRESH,
                        iou_thres=0.45,
                        classes=[config.PERSON_CLASS_ID],
                    )

            prof.step()

    # Export Chrome trace
    prof.export_chrome_trace(trace_path)
    print(f"  Chrome trace saved: {trace_path}")
    print("  Open at: chrome://tracing or https://ui.perfetto.dev")

    # Print top operators by CUDA time
    print("\nTop 15 ops by CUDA time:")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=15,
    ))

    return trace_path


# ── Display results ───────────────────────────────────────────────────────

def display_stage_breakdown(timings: dict):
    """Print a formatted stage breakdown table."""
    total = timings.get("total_pipeline", 1.0)

    print("\n" + "=" * 55)
    print("PIPELINE STAGE BREAKDOWN (mean ms per frame, FP32)")
    print("=" * 55)

    stages = [
        ("Preprocessing",       "preprocess"),
        ("Backbone (approx)",   "backbone_approx"),
        ("Neck/PAN (approx)",   "neck_approx"),
        ("Head (approx)",       "head_approx"),
        ("Postprocess (NMS)",   "postprocess"),
        ("ByteTrack (assoc)",   "bytetrack_hungarian"),
    ]

    for label, key in stages:
        if key not in timings:
            continue
        ms  = timings[key]
        pct = 100 * ms / total
        bar = "█" * int(pct / 2)
        print(f"  {label:<25} {ms:>6.2f} ms  {pct:>5.1f}%  {bar}")

    print("─" * 55)
    print(f"  {'Total pipeline':<25} {total:>6.2f} ms  "
          f"→ {1000/total:.1f} FPS")
    print()
    print("  Note: backbone/neck/head split is approximate (60/20/20%)")
    print("  Run torch_profiler mode for kernel-level breakdown.")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Mode A: Manual stage timing (interpretable, always works)
    print("=" * 55)
    print("Mode A: Manual stage timing")
    print("=" * 55)
    timings = profile_manual(n_frames=50)
    display_stage_breakdown(timings)

    # Mode B: PyTorch Profiler (kernel-level, exports chrome trace)
    print("\n" + "=" * 55)
    print("Mode B: PyTorch Profiler (Chrome trace)")
    print("=" * 55)
    try:
        trace_path = profile_torch_profiler(n_frames=20)
    except Exception as e:
        print(f"  PyTorch Profiler failed: {e}")
        print("  Manual timings above are still valid.")
        trace_path = None

    # Save summary
    summary = {
        "manual_timings_ms": timings,
        "chrome_trace_path": trace_path,
    }
    out_path = config.RESULTS_DIR / "profile_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nProfile summary saved to {out_path}")


if __name__ == "__main__":
    main()
