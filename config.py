"""
config.py — Central configuration for YOLOv8x inference optimisation benchmarks.

Edit this file to change paths, benchmark settings, or which backends to test.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
MOT17_ROOT      = Path("/content/MOT17/train")
RESULTS_DIR     = Path("/content/benchmark_results")
MODELS_DIR      = Path("/content/models")

# ── Model ─────────────────────────────────────────────────────────────────
DETECTOR_MODEL  = "yolov8x.pt"
ONNX_MODEL      = MODELS_DIR / "yolov8x.onnx"
ONNX_FP16_MODEL = MODELS_DIR / "yolov8x_fp16.onnx"
ONNX_INT8_MODEL = MODELS_DIR / "yolov8x_int8.onnx"
INPUT_SIZE      = 640        # pixel size for inference input

# ── Benchmark settings ────────────────────────────────────────────────────
WARMUP_RUNS     = 20         # discarded before timing starts
BENCHMARK_RUNS  = 100        # timed runs averaged for latency
BATCH_SIZE      = 1          # latency benchmarks use batch=1
CONF_THRESH     = 0.35
PERSON_CLASS_ID = 0          # COCO class 0 = person

# ── INT8 calibration ──────────────────────────────────────────────────────
# Sequence used for INT8 calibration and HOTA accuracy drop measurement
CALIBRATION_SEQ = "MOT17-09-SDP"   # short sequence (525 frames), medium density
CALIBRATION_FRAMES = 100           # frames sampled for PTQ calibration

# ── HOTA accuracy evaluation ──────────────────────────────────────────────
EVAL_SEQ        = "MOT17-09-SDP"
EVAL_SEQ_LEN    = 525
TRACKEVAL_DIR   = Path("/content/TrackEval")
TRACKER_NAME    = "YOLOv8x"        # base name; precision suffix appended per backend
