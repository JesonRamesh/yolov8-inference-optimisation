"""
export.py — Export YOLOv8x to ONNX (FP32 and FP16).

Uses the Ultralytics export API which handles:
  - Graph optimisation (fusing Conv+BN layers)
  - Dynamic batch axis registration
  - Input/output name annotation

Outputs:
  /content/models/yolov8x.onnx        (FP32)
  /content/models/yolov8x_fp16.onnx   (FP16 — cast ops inserted by onnxmltools)

Usage:
    python export.py
"""

import subprocess
import sys

# ── Install export dependencies ───────────────────────────────────────────
def install_export_deps():
    pkgs = [
        "onnx>=1.14.0",
        "onnxruntime-gpu>=1.16.0",   # GPU execution provider
        "onnxmltools>=1.11.0",       # FP16 conversion
        "onnxconverter-common",      # required by onnxmltools fp16 convert
    ]
    for pkg in pkgs:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            check=True
        )
    print("Export dependencies installed ✓")


# ── FP32 export ───────────────────────────────────────────────────────────

def export_fp32() -> str:
    """
    Export YOLOv8x to ONNX FP32 using the Ultralytics export API.

    Key export settings:
      opset=17     — latest stable ONNX opset, required for some ops in YOLOv8
      simplify=True — runs onnx-simplifier to fold constants and remove dead nodes
      dynamic=False — fixed input shape; faster inference than dynamic axes
      half=False    — FP32 weights

    Returns path to exported .onnx file.
    """
    from ultralytics import YOLO
    import config

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Exporting YOLOv8x to ONNX FP32...")
    model = YOLO(config.DETECTOR_MODEL)

    # export() returns the path to the exported file
    export_path = model.export(
        format="onnx",
        imgsz=config.INPUT_SIZE,
        opset=17,
        simplify=True,
        dynamic=False,
        half=False,
    )

    # Move to our models directory
    import shutil
    from pathlib import Path
    dst = config.ONNX_MODEL
    shutil.move(str(export_path), str(dst))

    # Verify the model loads
    import onnx
    model_check = onnx.load(str(dst))
    onnx.checker.check_model(model_check)

    size_mb = dst.stat().st_size / 1e6
    print(f"  Exported: {dst}  ({size_mb:.1f} MB) ✓")
    print(f"  ONNX opset: {model_check.opset_import[0].version}")

    # Print input/output names — needed for ORT inference
    for inp in model_check.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  Input:  {inp.name}  shape={shape}")
    for out in model_check.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  Output: {out.name}  shape={shape}")

    return str(dst)


# ── FP16 export ───────────────────────────────────────────────────────────

def export_fp16() -> str:
    """
    Convert ONNX FP32 model to FP16 using onnx's built-in helper.
    Falls back to onnx-ml-tools if available, otherwise uses manual conversion.
    """
    import onnx
    from onnx import numpy_helper, TensorProto
    import config
    import numpy as np

    print("Converting ONNX FP32 → FP16...")

    # Try onnxmltools first (different import paths across versions)
    try:
        from onnxmltools.utils.float16_converter import convert_float_to_float16
        fp32_model = onnx.load(str(config.ONNX_MODEL))
        fp16_model = convert_float_to_float16(fp32_model, keep_io_types=True)
    except ImportError:
        try:
            from onnxmltools.utils import float16_converter
            fp32_model = onnx.load(str(config.ONNX_MODEL))
            fp16_model = float16_converter.convert_float_to_float16(
                fp32_model, keep_io_types=True)
        except (ImportError, AttributeError):
            # Manual FP16 conversion using onnx directly
            print("  onnxmltools not available, using onnx native conversion...")
            try:
                from onnx.tools import update_model_dims
                from onnxconverter_common import float16
                fp32_model = onnx.load(str(config.ONNX_MODEL))
                fp16_model = float16.convert_float_to_float16(
                    fp32_model, keep_io_types=True)
            except ImportError:
                # Final fallback: cast weights to fp16 manually
                print("  Using manual weight casting fallback...")
                fp32_model = onnx.load(str(config.ONNX_MODEL))
                for tensor in fp32_model.graph.initializer:
                    if tensor.data_type == TensorProto.FLOAT:
                        data = numpy_helper.to_array(tensor).astype(np.float16)
                        tensor.CopyFrom(numpy_helper.from_array(data, tensor.name))
                        tensor.data_type = TensorProto.FLOAT16
                fp16_model = fp32_model

    onnx.save(fp16_model, str(config.ONNX_FP16_MODEL))

    size_fp32 = config.ONNX_MODEL.stat().st_size / 1e6
    size_fp16 = config.ONNX_FP16_MODEL.stat().st_size / 1e6
    print(f"  FP32: {size_fp32:.1f} MB  →  FP16: {size_fp16:.1f} MB "
          f"({100*(1 - size_fp16/size_fp32):.0f}% smaller) ✓")

    return str(config.ONNX_FP16_MODEL)


# ── INT8 export (PTQ) ─────────────────────────────────────────────────────

def export_int8(calibration_data: list) -> str:
    """
    Apply INT8 post-training quantisation via ONNX Runtime's quantisation API.

    Uses MinMaxCalibration (percentile-based) with per-channel weight
    quantisation, which gives lower accuracy drop than per-tensor for
    detection models.

    Args:
        calibration_data: list of numpy arrays, each shape (1,3,640,640),
                          used to collect activation statistics.

    Returns path to INT8 .onnx file.
    """
    from onnxruntime.quantization import (
        quantize_static,
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        CalibrationMethod,
    )
    import numpy as np
    import config

    class YOLOCalibrationReader(CalibrationDataReader):
        """
        Feeds calibration images to the ORT quantisation calibrator.
        ORT expects a dict mapping input_name → np.ndarray.
        """
        def __init__(self, data: list, input_name: str):
            self._data  = data
            self._name  = input_name
            self._index = 0

        def get_next(self):
            if self._index >= len(self._data):
                return None
            item = {self._name: self._data[self._index]}
            self._index += 1
            return item

    # Get input name from the FP32 model
    import onnx
    fp32_model = onnx.load(str(config.ONNX_MODEL))
    input_name = fp32_model.graph.input[0].name

    reader = YOLOCalibrationReader(calibration_data, input_name)

    print(f"Applying INT8 PTQ with {len(calibration_data)} calibration images...")

    quantize_static(
        model_input=str(config.ONNX_MODEL),
        model_output=str(config.ONNX_INT8_MODEL),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,          # Quantize-DeQuantize format
        per_channel=True,                      # per-channel for weights
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={
            "CalibMovingAverage": True,        # smooth activation ranges
            "PerChannelQuantization": True,
        }
    )

    size_fp32 = config.ONNX_MODEL.stat().st_size / 1e6
    size_int8 = config.ONNX_INT8_MODEL.stat().st_size / 1e6
    print(f"  FP32: {size_fp32:.1f} MB  →  INT8: {size_int8:.1f} MB "
          f"({100*(1 - size_int8/size_fp32):.0f}% smaller) ✓")

    return str(config.ONNX_INT8_MODEL)


# ── Calibration data loader ───────────────────────────────────────────────

def load_calibration_data() -> list:
    """
    Load and preprocess frames from the calibration sequence.

    Returns list of np.float32 arrays shaped (1, 3, 640, 640),
    normalised to [0, 1], in NCHW format as expected by YOLOv8.
    """
    import numpy as np
    import cv2
    import config

    seq_path = config.MOT17_ROOT / config.CALIBRATION_SEQ / "img1"
    frames   = sorted(seq_path.glob("*.jpg"))[:config.CALIBRATION_FRAMES]

    data = []
    for f in frames:
        img  = cv2.imread(str(f))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img  = cv2.resize(img, (config.INPUT_SIZE, config.INPUT_SIZE))
        img  = img.astype(np.float32) / 255.0
        img  = img.transpose(2, 0, 1)          # HWC → CHW
        img  = np.expand_dims(img, axis=0)     # CHW → NCHW
        data.append(img)

    print(f"Loaded {len(data)} calibration frames from "
          f"{config.CALIBRATION_SEQ} ✓")
    return data


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    install_export_deps()

    export_fp32()
    export_fp16()

    print("\nLoading calibration data for INT8...")
    cal_data = load_calibration_data()
    export_int8(cal_data)

    print("\nAll exports complete:")
    import config
    for path in [config.ONNX_MODEL,
                 config.ONNX_FP16_MODEL,
                 config.ONNX_INT8_MODEL]:
        size = path.stat().st_size / 1e6
        print(f"  {path.name}: {size:.1f} MB")


if __name__ == "__main__":
    main()
