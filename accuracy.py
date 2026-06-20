"""
accuracy.py — HOTA accuracy evaluation for INT8 vs FP32 backends.

Runs full detection + ByteTrack tracking on MOT17-09-SDP using both
FP32 and INT8 backends, then evaluates HOTA/MOTA/IDF1 for both and
reports the accuracy drop from quantisation.

The ByteTrack implementation is imported from the companion repo
(yolov8-bytetrack-mot17). If that repo is not cloned, a minimal
inline version is used automatically.

Usage:
    python accuracy.py
"""

import sys
import time
import json
import shutil
import numpy as np
import configparser
import cv2
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

import config


# ── Inline ByteTrack (minimal, matches companion repo implementation) ──────

def linear_assignment(cost, thresh):
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    cost_w = cost.copy()
    cost_w[cost_w > thresh] = thresh + 1e-4
    row_ind, col_ind = linear_sum_assignment(cost_w)
    matches, matched_r, matched_c = [], set(), set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= thresh:
            matches.append([r, c]); matched_r.add(r); matched_c.add(c)
    u_rows = [r for r in range(cost.shape[0]) if r not in matched_r]
    u_cols = [c for c in range(cost.shape[1]) if c not in matched_c]
    return matches, u_rows, u_cols


def iou_distance(atracks, btracks):
    if not atracks or not btracks:
        return np.zeros((len(atracks), len(btracks)))
    def to_xyxy(t): x, y, w, h = t.tlwh; return [x, y, x+w, y+h]
    ab = np.array([to_xyxy(t) for t in atracks])
    bb = np.array([to_xyxy(t) for t in btracks])
    ious = np.zeros((len(ab), len(bb)))
    for i, a in enumerate(ab):
        xi1 = np.maximum(a[0], bb[:,0]); yi1 = np.maximum(a[1], bb[:,1])
        xi2 = np.minimum(a[2], bb[:,2]); yi2 = np.minimum(a[3], bb[:,3])
        inter = np.maximum(xi2-xi1,0) * np.maximum(yi2-yi1,0)
        aa = (a[2]-a[0])*(a[3]-a[1]); ba = (bb[:,2]-bb[:,0])*(bb[:,3]-bb[:,1])
        ious[i] = inter / (aa + ba - inter + 1e-6)
    return 1 - ious


class TrackState:
    New=0; Tracked=1; Lost=2; Removed=3

class STrack:
    _count = 0
    def __init__(self, tlwh, score):
        self._tlwh=np.array(tlwh,dtype=float); self.score=score
        self.is_activated=False; self.state=TrackState.New
        self.mean=None; self.covariance=None; self.tracklet_len=0
        self.frame_id=0; self.start_frame=0; self.kalman_filter=None
        STrack._count+=1; self.track_id=STrack._count

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret=np.array(tlwh,dtype=float); ret[:2]+=ret[2:]/2; ret[2]/=ret[3]; return ret

    @property
    def tlwh(self):
        if self.mean is None: return self._tlwh.copy()
        ret=self.mean[:4].copy(); ret[2]*=ret[3]; ret[:2]-=ret[2:]/2; return ret

    def _kf_init(self):
        ndim,dt=4,1.
        F=np.eye(2*ndim,2*ndim)
        for i in range(ndim): F[i,ndim+i]=dt
        return F, np.eye(ndim,2*ndim), 1./20, 1./160

    def activate(self, frame_id):
        F,H,sp,sv=self._kf_init(); m=self.tlwh_to_xyah(self._tlwh)
        mean=np.r_[m,np.zeros_like(m)]
        std=[2*sp*m[3],2*sp*m[3],1e-2,2*sp*m[3],
             10*sv*m[3],10*sv*m[3],1e-5,10*sv*m[3]]
        self.mean,self.covariance=mean,np.diag(np.square(std))
        self._F,self._H,self._sp,self._sv=F,H,sp,sv
        self.state=TrackState.Tracked; self.is_activated=True
        self.frame_id=frame_id; self.start_frame=frame_id; self.tracklet_len=0

    def predict(self):
        m=self.mean; sp,sv=self._sp,self._sv
        std=[sp*m[3],sp*m[3],1e-2,sp*m[3],sv*m[3],sv*m[3],1e-5,sv*m[3]]
        Q=np.diag(np.square(std))
        self.mean=self._F@m; self.covariance=self._F@self.covariance@self._F.T+Q

    def update(self, new_track, frame_id):
        m=self.tlwh_to_xyah(new_track._tlwh); sp=self._sp
        std=[sp*self.mean[3],sp*self.mean[3],1e-1,sp*self.mean[3]]
        R=np.diag(np.square(std))
        S=self._H@self.covariance@self._H.T+R
        K=self.covariance@self._H.T@np.linalg.inv(S)
        self.mean=self.mean+K@(m-self._H@self.mean)
        self.covariance=self.covariance-K@S@K.T
        self.state=TrackState.Tracked; self.is_activated=True
        self.frame_id=frame_id; self.tracklet_len+=1; self.score=new_track.score

    def re_activate(self, new_track, frame_id):
        self.update(new_track, frame_id); self.tracklet_len=0


class ByteTracker:
    def __init__(self, fps=30):
        self.high_thresh=0.6; self.low_thresh=0.1; self.new_thresh=0.7
        self.match_thresh=0.8; self.buffer=int(fps)
        self.tracked=[]; self.lost=[]; self.frame_id=0; STrack._count=0

    def update(self, dets):
        self.frame_id+=1; fid=self.frame_id
        scores=dets[:,4] if len(dets) else np.array([])
        def make(d): return [STrack([r[0],r[1],r[2]-r[0],r[3]-r[1]],r[4]) for r in d]
        high=make(dets[scores>=self.high_thresh]) if len(dets) else []
        low=make(dets[(scores>=self.low_thresh)&(scores<self.high_thresh)]) if len(dets) else []
        for t in self.tracked+self.lost: t.predict()
        active=[t for t in self.tracked if t.is_activated]
        inactive=[t for t in self.tracked if not t.is_activated]
        pool=active+inactive
        m1,u_trk1,u_det1=linear_assignment(iou_distance(pool,high),self.match_thresh)
        for ti,di in m1: pool[ti].update(high[di],fid)
        u_active=[active[i] for i in u_trk1 if i<len(active)]
        m2,u_trk2,_=linear_assignment(iou_distance(u_active,low),0.5)
        for ti,di in m2: u_active[ti].update(low[di],fid)
        newly_lost=[u_active[i] for i in u_trk2]
        for t in newly_lost: t.state=TrackState.Lost
        unmatched_high=[high[i] for i in u_det1]
        m3,_,u_det2=linear_assignment(iou_distance(self.lost,unmatched_high),0.5)
        for ti,di in m3: self.lost[ti].re_activate(unmatched_high[di],fid)
        for i in u_det2:
            d=unmatched_high[i]
            if d.score>=self.new_thresh: d.activate(fid); self.tracked.append(d)
        reactivated=[self.lost[ti] for ti,_ in m3]
        self.lost=[t for t in self.lost if t not in reactivated]+newly_lost
        self.lost=[t for t in self.lost if fid-t.frame_id<=self.buffer]
        self.tracked=[t for t in self.tracked+reactivated if t.state==TrackState.Tracked]
        return [t for t in self.tracked if t.is_activated]


# ── ONNX Runtime inference ────────────────────────────────────────────────

def create_ort_session(model_path: str):
    import onnxruntime as ort
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        model_path,
        sess_options=sess_opts,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )


def preprocess_frame(img_path: str) -> tuple:
    """
    Preprocess a frame for YOLOv8 inference.

    Returns:
        blob:     (1,3,640,640) float32 numpy array
        orig_h:   original image height (for box rescaling)
        orig_w:   original image width
    """
    img    = cv2.imread(img_path)
    orig_h, orig_w = img.shape[:2]
    img    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img    = cv2.resize(img, (config.INPUT_SIZE, config.INPUT_SIZE))
    blob   = img.astype(np.float32) / 255.0
    blob   = blob.transpose(2, 0, 1)[np.newaxis, ...]   # NCHW
    return blob, orig_h, orig_w


def postprocess(output: np.ndarray, orig_h: int, orig_w: int,
                conf_thresh: float = config.CONF_THRESH) -> np.ndarray:
    """
    Parse YOLOv8 ONNX output to [x1,y1,x2,y2,conf] detections with NMS.

    YOLOv8 ONNX output shape: (1, 84, 8400)
      - 84 = 4 (cx,cy,w,h) + 80 (COCO class scores)
      - We filter to class 0 (person) only, then apply NMS
    """
    import torch
    import torchvision

    preds = output[0].transpose(1, 0)   # (8400, 84)

    cx, cy = preds[:, 0], preds[:, 1]
    w,  h  = preds[:, 2], preds[:, 3]
    person_scores = preds[:, 4 + config.PERSON_CLASS_ID]

    mask = person_scores >= conf_thresh
    if mask.sum() == 0:
        return np.empty((0, 5), dtype=np.float32)

    cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
    scores = person_scores[mask]

    # Rescale to original image coordinates
    scale_x = orig_w / config.INPUT_SIZE
    scale_y = orig_h / config.INPUT_SIZE
    x1 = (cx - w / 2) * scale_x
    y1 = (cy - h / 2) * scale_y
    x2 = (cx + w / 2) * scale_x
    y2 = (cy + h / 2) * scale_y

    # NMS — removes duplicate boxes for the same person
    boxes_t  = torch.from_numpy(
        np.stack([x1, y1, x2, y2], axis=1)).float()
    scores_t = torch.from_numpy(scores).float()
    keep     = torchvision.ops.nms(boxes_t, scores_t, iou_threshold=0.45)
    keep     = keep.numpy()

    dets = np.stack([x1[keep], y1[keep],
                     x2[keep], y2[keep],
                     scores[keep]], axis=1).astype(np.float32)
    return dets

# ── Full pipeline run ─────────────────────────────────────────────────────

def run_pipeline(model_path: str, label: str) -> list:
    """
    Run full detection + tracking on config.EVAL_SEQ.

    Returns list of MOT-format track strings.
    """
    session    = create_ort_session(model_path)
    input_name = session.get_inputs()[0].name

    seq_path = config.MOT17_ROOT / config.EVAL_SEQ
    cfg      = configparser.ConfigParser()
    cfg.read(seq_path / "seqinfo.ini")
    fps     = float(cfg["Sequence"].get("frameRate", 30))
    n_frames = int(cfg["Sequence"]["seqLength"])
    frames   = sorted((seq_path / "img1").glob("*.jpg"))

    tracker = ByteTracker(fps=fps)
    lines   = []
    t0      = time.time()

    for frame_path in tqdm(frames, desc=f"  {label}"):
        blob, orig_h, orig_w = preprocess_frame(str(frame_path))
        output = session.run(None, {input_name: blob})
        dets   = postprocess(output[0], orig_h, orig_w)
        fid    = int(frame_path.stem)

        for t in tracker.update(dets):
            x, y, w, h = t.tlwh
            lines.append(
                f"{fid},{t.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
                f"{t.score:.4f},-1,-1,-1"
            )

    elapsed = time.time() - t0
    print(f"  {label}: {len(lines)} track rows | "
          f"{n_frames/elapsed:.1f} FPS (end-to-end)")
    return lines


# ── TrackEval evaluation ──────────────────────────────────────────────────

def evaluate_tracks(track_lines: list, tracker_name: str) -> dict:
    """Run TrackEval on track lines and return metrics dict."""
    sys.path.insert(0, str(config.TRACKEVAL_DIR))
    import trackeval

    gt_dir      = config.TRACKEVAL_DIR / "data/gt/mot_challenge/MOT17-train"
    tracker_dir = (config.TRACKEVAL_DIR / "data/trackers/mot_challenge"
                   / "MOT17-train" / tracker_name / "data")

    for d in [gt_dir, tracker_dir]:
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True)

    seq = config.EVAL_SEQ
    (gt_dir / seq / "gt").mkdir(parents=True)
    (gt_dir / seq / "seqinfo.ini").write_text(
        (config.MOT17_ROOT / seq / "seqinfo.ini").read_text())
    (gt_dir / seq / "gt" / "gt.txt").write_bytes(
        (config.MOT17_ROOT / seq / "gt" / "gt.txt").read_bytes())
    (tracker_dir / f"{seq}.txt").write_text("\n".join(track_lines))

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({"USE_PARALLEL": False, "PRINT_RESULTS": False,
                        "TIME_PROGRESS": False, "PLOT_CURVES": False,
                        "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False})

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": str(config.TRACKEVAL_DIR / "data/gt/mot_challenge"),
        "TRACKERS_FOLDER": str(config.TRACKEVAL_DIR / "data/trackers/mot_challenge"),
        "BENCHMARK": "MOT17", "SPLIT_TO_EVAL": "train",
        "TRACKERS_TO_EVAL": [tracker_name],
        "CLASSES_TO_EVAL": ["pedestrian"],
        "TRACKER_SUB_FOLDER": "data", "SKIP_SPLIT_FOL": False,
        "PLOT_CURVES": False,
        "SEQ_INFO": {seq: config.EVAL_SEQ_LEN},
    })

    evaluator    = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    results, _   = evaluator.evaluate(dataset_list, [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(),
        trackeval.metrics.Identity(),
    ])

    seq_res = results["MotChallenge2DBox"][tracker_name][seq]["pedestrian"]

    def val(sub, key):
        v = seq_res[sub][key]
        v = float(v.mean()) if isinstance(v, np.ndarray) else float(v)
        return round(v * 100 if abs(v) <= 1.0 else v, 2)

    return {
        "HOTA":      val("HOTA",     "HOTA(0)"),
        "DetA":      val("HOTA",     "DetA"),
        "AssA":      val("HOTA",     "AssA"),
        "MOTA":      val("CLEAR",    "MOTA"),
        "IDF1":      val("Identity", "IDF1"),
        "Recall":    val("CLEAR",    "CLR_Re"),
        "Precision": val("CLEAR",    "CLR_Pr"),
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = {}

    backends = [
        ("fp32", str(config.ONNX_MODEL),      "YOLOv8x-FP32"),
        ("int8", str(config.ONNX_INT8_MODEL),  "YOLOv8x-INT8"),
    ]

    for key, model_path, tracker_name in backends:
        if not Path(model_path).exists():
            print(f"Skipping {tracker_name} — model not found")
            continue

        print(f"\nRunning pipeline: {tracker_name}")
        tracks  = run_pipeline(model_path, tracker_name)
        print(f"  Evaluating HOTA...")
        metrics = evaluate_tracks(tracks, tracker_name)
        all_metrics[key] = {"tracker_name": tracker_name, **metrics}
        print(f"  HOTA={metrics['HOTA']}  MOTA={metrics['MOTA']}  "
              f"IDF1={metrics['IDF1']}")

    # Accuracy drop table
    print("\n" + "=" * 65)
    print(f"ACCURACY: {config.EVAL_SEQ} (FP32 vs INT8)")
    print("=" * 65)
    header = f"{'Backend':<20} {'HOTA':>7} {'DetA':>7} {'AssA':>7} {'MOTA':>7} {'IDF1':>7}"
    print(header)
    print("─" * 65)

    fp32 = all_metrics.get("fp32", {})
    for key in ["fp32", "int8"]:
        if key not in all_metrics: continue
        m = all_metrics[key]
        print(f"  {m['tracker_name']:<18} "
              f"{m['HOTA']:>7.1f} {m['DetA']:>7.1f} "
              f"{m['AssA']:>7.1f} {m['MOTA']:>7.1f} {m['IDF1']:>7.1f}")

    if "fp32" in all_metrics and "int8" in all_metrics:
        fp32m = all_metrics["fp32"]; int8m = all_metrics["int8"]
        print("─" * 65)
        for metric in ["HOTA", "DetA", "AssA", "MOTA", "IDF1"]:
            drop = int8m[metric] - fp32m[metric]
            print(f"  Drop ({metric}): {drop:+.2f} pp")

    out_path = config.RESULTS_DIR / "accuracy_results.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
