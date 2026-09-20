"""Safety layer for the Cosmos augmentation batch (camera-only YOLO; the consumer's modality).

1. find_agent_window(mp4): pick the 121-frame window with the MOST agents, so the
   augmentation target actually contains agents to obscure + gives a valid difficulty
   baseline (the naive first-121-frame trim is often empty).
2. hallucination_gate(day_mp4, aug_mp4): reject an augmented clip that GAINS detections
   vs the original — added unlabeled agents (Cosmos hallucination) make obstacle.offline
   labels invalid. See memory cosmos-aug-hallucination.

Both reuse one YOLO model. Run under the alpamayo venv (ultralytics + cv2).
"""
import numpy as np, cv2
from ultralytics import YOLO

AD = [0, 1, 2, 3, 5, 7, 9, 11]   # person,bicycle,car,motorcycle,bus,truck,traffic light,stop sign
WIN = 121
_model = None


def _yolo():
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model


def _ndet(frame, thr=0.3):
    r = _yolo().predict(frame, conf=0.05, classes=AD, verbose=False, device="cpu")
    c = r[0].boxes.conf.cpu().numpy()
    c = c[c > thr]
    return int(len(c)), (float(c.mean()) if len(c) else 0.0)


def _grab(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx)); ok, f = cap.read()
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ok else None


def find_agent_window(mp4, win=WIN, samples=12):
    """Return (start_frame, agent_count, fps) for the win-frame window with most agents."""
    cap = cv2.VideoCapture(mp4)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if n <= 0:
        cap.release(); return 0, 0, fps
    counts = {}
    for i in [int(x) for x in np.linspace(0, max(0, n - 1), samples)]:
        f = _grab(cap, i)
        if f is not None:
            counts[i] = _ndet(f)[0]
    cap.release()
    if not counts:
        return 0, 0, fps
    best = max(counts, key=counts.get)
    start = max(0, min(max(0, n - win), best - win // 2))
    return start, counts[best], fps


def _frames_at(mp4, idxs):
    cap = cv2.VideoCapture(mp4); out = []
    for i in idxs:
        f = _grab(cap, i)
        if f is not None:
            out.append(f)
    cap.release(); return out


def hallucination_gate(day_mp4, aug_mp4, idxs=(30, 60, 90), tol=0.34):
    """Compare detections in the (trimmed) original vs the augmented clip on matched
    frames. PASS if the augmentation did not ADD agents (aug_ndet - day_ndet <= tol).
    tol allows ~1/3 frame of slack (one spurious det in one of 3 frames)."""
    dn = np.mean([_ndet(f)[0] for f in _frames_at(day_mp4, idxs)] or [0.0])
    an = np.mean([_ndet(f)[0] for f in _frames_at(aug_mp4, idxs)] or [0.0])
    dc = np.mean([_ndet(f)[1] for f in _frames_at(day_mp4, idxs)] or [0.0])
    ac = np.mean([_ndet(f)[1] for f in _frames_at(aug_mp4, idxs)] or [0.0])
    return {"day_ndet": round(float(dn), 2), "aug_ndet": round(float(an), 2),
            "added": round(float(an - dn), 2), "day_conf": round(float(dc), 3),
            "aug_conf": round(float(ac), 3), "harder": bool(ac < dc or an < dn),
            "passed": bool((an - dn) <= tol)}
