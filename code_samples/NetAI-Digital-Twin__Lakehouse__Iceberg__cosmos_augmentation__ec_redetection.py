"""E-C: re-detection agreement on kept augmented clips. For each KEPT clip, YOLO the
original vs augmented agent-window on matched frames, match detections by image-plane
IoU>=0.3, report the fraction of original detections that survive in the augmented clip
+ the mean centroid shift of survivors. Quantifies label alignment (was by-construction).
Run under the alpamayo venv. Reads gate_report.json + /tmp/batch_gate/*.mp4."""
import json, sys
import numpy as np, cv2
from ultralytics import YOLO

AD = [0, 1, 2, 3, 5, 7, 9, 11]; IDX = [30, 60, 90]; SZ = (960, 540)
GATE = sys.argv[1] if len(sys.argv) > 1 else "cosmos_augmentation/gate_report.json"
DIR = sys.argv[2] if len(sys.argv) > 2 else "/tmp/batch_gate"
m = YOLO("yolov8n.pt")


def dets(mp4):
    cap = cv2.VideoCapture(mp4); out = {}
    for i in IDX:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
        if not ok:
            continue
        f = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), SZ)   # common grid -> comparable coords
        r = m.predict(f, conf=0.05, classes=AD, verbose=False, device="cpu")
        b = r[0].boxes; keep = b.conf.cpu().numpy() > 0.25
        out[i] = b.xyxy.cpu().numpy()[keep]
    cap.release(); return out


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def ctr(x): return np.array([(x[0]+x[2])/2, (x[1]+x[3])/2])


kept = [g for g in json.load(open(GATE)) if g.get("kept")]
print(f"kept clips: {len(kept)}")
print(f"{'clip':10s} {'cond':5s} {'orig_det':>8s} {'matched%':>9s} {'shift_px':>9s}")
TM = TN = 0; shifts = []
for g in kept:
    s, c = g["clip"], g["cond"]
    D = dets(f"{DIR}/{s}_day.mp4"); A = dets(f"{DIR}/{s}_{c}_aug.mp4")
    md = nd = 0; sh = []
    for i in IDX:
        if i not in D or i not in A:
            continue
        for db in D[i]:
            nd += 1
            if len(A[i]):
                j = max(range(len(A[i])), key=lambda k: iou(db, A[i][k]))
                if iou(db, A[i][j]) >= 0.3:
                    md += 1; sh.append(float(np.linalg.norm(ctr(db) - ctr(A[i][j]))))
    TM += md; TN += nd; shifts += sh
    print(f"{s:10s} {c:5s} {nd:8d} {100*md/nd if nd else 0:8.1f}% {np.mean(sh) if sh else 0:8.1f}")
print(f"\nAGGREGATE: {TM}/{TN} = {100*TM/max(1,TN):.1f}% of original detections re-detected in the "
      f"augmented clip (IoU>=0.3)")
print(f"mean centroid shift of survivors: {np.mean(shifts) if shifts else float('nan'):.1f} px "
      f"(on a {SZ[0]}x{SZ[1]} grid)")
