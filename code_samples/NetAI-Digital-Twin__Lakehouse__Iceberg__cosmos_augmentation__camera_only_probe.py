"""Camera-only re-score probe — the test that matters for the camera-only endgame.

BEVFusion (lidar-fused) was robust to camera degradation because clean lidar masked
it. The downstream consumer's final product is CAMERA-ONLY, so the right difficulty
yardstick is a camera-only detector. Here: run YOLO (camera-only, no lidar) on a
front-camera frame per clip, original vs night/rain/fog-degraded, and measure the
confidence / detection drop. If it drops, perceptual augmentation registers for the
camera-only target -> the augmentation premise is rescued. No GPU/container needed.
"""
import glob, os, sys
import numpy as np, av
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transforms
from ultralytics import YOLO

ROOT = "netai-e2e/nvidia-physicalai-av-subset"
K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
CONDS = ["night", "rain", "fog"]
AD = [0, 1, 2, 3, 5, 7, 9, 11]   # person,bicycle,car,motorcycle,bus,truck,traffic light,stop sign
WEIGHTS = "yolov8n.pt" if os.path.exists("yolov8n.pt") else "yolo11n.pt"


def grab_frame(mp4, idx=30, w=640):
    c = av.open(mp4); img = None
    for i, fr in enumerate(c.decode(video=0)):
        if i >= idx:
            img = fr.to_ndarray(format="rgb24"); break
    c.close()
    if img is None:
        return None
    h0, w0 = img.shape[:2]
    return np.asarray(Image.fromarray(img).resize((w, int(h0 * w / w0))))


def score(model, img):
    """Camera-only perception strength: mean confidence of AD detections + count."""
    r = model.predict(img, conf=0.05, classes=AD, verbose=False, device="cpu")
    c = r[0].boxes.conf.cpu().numpy()
    c = c[c > 0.25]
    return (float(c.mean()) if len(c) else 0.0), int(len(c))


def main():
    print(f"camera-only detector = {WEIGHTS}", flush=True)
    model = YOLO(WEIGHTS)
    mp4s = sorted(glob.glob(f"{ROOT}/camera/camera_front_wide_120fov/*/*.mp4"))[:K * 2]
    agg = {c: ([], []) for c in CONDS}; done = 0
    print(f"{'clip':10s} {'orig(conf/n)':>13s} " + " ".join(f"{c+'(Δconf/Δn)':>16s}" for c in CONDS), flush=True)
    for mp4 in mp4s:
        if done >= K:
            break
        cid = os.path.basename(mp4).split(".")[0]
        try:
            orig = grab_frame(mp4)
            if orig is None:
                continue
            c0, n0 = score(model, orig)
            cells = [f"{c0:.3f}/{n0}"]
            for cond in CONDS:
                c1, n1 = score(model, transforms.apply(orig, cond))
                agg[cond][0].append(c1 - c0); agg[cond][1].append(n1 - n0)
                cells.append(f"{c1-c0:+.3f}/{n1-n0:+d}")
            print(f"{cid[:10]} {cells[0]:>13s} " + " ".join(f"{x:>16s}" for x in cells[1:]), flush=True)
            done += 1
        except Exception as e:
            print(f"{cid[:10]} WARN {str(e)[:60]}", flush=True)
    print(f"\nmean over {done} clips:", flush=True)
    for cond in CONDS:
        dc, dn = agg[cond]
        print(f"  {cond:6s}  Δconf={np.mean(dc):+.3f}  Δndet={np.mean(dn):+.2f}", flush=True)
    print("(negative Δ = camera degradation lowered camera-only perception -> aug registers)", flush=True)
    print(">>> CAMERA-ONLY PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
