"""Container-free validation of the cheap augmentation backend (transforms.py).

Decodes a real front-camera frame per clip (PyAV), applies the night/rain/fog
transforms, measures the perceptual degradation (brightness/contrast/edge-density
drop — proxies for what lowers a perception model's confidence), and saves a
before/after composite so the output can be eyeballed. No GPU / container / API.

Validates the GENERATION side of the loop (transform produces real perceptual
degradation, plumbing works). The full "does it lower BEVFusion confidence"
re-score needs the GPU container (currently k8s/containerd, deferred) — but we
already know real low-light clips drop BEVFusion confidence ~10% (perceptual axis
analysis), so degradation of this magnitude should register.
"""
import glob, os, sys
import numpy as np, av
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transforms

ROOT = "netai-e2e/nvidia-physicalai-av-subset"
OUT = "/tmp/aug_samples"; os.makedirs(OUT, exist_ok=True)
K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
CONDS = ["night", "rain", "fog"]


def grab_frame(mp4, idx=30, w=512):
    c = av.open(mp4)
    img = None
    for i, fr in enumerate(c.decode(video=0)):
        if i >= idx:
            img = fr.to_ndarray(format="rgb24"); break
    c.close()
    if img is None:
        return None
    h0, w0 = img.shape[:2]; h = int(h0 * w / w0)
    return np.asarray(Image.fromarray(img).resize((w, h)))


def stats(img):
    g = img.astype(np.float32).mean(-1)
    gy, gx = np.gradient(g)
    return dict(bright=float(g.mean()), contrast=float(g.std()),
                edge=float(np.hypot(gx, gy).mean()))


def main():
    mp4s = sorted(glob.glob(f"{ROOT}/camera/camera_front_wide_120fov/*/*.mp4"))[:K * 3]
    done = 0
    print(f"{'clip':10s} {'cond':6s} {'Δbright%':>8s} {'Δcontr%':>8s} {'Δedge%':>7s}", flush=True)
    for mp4 in mp4s:
        if done >= K:
            break
        cid = os.path.basename(mp4).split(".")[0]
        try:
            orig = grab_frame(mp4)
            if orig is None:
                continue
            s0 = stats(orig); panels = [orig]
            for cond in CONDS:
                t = transforms.apply(orig, cond); panels.append(t); s1 = stats(t)
                db = 100 * (s1["bright"] - s0["bright"]) / max(1, s0["bright"])
                dc = 100 * (s1["contrast"] - s0["contrast"]) / max(1, s0["contrast"])
                de = 100 * (s1["edge"] - s0["edge"]) / max(1e-3, s0["edge"])
                print(f"{cid[:10]} {cond:6s} {db:8.1f} {dc:8.1f} {de:7.1f}", flush=True)
            # composite: orig | night | rain | fog
            comp = np.concatenate(panels, axis=1)
            Image.fromarray(comp).save(f"{OUT}/{cid[:8]}_orig_night_rain_fog.png")
            done += 1
        except Exception as e:
            print(f"{cid[:10]} WARN {str(e)[:60]}", flush=True)
    print(f"\nsaved {done} before/after composites to {OUT}", flush=True)
    print("(Δ negative = degraded vs original — lower brightness/contrast/edges = harder to perceive)", flush=True)
    print(">>> CHEAP VALIDATE DONE", flush=True)


if __name__ == "__main__":
    main()
