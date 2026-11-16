"""Cheap classical degradation transforms — a stand-in 'generation backend' for
validating the difficulty-augmentation loop without Cosmos (no API, no cluster).

Same contract a Cosmos-Transfer backend would have: RGB frame(s) in + a target
condition -> transformed RGB frame(s) out, geometry preserved (so obstacle.offline
boxes + ego trajectory stay valid -> labels transfer for free). Swap this module
for the Cosmos backend once cluster generation is justified; the loop is identical.

Conditions: 'night' (low-light + sensor noise), 'rain' (low-contrast + streaks +
blur), 'fog' (haze + contrast loss). These intentionally degrade perception the way
the real hard conditions do — they are NOT photorealistic (that's Cosmos's value);
they exist to prove the pipeline + measure whether degradation registers in the
difficulty scorer's perceptual axis.
"""
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


def _gamma(img, g):
    return (255.0 * (np.clip(img, 0, 255) / 255.0) ** g).astype(np.uint8)


def night(img, strength=1.0):
    """Low-light: darken (gamma), cool blue shift, desaturate, add sensor noise."""
    out = _gamma(img, 1.0 + 1.6 * strength).astype(np.float32)
    out *= (1.0 - 0.45 * strength)                      # overall dimming
    out[..., 2] *= (1.0 + 0.12 * strength)              # slight blue (BGR: idx2=R? use RGB: idx2=B)
    gray = out.mean(-1, keepdims=True)
    out = out * (1 - 0.4 * strength) + gray * (0.4 * strength)   # desaturate
    noise = np.random.normal(0, 8 * strength, out.shape)         # low-light sensor noise
    return np.clip(out + noise, 0, 255).astype(np.uint8)


def rain(img, strength=1.0):
    """Wet/low-vis: dim, reduce contrast, vertical streaks, slight blur, desaturate."""
    out = (np.clip(img, 0, 255).astype(np.float32)) * (1 - 0.25 * strength)
    mean = out.mean()
    out = (out - mean) * (1 - 0.35 * strength) + mean            # contrast down
    h, w = out.shape[:2]
    rng = np.random.default_rng(0)
    n = int(0.0008 * strength * h * w)
    ys = rng.integers(0, h - 12, n); xs = rng.integers(0, w, n)
    for y, x in zip(ys, xs):
        out[y:y + rng.integers(6, 12), x] += 60 * strength       # rain streaks
    gray = out.mean(-1, keepdims=True)
    out = out * (1 - 0.3 * strength) + gray * (0.3 * strength)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if cv2 is not None and strength > 0:
        out = cv2.GaussianBlur(out, (3, 3), 0.6 * strength)
    return out


def fog(img, strength=1.0):
    """Haze: blend toward gray, contrast loss, stronger toward top (distance)."""
    out = np.clip(img, 0, 255).astype(np.float32)
    h = out.shape[0]
    haze = np.linspace(0.6, 0.3, h).reshape(h, 1, 1) * strength   # more haze up top
    out = out * (1 - haze) + 200 * haze
    mean = out.mean()
    out = (out - mean) * (1 - 0.4 * strength) + mean
    return np.clip(out, 0, 255).astype(np.uint8)


TRANSFORMS = {"night": night, "rain": rain, "fog": fog}


def apply(frames, condition, strength=1.0):
    """Apply a condition transform to a frame or list of frames (RGB uint8)."""
    fn = TRANSFORMS[condition]
    if isinstance(frames, list):
        return [fn(f, strength) for f in frames]
    return fn(frames, strength)
