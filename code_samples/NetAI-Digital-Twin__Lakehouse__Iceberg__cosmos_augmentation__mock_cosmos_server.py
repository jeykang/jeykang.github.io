#!/usr/bin/env python3
"""
Mock Cosmos NIM server for end-to-end pipeline testing.

Implements the same /v1/infer and /v1/health/ready endpoints as the real
Cosmos NIM container, but generates simple labeled test videos instead of
running a world foundation model.  Useful for:

  - Testing on platforms where the real NIM image is unavailable (e.g. ARM)
  - CI/CD pipeline validation
  - Development without GPU resources

Usage:
    python -m cosmos_augmentation.mock_cosmos_server          # port 8000
    python -m cosmos_augmentation.mock_cosmos_server --port 9000

The generated videos are short MP4s with coloured frames indicating the
requested variation (foggy=grey, rainy=blue, night=dark, etc.) and the
prompt text overlaid.
"""

import argparse
import base64
import io
import json
import random
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Variation → background colour (BGR for OpenCV)
VARIATION_COLORS = {
    "fog":          (180, 180, 180),
    "foggy":        (180, 180, 180),
    "rain":         (140, 100, 60),
    "rainy":        (140, 100, 60),
    "night":        (30, 20, 10),
    "snow":         (230, 220, 210),
    "snowy":        (230, 220, 210),
    "golden_hour":  (60, 140, 220),
    "golden hour":  (60, 140, 220),
    "overcast":     (160, 160, 150),
}

DEFAULT_COLOR = (100, 120, 80)


def _guess_variation(prompt: str) -> str:
    """Extract the variation name from the prompt text."""
    prompt_lower = prompt.lower()
    for key in VARIATION_COLORS:
        if key in prompt_lower:
            return key
    return "unknown"


def _generate_video(prompt: str, seed: int, width: int = 704, height: int = 480,
                    fps: int = 24, num_frames: int = 48) -> bytes:
    """Generate a short labelled MP4 test video.

    Each frame has a background colour matching the variation, with the
    prompt text and frame number overlaid.
    """
    variation = _guess_variation(prompt)
    bg_bgr = VARIATION_COLORS.get(variation, DEFAULT_COLOR)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    buf_path = "/tmp/_mock_cosmos_out.mp4"
    writer = cv2.VideoWriter(buf_path, fourcc, fps, (width, height))

    rng = random.Random(seed)

    for i in range(num_frames):
        # Create background with slight per-frame variation for realism
        noise = rng.randint(-10, 10)
        color = tuple(max(0, min(255, c + noise)) for c in bg_bgr)
        frame = np.full((height, width, 3), color, dtype=np.uint8)

        # Add some random "scene" rectangles
        for _ in range(5):
            x1, y1 = rng.randint(0, width - 50), rng.randint(0, height - 50)
            x2, y2 = x1 + rng.randint(20, 150), y1 + rng.randint(20, 100)
            c = tuple(max(0, min(255, cc + rng.randint(-40, 40))) for cc in bg_bgr)
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, -1)

        # Road-like perspective lines
        cv2.line(frame, (width // 2, height // 3), (0, height), (80, 80, 80), 2)
        cv2.line(frame, (width // 2, height // 3), (width, height), (80, 80, 80), 2)
        cv2.line(frame, (width // 2, height // 3), (width // 2, height), (200, 200, 200), 1)

        # Overlay text via Pillow for clean rendering
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # Title
        draw.text((20, 20), f"MOCK COSMOS — {variation.upper()}", fill=(255, 255, 255))
        draw.text((20, 45), f"seed={seed}  frame={i + 1}/{num_frames}", fill=(200, 200, 200))

        # Prompt (truncated)
        prompt_short = prompt[:80] + ("..." if len(prompt) > 80 else "")
        draw.text((20, height - 40), prompt_short, fill=(180, 180, 180))

        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        writer.write(frame)

    writer.release()

    with open(buf_path, "rb") as f:
        return f.read()


class CosmosHandler(BaseHTTPRequestHandler):
    """HTTP handler mimicking Cosmos NIM API."""

    def do_GET(self):
        if self.path == "/v1/health/ready":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ready"}')
        elif self.path == "/v1/health/live":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status": "live"}')
        elif self.path == "/v1/metadata":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            meta = {"model": "mock-cosmos-predict1-7b-text2world", "version": "mock-1.0"}
            self.wfile.write(json.dumps(meta).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/v1/infer":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))

        prompt = body.get("prompt", "no prompt provided")
        seed = body.get("seed", random.randint(1, 99999))

        print(f"[INFER] prompt={prompt[:60]}...  seed={seed}")
        t0 = time.time()

        video_bytes = _generate_video(prompt, seed)
        b64_video = base64.b64encode(video_bytes).decode("ascii")

        elapsed = time.time() - t0
        print(f"[INFER] Generated {len(video_bytes)} bytes in {elapsed:.2f}s")

        response = {
            "b64_video": b64_video,
            "seed": seed,
            "upsampled_prompt": prompt,
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        # Quieter logging
        print(f"[HTTP] {args[0]}")


def main():
    parser = argparse.ArgumentParser(description="Mock Cosmos NIM server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), CosmosHandler)
    print(f"Mock Cosmos NIM listening on {args.host}:{args.port}")
    print(f"  Health:  GET  http://localhost:{args.port}/v1/health/ready")
    print(f"  Infer:   POST http://localhost:{args.port}/v1/infer")
    server.serve_forever()


if __name__ == "__main__":
    main()
