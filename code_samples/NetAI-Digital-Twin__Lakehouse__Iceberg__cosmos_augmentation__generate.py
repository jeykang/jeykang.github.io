"""
Cosmos API client and variation generator.

Supports two backends:
  - NIM: Self-hosted container at a configurable endpoint (POST /v1/infer).
  - API Catalog: NVIDIA build.nvidia.com hosted API with nvapi- key auth.

Both backends use the same /v1/infer request format and return base64 MP4.
No special SDK required — just HTTP + base64.
"""

import base64
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import BASE_DRIVING_PROMPT, VARIATION_PROMPTS, CosmosConfig
from .extract import ClipRecord


@dataclass
class GeneratedVideo:
    """Result of a single Cosmos generation call."""

    clip_id: str
    variation: str
    prompt: str
    model: str
    seed: Optional[int]
    video_bytes: bytes
    generation_time_s: float
    source_split: str = ""


class CosmosClient:
    """Unified client for Cosmos NIM and API Catalog backends."""

    def __init__(self, config: CosmosConfig):
        self.config = config
        self.session = self._build_session(config.max_retries)

    @staticmethod
    def _build_session(max_retries: int) -> requests.Session:
        session = requests.Session()
        retries = Retry(
            total=max_retries,
            backoff_factor=2,
            status_forcelist=[502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _headers(self) -> dict:
        """Build request headers based on backend."""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.backend == "api-catalog" and self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def health_check(self) -> bool:
        """Return True if the backend is reachable.

        For NIM, calls GET /v1/health/ready.
        For API Catalog, does a lightweight OPTIONS/GET probe.
        """
        if self.config.backend == "api-catalog":
            # API Catalog has no health endpoint; verify the key is set
            if not self.config.api_key:
                print("  [WARN] COSMOS_API_KEY not set — cannot use api-catalog backend")
                return False
            return True
        try:
            resp = self.session.get(self.config.health_url, timeout=10)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False

    # ── generation endpoints ─────────────────────────────────────────────

    def _infer(self, payload: dict) -> bytes:
        """POST to the infer URL and return decoded MP4 bytes."""
        resp = self.session.post(
            self.config.infer_url,
            json=payload,
            headers=self._headers(),
            timeout=self.config.timeout_seconds,
        )
        resp.raise_for_status()
        b64_video = resp.json()["b64_video"]
        return base64.b64decode(b64_video)

    def transfer(
        self,
        input_video_b64: str,
        prompt: str,
        controls: Optional[Dict] = None,
        seed: Optional[int] = None,
    ) -> bytes:
        """Video-to-video transfer with optional control signals."""
        payload: dict = {
            "prompt": prompt,
            "video": input_video_b64,
            "resolution": self.config.resolution,
            "guidance_scale": self.config.guidance_scale,
        }
        if seed is not None:
            payload["seed"] = seed
        if controls:
            payload.update(controls)
        return self._infer(payload)

    def text2world(self, prompt: str, seed: Optional[int] = None) -> bytes:
        """Generate a video from a text prompt only."""
        payload: dict = {
            "prompt": prompt,
            "guidance_scale": self.config.guidance_scale,
        }
        if seed is not None:
            payload["seed"] = seed
        return self._infer(payload)

    def video2world(
        self,
        input_video_b64: str,
        prompt: str,
        seed: Optional[int] = None,
    ) -> bytes:
        """Predict future frames from an input video + prompt."""
        payload: dict = {
            "prompt": prompt,
            "video": input_video_b64,
            "guidance_scale": self.config.guidance_scale,
        }
        if seed is not None:
            payload["seed"] = seed
        return self._infer(payload)


def generate_variations(
    client: CosmosClient,
    clip: ClipRecord,
    input_video_b64: Optional[str],
    variations: List[str],
    seed: Optional[int] = None,
) -> List[GeneratedVideo]:
    """Generate visual variations for a single clip.

    For each variation name (e.g. "foggy"), looks up the prompt template
    in VARIATION_PROMPTS, calls the appropriate Cosmos model, and returns
    a list of GeneratedVideo results.
    """
    results: List[GeneratedVideo] = []
    model_name = client.config.model

    for variation in variations:
        suffix = VARIATION_PROMPTS.get(variation, variation)
        prompt = f"{BASE_DRIVING_PROMPT}, {suffix}"

        print(f"    Generating '{variation}' for clip {clip.clip_id[:12]}... ", end="", flush=True)
        t0 = time.perf_counter()

        try:
            if model_name in ("transfer", "transfer2.5") and input_video_b64:
                video_bytes = client.transfer(input_video_b64, prompt, seed=seed)
            elif model_name == "video2world" and input_video_b64:
                video_bytes = client.video2world(input_video_b64, prompt, seed=seed)
            else:
                video_bytes = client.text2world(prompt, seed=seed)
        except requests.RequestException as exc:
            elapsed = time.perf_counter() - t0
            print(f"FAILED ({elapsed:.1f}s) — {exc}")
            continue

        elapsed = time.perf_counter() - t0
        print(f"OK ({elapsed:.1f}s, {len(video_bytes) / 1024 / 1024:.1f} MB)")

        results.append(
            GeneratedVideo(
                clip_id=clip.clip_id,
                variation=variation,
                prompt=prompt,
                model=model_name,
                seed=seed,
                video_bytes=video_bytes,
                generation_time_s=elapsed,
                source_split=clip.split,
            )
        )

    return results
