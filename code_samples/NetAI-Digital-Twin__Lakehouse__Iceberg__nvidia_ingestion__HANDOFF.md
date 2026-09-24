# Cosmos Augmentation Pipeline — Handoff Document

**Date**: April 2026
**For**: Colleague taking over Cosmos integration
**Status**: Code complete, mock-tested, NOT validated against live Cosmos API

---

## 1. What This Is

The Cosmos augmentation pipeline generates **synthetic driving scene variations** (fog, rain, night, snow, etc.) from real AV clips using Nvidia's Cosmos world foundation models. It sits downstream of the medallion lakehouse pipeline (Bronze/Silver/Gold) and consumes clips from the **Gold tier** — the hardest, most interesting clips identified by edge-case scoring.

**Why this matters**: The Gold tier identifies difficult driving scenarios (dawn/dusk, adverse weather, high ego dynamics). Cosmos generates synthetic variations of those clips to expand the training distribution — e.g., taking a daytime urban clip and generating foggy, rainy, and nighttime versions.

### Pipeline Flow

```
Gold Iceberg tables  ──►  Extract eligible clips
                              │
                              ▼
                     Cosmos API (NIM or API Catalog)
                     Generate per-clip variations
                     (foggy, rainy, night, snowy, golden_hour, overcast)
                              │
                              ▼
                     Upload MP4s to MinIO/S3
                     Write metadata to nvidia_cosmos Iceberg namespace
                     (generated_scenes + generation_lineage tables)
```

### Separation of Concerns

The Cosmos pipeline is **fully isolated** from the medallion pipeline:

- Reads from `nvidia_gold` namespace (read-only)
- Writes to its own `nvidia_cosmos` namespace
- Has its own config (`cosmos_augmentation/config.py`), independent of `nvidia_ingestion/config.py`
- Can run independently — doesn't need Bronze/Silver/Gold to be re-run

---

## 2. Code Architecture

All code lives in `cosmos_augmentation/` (6 files):

| File | Purpose |
|------|---------|
| `config.py` | Configuration: backend selection, variation prompts, API endpoints, Spark session |
| `cosmos_runner.py` | CLI entry point with 4 commands: `health`, `extract-only`, `generate`, `ingest-only` |
| `extract.py` | Reads Gold `sensor_fusion_clip` table, produces `ClipRecord` objects |
| `generate.py` | `CosmosClient` — unified HTTP client for NIM and API Catalog backends |
| `ingest_results.py` | Uploads MP4s to MinIO, writes `generated_scenes` + `generation_lineage` Iceberg tables |
| `mock_cosmos_server.py` | Standalone mock HTTP server mimicking `/v1/infer` and `/v1/health/ready` |

### Two Backends

The pipeline supports two ways to call Cosmos:

#### Backend 1: NIM (Self-Hosted Container)

- Docker image: `nvcr.io/nim/nvidia/cosmos-transfer2-5-2b:latest`
- Runs on-prem, requires NVIDIA GPU + `nvidia-container-toolkit`
- Endpoint: `http://cosmos:8000/v1/infer`
- Auth: NGC API key in environment
- **Status**: Docker compose config exists (commented out in `docker-compose.yml` lines 199-221) but **was never successfully run** — the DGX Spark we initially tried didn't support the NIM container image (see Blocker section below)

#### Backend 2: API Catalog (build.nvidia.com)

- Cloud-hosted, no local GPU required
- Endpoint: `https://integrate.api.nvidia.com/v1/cosmos/nvidia/<model>`
- Auth: `nvapi-` key from build.nvidia.com
- **Status**: Code is implemented but **never tested against the live API**

Both backends use the same request format (`POST` with JSON body, response contains `b64_video` base64-encoded MP4).

### Cosmos Models Supported

| Model | Slug | Use Case |
|-------|------|----------|
| Cosmos Transfer 1-7B | `transfer` | Video-to-video style transfer (needs input video) |
| Cosmos Transfer 2.5-2B | `transfer2.5` | Newer transfer model (needs input video) |
| Cosmos Predict 1-7B Text2World | `text2world` | Text prompt to video (no input video needed) |
| Cosmos Predict 1-7B Video2World | `video2world` | Input video + prompt to future prediction |

**Current default**: `text2world` — generates scenes from text prompts only. This was chosen because feeding real camera MP4s into `transfer` requires frame extraction and base64 encoding of video, which is implemented but untested. The `text2world` path is simpler for initial validation.

### Variation Prompts

Defined in `config.py` as `VARIATION_PROMPTS`:

```python
VARIATION_PROMPTS = {
    "foggy":       "dense fog, low visibility, diffused headlights, moisture in the air",
    "rainy":       "heavy rain, wet reflective road surface, rain drops on windshield",
    "night":       "nighttime, dark sky, street lights, headlight illumination, glare",
    "snowy":       "heavy snowfall, snow-covered road, white landscape, reduced visibility",
    "golden_hour": "golden hour sunset lighting, long warm shadows, sun low on horizon",
    "overcast":    "overcast sky, flat diffused lighting, grey clouds",
}
```

Each is appended to: `"First person view from a car driving on an urban road, photorealistic, high detail"`

### Iceberg Output Tables

Written to `nvidia_cosmos` namespace:

**`generated_scenes`** (partitioned by `variation`):
- `clip_id` — source clip identifier
- `variation` — e.g., "foggy", "rainy"
- `prompt` — full text prompt used
- `model` — Cosmos model slug
- `seed` — reproducibility seed
- `video_s3_uri` — S3 path to generated MP4
- `generation_time_s` — API call duration
- `source_split` — train/val/test from source clip
- `created_at` — timestamp

**`generation_lineage`**:
- `source_clip_id` — Gold clip that was augmented
- `source_table` — fully qualified Iceberg table name
- `variation` — variation type
- `generated_video_uri` — S3 path to output
- `model` — model used
- `created_at` — timestamp

---

## 3. The Blocker (and How to Move Past It)

### What Happened

Previous development was on a **DGX Spark** (Nvidia's ARM-based desktop AI workstation). The Cosmos NIM container (`nvcr.io/nim/nvidia/cosmos-transfer2-5-2b`) could not run on that platform — it's built for x86_64 with specific CUDA/driver requirements that the DGX Spark didn't meet. This was a hard wall: the container wouldn't start, and there was no workaround.

### Current Environment

The lakehouse has since moved to a **standard x86_64 server**:

- **CPU**: 24-core Intel Xeon
- **RAM**: 188 GB
- **GPUs**: RTX 6000 Turing (24GB VRAM) + A10 Ampere (23GB VRAM)
- **nvidia-container-toolkit**: Needs to be verified/installed
- **Docker**: Already running (spark-iceberg, polaris, minio, trino, superset containers active)

The NIM container blocker should be resolved on this hardware, but it has not been attempted yet.

### Three Paths Forward

1. **Try NIM on current hardware** (recommended first attempt):
   ```bash
   # 1. Get NGC API key from https://ngc.nvidia.com
   # 2. Login to NGC container registry
   docker login nvcr.io -u '$oauthtoken' -p <NGC_API_KEY>
   
   # 3. Uncomment the cosmos service in docker-compose.yml (lines 199-221)
   # 4. Set NGC_API_KEY in .env
   echo "NGC_API_KEY=<your-key>" >> .env
   
   # 5. Start the container
   docker compose up cosmos -d
   
   # 6. Wait for health check (can take 5+ minutes for model loading)
   docker compose logs -f cosmos
   
   # 7. Test
   python -m cosmos_augmentation.cosmos_runner health --backend nim
   ```
   The Cosmos Transfer 2.5-2B model needs ~12-16GB VRAM. Either the RTX 6000 or A10 should work.

2. **Use API Catalog** (no GPU needed, cloud-hosted):
   ```bash
   # 1. Get an API key from https://build.nvidia.com
   #    Navigate to a Cosmos model page and generate an nvapi- key
   
   # 2. Test
   python -m cosmos_augmentation.cosmos_runner health \
       --backend api-catalog --api-key nvapi-XXXX
   
   # 3. Generate
   python -m cosmos_augmentation.cosmos_runner generate \
       --backend api-catalog --api-key nvapi-XXXX \
       --max-clips 2 --variations foggy,rainy
   ```
   **Caveat**: API Catalog may have rate limits, latency, and costs. Good for validation, may not scale to thousands of clips.

3. **Use the mock server** (for pipeline testing without any API):
   ```bash
   # Terminal 1: start mock server
   python -m cosmos_augmentation.mock_cosmos_server --port 8000
   
   # Terminal 2: run pipeline against mock
   python -m cosmos_augmentation.cosmos_runner generate \
       --backend nim --endpoint http://localhost:8000 \
       --max-clips 2 --variations foggy,rainy
   ```
   The mock server generates labeled test MP4s (colored frames with variation name overlaid). It exercises the full pipeline path: extract → call API → upload to MinIO → write Iceberg metadata.

---

## 4. What Needs To Be Done

### Step 1: Validate Against Live API

This is the critical gap. The entire pipeline has been tested against the mock server but never against a real Cosmos endpoint. Things that could break:

- **Request format**: The API payload structure (`prompt`, `video`, `guidance_scale`, `seed`) matches the Cosmos documentation as of early April 2026, but NVIDIA may have changed it
- **Response format**: We expect `{"b64_video": "<base64>"}` — verify this is still correct
- **Video quality**: `text2world` generates from text prompts alone. The output may not be realistic enough for AV training data. `transfer` mode (video-to-video) would produce better results but requires feeding in camera MP4 frames
- **Latency**: A single generation call may take 30-120 seconds. Budget accordingly

**Validation procedure**:
```bash
# Start small: 1 clip, 1 variation
python -m cosmos_augmentation.cosmos_runner generate \
    --backend api-catalog --api-key nvapi-XXXX \
    --max-clips 1 --variations foggy --seed 42

# Check the output
# - Look at the generated MP4 in MinIO (localhost:9001, default creds minioadmin/minioadmin)
# - Check Iceberg metadata: 
#   spark.table("iceberg.nvidia_cosmos.generated_scenes").show()
```

### Step 2: Enable Video-to-Video Transfer

Currently, `generate_variations()` in `generate.py` passes `input_video_b64=None`, which means it always falls back to `text2world` (text-only generation). For higher-quality results:

1. Extract camera frames from the source clip's MP4 file
2. Encode as base64
3. Pass to the `transfer` or `video2world` model

The camera MP4 paths are available via the Gold `sensor_fusion_clip` table. The `extract.py` module would need to be extended to:
- Resolve the MP4 file path from the clip metadata
- Read + encode the video
- Pass it through to `generate_variations()`

This is the most impactful improvement after basic validation.

### Step 3: Scale Up

Once validated:
```bash
# Generate all 5 default variations for all Gold clips
python -m cosmos_augmentation.cosmos_runner generate \
    --backend api-catalog --api-key nvapi-XXXX \
    --variations foggy,rainy,night,snowy,golden_hour
```

Monitor MinIO storage consumption (generated MP4s accumulate) and Iceberg table growth.

### Step 4: Quality Assessment

No automated quality check exists yet. Suggested approach:
- Visual inspection of generated videos vs source clips
- FID/FVD scores if a reference distribution is available
- Domain expert review of whether synthetic variations are realistic enough for AV model training

---

## 5. Dataset Context (What You're Augmenting)

The source data is the **Nvidia PhysicalAI Autonomous Vehicles** dataset (HuggingFace, gated). Our subset:

- **340 chunks** (~11% of full dataset), ~33,767 clips, ~188 hours of driving
- **25 countries**, all seasons, all times of day
- **7 cameras** (120/70/30 FOV), **1 lidar** (360 roof-mounted), **19 radar sensors**
- **20-second clips** from the Hyperion 8/8.1 sensor platform

The Gold tier selects the hardest clips via composite scoring (time of day, season/geography, sensor coverage, ego dynamics). These are the clips that get fed to Cosmos for augmentation.

See `nvidia_ingestion/DATASET.md` for the full dataset breakdown including schemas, sensor coverage, and data relationships.

---

## 6. File Reference

### Cosmos Pipeline (`cosmos_augmentation/`)

| File | What to read it for |
|------|-------------------|
| `config.py` | Backend URLs, variation prompts, model selection, environment variables |
| `cosmos_runner.py` | CLI interface, pipeline orchestration (3-phase: extract → generate → ingest) |
| `extract.py` | How clips are pulled from Gold tables. Extend this for video-to-video input |
| `generate.py` | HTTP client, retry logic, model dispatch. Fix here if API format changes |
| `ingest_results.py` | S3 upload, Iceberg table schemas. Adjust partitioning/optimizations here |
| `mock_cosmos_server.py` | Run for local testing. Good reference for expected request/response format |

### Related Files

| File | Relevance |
|------|-----------|
| `docker-compose.yml` | Cosmos NIM service definition (commented out, lines 199-221) |
| `docker-compose.yml` | spark-iceberg container mounts `cosmos_augmentation/` at `/opt/spark/cosmos_augmentation` |
| `nvidia_ingestion/config.py` | `NvidiaConfig` — provides Gold namespace name, Spark settings |
| `nvidia_ingestion/DATASET.md` | Full dataset breakdown (sensors, schemas, coverage, scale) |

### Infrastructure

- **Spark container**: `docker exec -it spark-iceberg bash`, then run python commands
- **MinIO console**: `http://localhost:9001` (minioadmin/minioadmin) — check generated videos
- **Iceberg catalog**: Polaris at `http://localhost:8181`
- **NFS dataset root**: `/mnt/shared/netai-e2e/nvidia-physicalai-av-subset/` (from inside spark container, path depends on `docker-compose.override.yml`)

---

## 7. Quick Start

```bash
# 1. Get into the spark container
docker exec -it spark-iceberg bash

# 2. Test with mock server first (no API key needed)
#    In another terminal:
python -m cosmos_augmentation.mock_cosmos_server

#    Back in spark container:
python -m cosmos_augmentation.cosmos_runner generate \
    --backend nim --endpoint http://<host-ip>:8000 \
    --max-clips 2 --variations foggy,rainy

# 3. Verify output
python -c "
from cosmos_augmentation.config import CosmosPipelineConfig, build_spark_session
cfg = CosmosPipelineConfig()
spark = build_spark_session(cfg)
spark.table('iceberg.nvidia_cosmos.generated_scenes').show()
spark.table('iceberg.nvidia_cosmos.generation_lineage').show()
"

# 4. When ready for real API:
python -m cosmos_augmentation.cosmos_runner generate \
    --backend api-catalog --api-key nvapi-XXXX \
    --max-clips 1 --variations foggy --seed 42
```
