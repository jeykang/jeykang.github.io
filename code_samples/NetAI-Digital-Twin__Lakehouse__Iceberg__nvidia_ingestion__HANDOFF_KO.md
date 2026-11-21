# Cosmos 증강 파이프라인 — 인수인계 문서

**날짜**: 2026년 4월
**대상**: Cosmos 통합 작업을 인수받는 동료
**상태**: 코드 완성, Mock 테스트 완료, 실제 Cosmos API 검증 미완료

---

## 1. 개요

Cosmos 증강 파이프라인은 Nvidia의 Cosmos world foundation model을 사용하여 실제 자율주행 클립에서 **합성 주행 장면 변형**(안개, 비, 야간, 눈 등)을 생성합니다. Medallion lakehouse 파이프라인(Bronze/Silver/Gold) 하위에 위치하며, **Gold 티어** — edge-case 스코어링을 통해 선별된 가장 어렵고 흥미로운 클립 — 을 입력으로 사용합니다.

**이 작업의 의미**: Gold 티어는 어려운 주행 시나리오(새벽/황혼, 악천후, 높은 ego dynamics)를 식별합니다. Cosmos는 이러한 클립의 합성 변형을 생성하여 학습 데이터 분포를 확장합니다. 예를 들어, 낮 시간대 도심 클립을 안개, 비, 야간 버전으로 변환할 수 있습니다.

### 파이프라인 흐름

```
Gold Iceberg 테이블  ──►  대상 클립 추출
                              │
                              ▼
                     Cosmos API (NIM 또는 API Catalog)
                     클립별 변형 생성
                     (foggy, rainy, night, snowy, golden_hour, overcast)
                              │
                              ▼
                     MP4를 MinIO/S3에 업로드
                     nvidia_cosmos Iceberg 네임스페이스에 메타데이터 기록
                     (generated_scenes + generation_lineage 테이블)
```

### 관심사 분리

Cosmos 파이프라인은 medallion 파이프라인과 **완전히 격리**되어 있습니다:

- `nvidia_gold` 네임스페이스에서 읽기 전용으로 데이터 참조
- 자체 `nvidia_cosmos` 네임스페이스에 기록
- 독립적인 설정 파일 (`cosmos_augmentation/config.py`), `nvidia_ingestion/config.py`와 무관
- 독립 실행 가능 — Bronze/Silver/Gold 재실행 불필요

---

## 2. 코드 구조

모든 코드는 `cosmos_augmentation/` 디렉토리에 있습니다 (6개 파일):

| 파일 | 역할 |
|------|------|
| `config.py` | 설정: backend 선택, 변형 프롬프트, API endpoint, Spark 세션 |
| `cosmos_runner.py` | CLI 진입점, 4개 명령어: `health`, `extract-only`, `generate`, `ingest-only` |
| `extract.py` | Gold `sensor_fusion_clip` 테이블에서 클립 읽기, `ClipRecord` 객체 생성 |
| `generate.py` | `CosmosClient` — NIM 및 API Catalog backend 통합 HTTP 클라이언트 |
| `ingest_results.py` | MP4를 MinIO에 업로드, `generated_scenes` + `generation_lineage` Iceberg 테이블 기록 |
| `mock_cosmos_server.py` | `/v1/infer` 및 `/v1/health/ready` endpoint를 모방하는 독립 Mock HTTP 서버 |

### 두 가지 Backend

파이프라인은 Cosmos를 호출하는 두 가지 방식을 지원합니다:

#### Backend 1: NIM (자체 호스팅 컨테이너)

- Docker 이미지: `nvcr.io/nim/nvidia/cosmos-transfer2-5-2b:latest`
- 온프레미스 실행, NVIDIA GPU + `nvidia-container-toolkit` 필요
- Endpoint: `http://cosmos:8000/v1/infer`
- 인증: 환경 변수에 NGC API key 설정
- **현재 상태**: Docker Compose 설정 존재 (`docker-compose.yml` 199-221행, 주석 처리됨), 그러나 **실제 실행된 적 없음** — 초기 개발에 사용한 DGX Spark에서 NIM 컨테이너 이미지를 지원하지 않았음 (아래 차단 요인 섹션 참조)

#### Backend 2: API Catalog (build.nvidia.com)

- 클라우드 호스팅, 로컬 GPU 불필요
- Endpoint: `https://integrate.api.nvidia.com/v1/cosmos/nvidia/<model>`
- 인증: build.nvidia.com에서 발급받은 `nvapi-` 키
- **현재 상태**: 코드 구현 완료, **실제 API 테스트 미완료**

두 backend 모두 동일한 요청 형식을 사용합니다 (JSON body로 `POST`, 응답에 `b64_video` base64 인코딩 MP4 포함).

### 지원하는 Cosmos 모델

| 모델 | Slug | 용도 |
|------|------|------|
| Cosmos Transfer 1-7B | `transfer` | 비디오-투-비디오 스타일 변환 (입력 비디오 필요) |
| Cosmos Transfer 2.5-2B | `transfer2.5` | 최신 transfer 모델 (입력 비디오 필요) |
| Cosmos Predict 1-7B Text2World | `text2world` | 텍스트 프롬프트만으로 비디오 생성 (입력 비디오 불필요) |
| Cosmos Predict 1-7B Video2World | `video2world` | 입력 비디오 + 프롬프트로 미래 프레임 예측 |

**현재 기본값**: `text2world` — 텍스트 프롬프트만으로 장면을 생성합니다. 실제 카메라 MP4를 `transfer`에 전달하려면 프레임 추출 및 base64 인코딩이 필요한데, 구현은 되어 있으나 테스트되지 않았습니다. `text2world` 경로가 초기 검증에 더 간단합니다.

### 변형 프롬프트

`config.py`에 `VARIATION_PROMPTS`로 정의:

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

각 프롬프트는 다음 기본 프롬프트에 추가됩니다: `"First person view from a car driving on an urban road, photorealistic, high detail"`

### Iceberg 출력 테이블

`nvidia_cosmos` 네임스페이스에 기록:

**`generated_scenes`** (`variation`으로 파티셔닝):
- `clip_id` — 원본 클립 식별자
- `variation` — 예: "foggy", "rainy"
- `prompt` — 사용된 전체 텍스트 프롬프트
- `model` — Cosmos 모델 slug
- `seed` — 재현성 시드
- `video_s3_uri` — 생성된 MP4의 S3 경로
- `generation_time_s` — API 호출 소요 시간
- `source_split` — 원본 클립의 train/val/test 구분
- `created_at` — 타임스탬프

**`generation_lineage`**:
- `source_clip_id` — 증강된 Gold 클립
- `source_table` — 정규화된 Iceberg 테이블 이름
- `variation` — 변형 유형
- `generated_video_uri` — 출력 S3 경로
- `model` — 사용된 모델
- `created_at` — 타임스탬프

---

## 3. 차단 요인 (및 해결 방안)

### 이전 상황

이전 개발은 **DGX Spark** (Nvidia의 ARM 기반 데스크탑 AI 워크스테이션)에서 진행했습니다. Cosmos NIM 컨테이너(`nvcr.io/nim/nvidia/cosmos-transfer2-5-2b`)가 해당 플랫폼에서 실행되지 않았습니다 — x86_64 아키텍처 및 특정 CUDA/드라이버 요구사항을 충족하지 못했기 때문입니다. 컨테이너가 시작되지 않았고, 우회 방법도 없었습니다.

### 현재 환경

Lakehouse는 이후 **표준 x86_64 서버**로 이전되었습니다:

- **CPU**: 24코어 Intel Xeon
- **RAM**: 188 GB
- **GPU**: RTX 6000 Turing (24GB VRAM) + A10 Ampere (23GB VRAM)
- **nvidia-container-toolkit**: 설치 여부 확인 필요
- **Docker**: 이미 실행 중 (spark-iceberg, polaris, minio, trino, superset 컨테이너 가동 중)

현재 하드웨어에서는 NIM 컨테이너 차단 요인이 해결되어야 하지만, 아직 시도하지 않았습니다.

### 세 가지 진행 방안

1. **현재 하드웨어에서 NIM 시도** (권장 첫 번째 시도):
   ```bash
   # 1. https://ngc.nvidia.com에서 NGC API key 발급
   # 2. NGC 컨테이너 레지스트리 로그인
   docker login nvcr.io -u '$oauthtoken' -p <NGC_API_KEY>
   
   # 3. docker-compose.yml에서 cosmos 서비스 주석 해제 (199-221행)
   # 4. .env에 NGC_API_KEY 설정
   echo "NGC_API_KEY=<your-key>" >> .env
   
   # 5. 컨테이너 시작
   docker compose up cosmos -d
   
   # 6. Health check 대기 (모델 로딩에 5분 이상 소요 가능)
   docker compose logs -f cosmos
   
   # 7. 테스트
   python -m cosmos_augmentation.cosmos_runner health --backend nim
   ```
   Cosmos Transfer 2.5-2B 모델은 ~12-16GB VRAM이 필요합니다. RTX 6000 또는 A10 모두 사용 가능합니다.

2. **API Catalog 사용** (GPU 불필요, 클라우드 호스팅):
   ```bash
   # 1. https://build.nvidia.com에서 API key 발급
   #    Cosmos 모델 페이지에서 nvapi- 키 생성
   
   # 2. 테스트
   python -m cosmos_augmentation.cosmos_runner health \
       --backend api-catalog --api-key nvapi-XXXX
   
   # 3. 생성
   python -m cosmos_augmentation.cosmos_runner generate \
       --backend api-catalog --api-key nvapi-XXXX \
       --max-clips 2 --variations foggy,rainy
   ```
   **주의사항**: API Catalog에는 속도 제한, 지연 시간, 비용이 있을 수 있습니다. 검증용으로는 적합하나, 수천 개 클립으로 확장하기에는 부적합할 수 있습니다.

3. **Mock 서버 사용** (API 없이 파이프라인 테스트):
   ```bash
   # 터미널 1: mock 서버 시작
   python -m cosmos_augmentation.mock_cosmos_server --port 8000
   
   # 터미널 2: mock 대상으로 파이프라인 실행
   python -m cosmos_augmentation.cosmos_runner generate \
       --backend nim --endpoint http://localhost:8000 \
       --max-clips 2 --variations foggy,rainy
   ```
   Mock 서버는 레이블이 표시된 테스트 MP4 (변형 이름이 오버레이된 색상 프레임)를 생성합니다. 전체 파이프라인 경로를 검증합니다: 추출 -> API 호출 -> MinIO 업로드 -> Iceberg 메타데이터 기록.

---

## 4. 해야 할 작업

### 1단계: 실제 API 검증

가장 핵심적인 공백입니다. 전체 파이프라인이 mock 서버에 대해서만 테스트되었고, 실제 Cosmos endpoint에 대해서는 한 번도 테스트되지 않았습니다. 문제가 발생할 수 있는 부분:

- **요청 형식**: API payload 구조 (`prompt`, `video`, `guidance_scale`, `seed`)가 2026년 4월 초 기준 Cosmos 문서와 일치하지만, NVIDIA가 변경했을 수 있음
- **응답 형식**: `{"b64_video": "<base64>"}` 형태를 기대함 — 여전히 올바른지 확인 필요
- **비디오 품질**: `text2world`는 텍스트 프롬프트만으로 생성함. 출력이 자율주행 학습 데이터로 사용하기에 충분히 사실적이지 않을 수 있음. `transfer` 모드 (비디오-투-비디오)가 더 나은 결과를 생성하지만 카메라 MP4 프레임 입력이 필요
- **지연 시간**: 단일 생성 호출에 30-120초 소요 가능. 이에 맞춰 계획 수립 필요

**검증 절차**:
```bash
# 소규모 시작: 1개 클립, 1개 변형
python -m cosmos_augmentation.cosmos_runner generate \
    --backend api-catalog --api-key nvapi-XXXX \
    --max-clips 1 --variations foggy --seed 42

# 출력 확인
# - MinIO에서 생성된 MP4 확인 (localhost:9001, 기본 인증정보 minioadmin/minioadmin)
# - Iceberg 메타데이터 확인:
#   spark.table("iceberg.nvidia_cosmos.generated_scenes").show()
```

### 2단계: 비디오-투-비디오 Transfer 활성화

현재 `generate.py`의 `generate_variations()`는 `input_video_b64=None`을 전달하여 항상 `text2world` (텍스트 전용 생성)로 fallback됩니다. 더 높은 품질의 결과를 위해:

1. 원본 클립의 MP4 파일에서 카메라 프레임 추출
2. base64로 인코딩
3. `transfer` 또는 `video2world` 모델에 전달

카메라 MP4 경로는 Gold `sensor_fusion_clip` 테이블을 통해 확인할 수 있습니다. `extract.py` 모듈을 다음과 같이 확장해야 합니다:
- 클립 메타데이터에서 MP4 파일 경로 확인
- 비디오 읽기 + 인코딩
- `generate_variations()`에 전달

기본 검증 이후 가장 영향력 있는 개선 사항입니다.

### 3단계: 규모 확장

검증 완료 후:
```bash
# 모든 Gold 클립에 대해 기본 5개 변형 전체 생성
python -m cosmos_augmentation.cosmos_runner generate \
    --backend api-catalog --api-key nvapi-XXXX \
    --variations foggy,rainy,night,snowy,golden_hour
```

MinIO 스토리지 사용량(생성된 MP4 누적)과 Iceberg 테이블 증가를 모니터링하세요.

### 4단계: 품질 평가

아직 자동화된 품질 검사가 없습니다. 제안하는 접근 방식:
- 생성된 비디오와 원본 클립의 시각적 비교
- 참조 분포가 있는 경우 FID/FVD 스코어 측정
- 합성 변형이 자율주행 모델 학습에 충분히 사실적인지 도메인 전문가 검토

---

## 5. 데이터셋 개요 (증강 대상)

원본 데이터는 **Nvidia PhysicalAI Autonomous Vehicles** 데이터셋 (HuggingFace, 게이트 접근)입니다. 우리의 서브셋:

- **340개 청크** (전체 데이터셋의 ~11%), ~33,767개 클립, ~188시간 주행 데이터
- **25개국**, 전 계절, 전 시간대
- **카메라 7대** (120/70/30 FOV), **LiDAR 1대** (360도 루프 장착), **레이더 19대**
- **20초 클립** — Hyperion 8/8.1 센서 플랫폼

Gold 티어는 복합 스코어링(시간대, 계절/지역, 센서 커버리지, ego dynamics)을 통해 가장 어려운 클립을 선별합니다. 이 클립들이 Cosmos 증강의 입력이 됩니다.

전체 데이터셋 세부 사항(스키마, 센서 커버리지, 데이터 관계)은 `nvidia_ingestion/DATASET.md`를 참조하세요.

---

## 6. 파일 참조

### Cosmos 파이프라인 (`cosmos_augmentation/`)

| 파일 | 참조 목적 |
|------|----------|
| `config.py` | Backend URL, 변형 프롬프트, 모델 선택, 환경 변수 |
| `cosmos_runner.py` | CLI 인터페이스, 파이프라인 오케스트레이션 (3단계: 추출 -> 생성 -> 적재) |
| `extract.py` | Gold 테이블에서 클립을 가져오는 방법. 비디오-투-비디오 입력을 위해 이 파일을 확장 |
| `generate.py` | HTTP 클라이언트, 재시도 로직, 모델 디스패치. API 형식 변경 시 여기서 수정 |
| `ingest_results.py` | S3 업로드, Iceberg 테이블 스키마. 파티셔닝/최적화 조정은 여기서 |
| `mock_cosmos_server.py` | 로컬 테스트용 실행. 예상 요청/응답 형식의 좋은 참조 |

### 관련 파일

| 파일 | 관련성 |
|------|--------|
| `docker-compose.yml` | Cosmos NIM 서비스 정의 (주석 처리됨, 199-221행) |
| `docker-compose.yml` | spark-iceberg 컨테이너가 `cosmos_augmentation/`을 `/opt/spark/cosmos_augmentation`에 마운트 |
| `nvidia_ingestion/config.py` | `NvidiaConfig` — Gold 네임스페이스 이름, Spark 설정 제공 |
| `nvidia_ingestion/DATASET.md` | 전체 데이터셋 세부 정보 (센서, 스키마, 커버리지, 규모) |

### 인프라

- **Spark 컨테이너**: `docker exec -it spark-iceberg bash`로 접속 후 python 명령 실행
- **MinIO 콘솔**: `http://localhost:9001` (minioadmin/minioadmin) — 생성된 비디오 확인
- **Iceberg 카탈로그**: Polaris, `http://localhost:8181`
- **NFS 데이터셋 루트**: `/mnt/shared/netai-e2e/nvidia-physicalai-av-subset/` (spark 컨테이너 내부 기준, 경로는 `docker-compose.override.yml`에 따라 다름)

---

## 7. 빠른 시작

```bash
# 1. Spark 컨테이너 접속
docker exec -it spark-iceberg bash

# 2. Mock 서버로 먼저 테스트 (API key 불필요)
#    다른 터미널에서:
python -m cosmos_augmentation.mock_cosmos_server

#    Spark 컨테이너에서:
python -m cosmos_augmentation.cosmos_runner generate \
    --backend nim --endpoint http://<host-ip>:8000 \
    --max-clips 2 --variations foggy,rainy

# 3. 출력 확인
python -c "
from cosmos_augmentation.config import CosmosPipelineConfig, build_spark_session
cfg = CosmosPipelineConfig()
spark = build_spark_session(cfg)
spark.table('iceberg.nvidia_cosmos.generated_scenes').show()
spark.table('iceberg.nvidia_cosmos.generation_lineage').show()
"

# 4. 실제 API 사용 준비가 되면:
python -m cosmos_augmentation.cosmos_runner generate \
    --backend api-catalog --api-key nvapi-XXXX \
    --max-clips 1 --variations foggy --seed 42
```
