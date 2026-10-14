# Nvidia PhysicalAI Autonomous Vehicles 데이터셋 — 상세 분석

**출처**: [nvidia/PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) (HuggingFace, 게이트 접근)
**버전**: v25.10 (초기 릴리스), v26.03 기능 일부 미포함
**로컬 서브셋**: 전체 ~3,146개 청크 중 340개
**디스크 전체 크기**: ~13 TB (압축 해제 후)

> **2026-04-29 시점**, 이 데이터셋은 Iceberg에서 canonical KAIST 스키마
> (`kaist_schema_v2.dbml`)로 노출됩니다 — `iceberg.nvidia_bronze`의 16개 테이블:
> Session, Clip, Episode, Frame, Calibration, Camera, Lidar, Radar, CanBus,
> HDMap, Session_EgoMotion, Category, DynamicObject, Occupancy, Motion,
> EgoMotion. 전체 매핑 명세와 빌드 벤치마크는 `MEDALLION_PROGRESS.md §11`을
> 참조. 아래 섹션들은 NFS 상의 원본 데이터 레이아웃을 설명하며, canonical
> 테이블이 그로부터 재구성됩니다.

---

## 1. 데이터셋 개요

Nvidia PhysicalAI AV 데이터셋은 공개된 멀티센서 자율주행 데이터셋 중 최대 규모입니다:

| 항목 | 전체 데이터셋 | 우리 서브셋 |
|------|-------------|------------|
| 총 클립 수 | 310,895 | ~33,767 (340개 청크) |
| 클립 길이 | 각 ~140초 (egomotion 기준) | 각 ~140초 |
| 총 주행 시간 | 1,727시간 | ~188시간 |
| 국가 수 | 25 | 25 (서브셋에 모두 포함) |
| 센서 플랫폼 | Hyperion 8 / 8.1 | Hyperion 8 / 8.1 |
| 총 크기 | ~133 TB | ~13 TB |

각 "클립"은 모든 센서에서 동시에 기록된 다중 센서 녹화 데이터이며, UUID로 식별됩니다 (예: `25cd4769-5dcf-4b53-a351-bf2c5deb6124`). 이전 문서에는 20초 클립이라고 적혀 있었으나, 2026-05-04에 32,651개 클립의 egomotion 데이터를 직접 조사한 결과 클립별 상대 시간 범위는 **약 140초**입니다 (egomotion `max_ts` p50 = 139,257,000 µs, p99 = 140,192,000 µs). 카메라 mp4가 동일한 범위를 커버하는지 또는 그중 20초만 커버하는지는 아직 미확인 — 140초 수치는 egomotion 타임스탬프 범위로 취급하고 비디오 실제 길이와의 비교 후 사용하시기 바랍니다.

각 "청크"는 센서 유형별로 하나의 zip 아카이브로 패키징된 ~100개 클립의 배치입니다.

---

## 2. 센서 모달리티

### 2.1 카메라 (7개 센서)

**형식**: MP4 비디오 파일 (H.264, 1080p, 30fps) + 타임스탬프 parquet sidecar

**센서 목록**:
| 카메라 | 화각 | 커버리지 (청크) |
|--------|------|----------------|
| `camera_front_wide_120fov` | 전방 120° | 340/340 |
| `camera_front_tele_30fov` | 전방 망원 30° | 340/340 |
| `camera_cross_left_120fov` | 좌측 120° | 340/340 |
| `camera_cross_right_120fov` | 우측 120° | 340/340 |
| `camera_rear_left_70fov` | 후방 좌측 70° | 340/340 |
| `camera_rear_right_70fov` | 후방 우측 70° | **307/340** (불완전) |
| `camera_rear_tele_30fov` | 후방 망원 30° | **175/340** (불완전) |

**청크별 파일 구조**:
```
camera/<sensor>/<sensor>.chunk_XXXX/
  ├── <clip_uuid>.<sensor>.mp4              # 30fps, 20초 비디오
  ├── <clip_uuid>.<sensor>.timestamps.parquet    # 프레임별 타임스탬프
  └── <clip_uuid>.<sensor>.blurred_boxes.parquet # 개인정보 블러 영역
```

**타임스탬프 parquet 스키마** (클립별):
- 비디오 프레임과 동기화된 프레임 수준 타임스탬프
- 카메라 프레임을 LiDAR/레이더/egomotion과 정렬하는 데 사용

**블러 박스 parquet**: 개인정보 보호를 위해 블러 처리된 영역(얼굴, 번호판)의 바운딩 박스.

### 2.2 LiDAR (1개 센서)

**센서**: `lidar_top_360fov` — 루프 장착 360도 LiDAR
**형식**: Draco 압축 포인트 클라우드 blob이 포함된 Parquet 파일
**커버리지**: 340/340개 청크 추출 완료 (2026년 4월 데이터 손실 사고 이후 복구)

**청크별 파일 구조**:
```
lidar/lidar_top_360fov/lidar_top_360fov.chunk_XXXX/
  └── <clip_uuid>.lidar_top_360fov.parquet    # 파일당 ~216 MB
```

**스키마** (클립별 parquet):
- Draco 압축 3D 포인트 클라우드 데이터 포함
- 클립당 ~200 LiDAR 스핀 (클립 길이 약 ~140초, egomotion 범위 기준 — §1 참조)
- XYZ + intensity를 얻으려면 DracoPy 라이브러리로 디코딩 필요
- 포인트 클라우드 밀도로 인해 각 parquet 파일이 ~216 MB

**디코딩 예시**:
```python
import DracoPy
# parquet에서 Draco blob 컬럼 읽기
# 각 blob을 디코딩하여 포인트 클라우드 배열 획득
compressed = row["lidar_data"]  # 바이너리 blob
mesh = DracoPy.decode(compressed)
points = mesh.points  # Nx3 float 배열
```

### 2.3 레이더 (19개 센서)

**형식**: 3D 레이더 탐지가 포함된 Parquet 파일
**커버리지**: 2026년 4월 복구 이후 19개 센서 모두 채워짐. 아래 표의 청크 수는 "추출 완료 / 설계 청크 수"입니다 — 서브셋 전체는 340개 청크지만 각 레이더 센서는 그중 일부 청크에만 데이터를 제공합니다 (`srr_0`/`mrr_2`는 보통 ~120개, `srr_3`/`imaging_lrr_1` 등은 ~134개). 설계 대비 부족분 = HuggingFace에서 404로 응답한 49개 업스트림 삭제 zip.

**센서 목록** (위치별 그룹):

| 위치 | 센서 | 유형 | 청크 수 (추출 / 설계) |
|------|------|------|---------------------|
| 전방 중앙 | `radar_front_center_srr_0` | 단거리 | 120 / 121 |
| 전방 중앙 | `radar_front_center_mrr_2` | 중거리 | 130 / 134 |
| 전방 중앙 | `radar_front_center_imaging_lrr_1` | 장거리 이미징 | 130 / 134 |
| 전방 좌측 | `radar_corner_front_left_srr_0` | 단거리 | 120 / 120 |
| 전방 좌측 | `radar_corner_front_left_srr_3` | 단거리 (대안) | 130 / 134 |
| 전방 우측 | `radar_corner_front_right_srr_0` | 단거리 | 120 / 121 |
| 전방 우측 | `radar_corner_front_right_srr_3` | 단거리 (대안) | 130 / 134 |
| 후방 좌측 | `radar_corner_rear_left_srr_0` | 단거리 | 120 / 121 |
| 후방 좌측 | `radar_corner_rear_left_srr_3` | 단거리 (대안) | 130 / 134 |
| 후방 우측 | `radar_corner_rear_right_srr_0` | 단거리 | 120 / 121 |
| 후방 우측 | `radar_corner_rear_right_srr_3` | 단거리 (대안) | 130 / 134 |
| 후방 좌측 | `radar_rear_left_mrr_2` | 중거리 | 130 / 134 |
| 후방 좌측 | `radar_rear_left_srr_0` | 단거리 | 120 / 121 |
| 후방 우측 | `radar_rear_right_mrr_2` | 중거리 | 130 / 134 |
| 후방 우측 | `radar_rear_right_srr_0` | 단거리 | 120 / 121 |
| 좌측면 | `radar_side_left_srr_0` | 단거리 | 120 / 121 |
| 좌측면 | `radar_side_left_srr_3` | 단거리 (대안) | 32 / 36 |
| 우측면 | `radar_side_right_srr_0` | 단거리 | 120 / 121 |
| 우측면 | `radar_side_right_srr_3` | 단거리 (대안) | 32 / 36 |

**레이더 설정**: 클립에는 `"low"` 또는 `"high"` 레이더 설정이 있으며, `sensor_presence.parquet`의 `radar_config` 필드에 표시됩니다. low 설정은 ~9개 센서, high 설정은 최대 19개 센서를 사용합니다.

**청크별 파일 구조**:
```
radar/<sensor>/<sensor>.chunk_XXXX/
  └── <clip_uuid>.<sensor>.parquet
```

**스키마**: 탐지별 속도 추정, 신호 강도, 신뢰도가 포함된 3D 레이더 탐지 데이터.

### 2.4 Egomotion (라벨)

**형식**: ego 차량 궤적이 포함된 Parquet 파일
**커버리지**: 340개 청크 (NFS silly-rename 파일에서 복구)
**크기**: 총 ~14 GB (청크 zip당 ~40 MB)

**청크별 파일 구조**:
```
labels/egomotion/egomotion.chunk_XXXX/
  └── <clip_uuid>.egomotion.parquet
```

**스키마** (클립별 parquet):
- `timestamp` — 로컬 프레임 시간
- `x`, `y`, `z` — 위치 (미터 단위, t=0 기준 원점 상대)
- `qw`, `qx`, `qy`, `qz` — 방향 쿼터니언
- 중력 기준 자세 (yaw, pitch, roll) 추정
- 원점은 timestamp 0에서의 ego 차량 위치, yaw = 0

**활용 사례**: ego 차량 동역학 분석, 궤적 예측, LiDAR-ego 융합.

### 2.5 캘리브레이션

**형식**: Parquet 파일 (청크당 1개)
**커버리지**: 각 340개 파일, 완전

| 유형 | 파일 | 설명 |
|------|------|------|
| 카메라 내부 파라미터 | `calibration/camera_intrinsics/` | 카메라별 초점 거리, 주점, 왜곡 계수 |
| 센서 외부 파라미터 | `calibration/sensor_extrinsics/` | 센서와 ego 차량 프레임 간 6DoF 변환 |
| 차량 치수 | `calibration/vehicle_dimensions/` | 차량 길이, 너비, 높이, 축거 |

---

## 3. 메타데이터

### 3.1 `clip_index.parquet` (310,895행)

최상위 클립 레지스트리. 클립당 1행.

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `clip_id` | string | UUID, 기본 키 (예: `25cd4769-5dcf-4b53-...`) |
| `chunk` | int64 | 청크 번호 (전체 데이터셋 0-3145, 서브셋은 특정 청크 사용) |
| `split` | string | `train`, `val`, 또는 `test` |
| `clip_is_valid` | bool | 데이터 유효성 플래그 |

### 3.2 `metadata/data_collection.parquet` (310,895행)

클립별 수집 컨텍스트. clip_id로 clip_index와 조인.

| 컬럼 | 타입 | 설명 | 예시 값 |
|------|------|------|---------|
| `clip_id` | string | UUID, 조인 키 | |
| `country` | string | 수집 국가 | "United States", "Germany", "Finland", ... (25개국) |
| `month` | int64 | 월 (1-12) | 5, 8, 12 |
| `hour_of_day` | int64 | 시 (0-23) | 17, 14, 3 |
| `platform_class` | string | 센서 플랫폼 | "hyperion_8", "hyperion_8.1" |

### 3.3 `metadata/sensor_presence.parquet` (310,895행)

클립별 센서 가용성 매트릭스. 센서별 boolean 플래그.

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `clip_id` | string | UUID, 조인 키 |
| `camera_cross_left_120fov` | bool | 카메라 존재 여부 (모든 클립에 True) |
| `camera_cross_right_120fov` | bool | (7개 카메라 모두 동일) |
| ... | ... | |
| `lidar_top_360fov` | bool | LiDAR 존재 여부 |
| `radar_corner_front_left_srr_0` | bool | 레이더 센서 존재 여부 |
| ... | ... | (총 19개 레이더 센서) |
| `radar_config` | string | `"low"` 또는 `"high"` 레이더 구성 |

### 3.4 `selected_chunks.csv` (340행)

우리 서브셋에 포함된 각 청크의 메타데이터.

| 컬럼 | 타입 | 설명 | 예시 값 |
|------|------|------|---------|
| `chunk` | int | 청크 번호 | 3, 10, 15, ... |
| `country` | string | 주요 국가 | "United States", "Germany" |
| `season` | string | 수집 시 계절 | "winter", "spring", "summer", "fall" |
| `hour_bin` | string | 시간대 구간 | "morning", "afternoon", "evening", "night" |
| `platform` | string | 센서 플랫폼 | "hyperion_8", "hyperion_8.1" |
| `n_clips` | int | 해당 청크의 클립 수 | ~100 |
| `split` | string | 데이터셋 분할 | "train", "val", "test" |

**340개 청크 서브셋의 분포:**
- **국가**: 25개 (전체 포함)
- **계절**: winter, spring, summer, fall
- **시간대**: morning, afternoon, evening, night
- **플랫폼**: hyperion_8, hyperion_8.1
- **총 클립 수**: 33,767

---

## 4. 미포함 데이터 (v26.03 추가분)

다음 데이터는 2026년 3월 업데이트에서 추가되었으며, **아직 다운로드되지 않았습니다**:

| 데이터 | 경로 | 추정 크기 | 설명 |
|--------|------|----------|------|
| 장애물 라벨 | `labels/obstacle.offline/` | ~50-100 GB | 기계 생성 3D 장애물 탐지 (ground truth 아님). 프레임별 바운딩 박스 + 클래스 라벨. |
| 오프라인 egomotion | `labels/egomotion.offline/` | ~14 GB | 신호 처리 최적화된 ego motion (온라인 버전보다 부드러움) |
| 추론 라벨 | `reasoning/ood_reasoning.parquet` | ~10 MB | 1,740개 클립에 대한 사람이 검증한 OOD 추론 라벨 (Chain of Causation 어노테이션) |
| 오프라인 캘리브레이션 | `calibration/camera_intrinsics.offline/`, `lidar_intrinsics.offline/`, `sensor_extrinsics.offline/` | ~5 GB | NuRec 재구성을 위한 오프라인 최적화 캘리브레이션 |
| Feature presence | `metadata/feature_presence.parquet` | ~12 MB | 클립별 오프라인 feature 플래그가 포함된 업데이트된 센서 존재 정보 |

**다운로드 명령어** (HuggingFace 토큰 필요):
```bash
hf download nvidia/PhysicalAI-Autonomous-Vehicles --repo-type dataset \
  --include "labels/obstacle.offline/*" \
  --include "labels/egomotion.offline/*" \
  --include "reasoning/*" \
  --local-dir /path/to/dataset
```

---

## 5. 데이터 관계

```
clip_index.parquet ──(clip_id)──┬── data_collection.parquet
                                ├── sensor_presence.parquet
                                ├── labels/egomotion/<chunk>/<clip_id>.egomotion.parquet
                                ├── lidar/<sensor>/<chunk>/<clip_id>.lidar_top_360fov.parquet
                                ├── radar/<sensor>/<chunk>/<clip_id>.<sensor>.parquet
                                └── camera/<sensor>/<chunk>/<clip_id>.<sensor>.mp4
                                                           <clip_id>.<sensor>.timestamps.parquet
                                                           <clip_id>.<sensor>.blurred_boxes.parquet

selected_chunks.csv ──(chunk)── clip_index.parquet

calibration/ ──(청크별, clip_id 없음)── 전역 센서 파라미터
```

**조인 키**: `clip_id` (UUID 문자열)가 모든 센서 데이터 및 메타데이터에 걸친 범용 조인 키입니다.

**청크**: ~100개 클립의 그룹. 각 청크는 센서별 하나의 zip 아카이브에 대응합니다. 청크 번호는 `clip_index.chunk`와 `selected_chunks.csv.chunk`를 연결합니다.

---

## 6. 주요 제한 사항

1. **3D 바운딩 박스 ground truth 없음** — 장애물 라벨(v26.03)은 기계 생성이며, 사람이 어노테이션한 것이 아닙니다. ground truth 대비 표준 mAP 평가를 수행할 수 없습니다.

2. **LiDAR는 Draco 압축** — 원시 XYZ 포인트 클라우드는 SQL로 직접 조회할 수 없습니다. 사용 전 DracoPy 디코딩이 필요합니다.

3. **카메라 데이터는 프레임이 아닌 비디오** — MP4 파일은 LiDAR 타임스탬프와 정렬하기 위해 프레임 추출이 필요합니다. 타임스탬프 parquet이 프레임-타임스탬프 매핑을 제공합니다.

4. **레이더 커버리지 불균일** — `radar_config="low"`인 클립은 ~9개 센서, `"high"`는 최대 19개 센서를 사용합니다. 모든 클립에 모든 레이더 센서가 존재하지는 않습니다.

5. **카메라 2개 센서 불완전** — `camera_rear_right_70fov` (307/340 청크 존재, 122개에 0이 아닌 parquet)와 `camera_rear_tele_30fov` (175/340 청크 존재, 2개에 0이 아닌 parquet)에 공백이 있습니다. 2026년 4월 재다운로드는 radar+lidar만 대상이었으므로 이 두 센서는 향후 별도의 카메라 복구 패스가 필요합니다.

7. **49개 레이더 zip 업스트림 삭제** — 2026년 4월 재다운로드에서 `transfer_manifest.json`에 등재된 49개 zip이 HuggingFace에서 `404`로 응답했습니다 (업스트림에서 삭제됨). 모두 레이더 (대부분 `chunk_1057`, `chunk_3109`). 예상 레이더 parquet의 ~0.6%에 영향. 클립 단위 영향은 아직 분석되지 않았으며 Silver `missing_sensors` 품질 검사로 처리됩니다.

6. **서브셋은 340/3146 청크** — 전체 데이터셋의 약 11%. 서브셋은 25개국 전체, 전 계절, 전 시간대를 포함하도록 선별되었습니다.

---

## 7. 규모 참조

Canonical Bronze 행 수 (복구 후 + canonical 재구성, 2026-04-29):

| Canonical 테이블 | 행 수 | 비고 |
|----------------|------|------|
| Session | 3,116 | clip_index의 chunk별 1개 (전체 데이터셋) |
| Clip | 310,895 | 클립당 1개 |
| Calibration | 458,873 | (clip_id, sensor_name)당 1개 |
| Camera | 109,171,395 | 7개 카메라 프레임 메타데이터 |
| Lidar | 6,164,244 | 340개 청크의 spin 메타데이터 |
| Radar | 11,730,962,796 | 19개 레이더 센서 탐지 |
| EgoMotion | 101,745,981 | 타임스탬프별 ego 상태 |
| Frame | 257,290,851 | distinct (clip_id, sensor_timestamp) |
| Episode, CanBus, HDMap, Session_EgoMotion, Category, DynamicObject, Occupancy, Motion | 0 each | 빈 테이블 (Nvidia 소스 없음) |
| **합계** | **12,206,108,151** | 16개 canonical 테이블 전체 |

### (이하는 원본 add_files() 등록 시점의 참조 정보)

Iceberg 등록 기준 행 수 (이전 벤치마크 실행):

| 테이블 | 행 수 | 비고 |
|--------|-------|------|
| clip_index | 310,895 | 클립당 1행 (서브셋에 없는 청크의 클립도 포함) |
| egomotion | ~1억 100만 | 20초 클립당 ~300 샘플, ~15 Hz |
| lidar | ~33K | 클립당 1개 parquet 파일 (각각 ~200 프레임 포함) |
| radar (센서별) | ~1억-21.7억 | 센서에 따라 다름; front_center_imaging_lrr_1이 21.7억 행 |
| radar (19개 센서 전체) | ~113억 | 모든 레이더 센서 합산 |

**쿼리 성능** (벤치마크 기준):
- Iceberg `count(*)` (21.7억 행): **129ms** (manifest 메타데이터를 통한 O(1))
- Egomotion count: **439ms**
- Silver 뷰 생성 (31개 뷰): **14.1초**
- Gold 뷰 생성 (3개 뷰): **7.6초**
