import json
import os
import time

# 데이터 경로
SCALE_FACTOR = 7  # 여기를 바꿔가며 실험하세요!
RAW_DATA_PATH = "/user_data/nuscenes-mini/v1.0-mini/v1.0-mini"

def load_json(filename):
    with open(os.path.join(RAW_DATA_PATH, filename), 'r') as f:
        return json.load(f)

print(">>> [Baseline: Pure Python] 실험 시작")
start_time = time.time()

# -----------------------------------------------------------------------------
# 1. Initialization
# -----------------------------------------------------------------------------
print("Step 1: Loading JSON files into memory...")

# 파일 로드 (instance.json 추가)
data_samples = load_json('sample.json')
data_sample_data = load_json('sample_data.json')
data_annotations = load_json('sample_annotation.json')
data_categories = load_json('category.json')
data_instances = load_json('instance.json')  # [추가됨]
data_sensors = load_json('sensor.json')
data_calibrated_sensors = load_json('calibrated_sensor.json')

# =========================================================
# [실험 변수] 데이터 스케일 팩터 (1, 5, 10, 100 변경)
# =========================================================
print(f">>> [Experiment] Scaling Data by {SCALE_FACTOR}x ...")

# 리스트 단순 복제 (메모리 사용량도 정직하게 늘어남)
data_samples = data_samples * SCALE_FACTOR
data_sample_data = data_sample_data * SCALE_FACTOR
data_annotations = data_annotations * SCALE_FACTOR

print("Step 2: Building Index (Token mapping)...")

# List를 Dictionary로 변환하여 인덱스 생성
sample_map = {s['token']: s for s in data_samples}
# category.json의 
category_map = {c['token']: c['name'] for c in data_categories}

# [추가됨] Instance Token -> Category Token 맵핑
# Annotation은 category_token을 직접 안 가지고 instance_token만 가짐
instance_to_category_map = {i['token']: i['category_token'] for i in data_instances}

# Sensor & Channel 맵핑: calibrated_sensor_token -> channel(RADAR_FRONT)
sensor_channel_map = {s['token']: s['channel'] for s in data_sensors}
calib_to_channel_map = {}
for cs in data_calibrated_sensors:
    s_token = cs['sensor_token']
    if s_token in sensor_channel_map:
        calib_to_channel_map[cs['token']] = sensor_channel_map[s_token]

# Annotation Grouping
ann_by_sample = {} 
for ann in data_annotations:
    s_token = ann['sample_token']
    if s_token not in ann_by_sample:
        ann_by_sample[s_token] = []
    ann_by_sample[s_token].append(ann)

init_end_time = time.time()
print(f"Initialization Time (Load + Index): {init_end_time - start_time:.4f} sec")

# -----------------------------------------------------------------------------
# 2. Data Lookup & Filtering
# -----------------------------------------------------------------------------
print("Step 3: Filtering Data (CAM_FRONT & Pedestrian)...")

dataset_pairs = []

for sd in data_sample_data:
    # Channel 확인 (sensor 장비 + 방향 확인)
    calib_token = sd['calibrated_sensor_token']
    channel = calib_to_channel_map.get(calib_token)
    
    if channel != 'CAM_FRONT':
        continue
    
    s_token = sd['sample_token']
    
    # if s_token in ann_by_sample:
    for ann in ann_by_sample[s_token]:
        # [수정됨] Annotation -> Instance -> Category 연결
        inst_token = ann['instance_token']
        cat_token = instance_to_category_map.get(inst_token)
        cat_name = category_map.get(cat_token)
        
        if cat_name == 'human.pedestrian.adult':
            dataset_pairs.append({
                'img_path': sd['filename'],
                'bbox_translation': ann['translation'],
                'bbox_size': ann['size']
            })
    # else:
    #     print(f"Warning: No annotations for sample_token {s_token}")

end_time = time.time()

# -----------------------------------------------------------------------------
# 3. Report
# -----------------------------------------------------------------------------
print("-" * 30)
print(f"Total Rows Found: {len(dataset_pairs)}")
print(f"Total Time Elapsed: {end_time - start_time:.4f} sec")
print("-" * 30)