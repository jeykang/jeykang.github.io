from minio import Minio
from minio.error import S3Error
from minio.versioningconfig import VersioningConfig

import json, os
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv(dotenv_path='.my_env')

def upload_file(client, i):
    # The file to upload, change this path if needed
    source_file = "uwb_data.json"

    # The destination bucket and filename on the MinIO server
    bucket_name = "lake-uwb"
    destination_file = "uwb-rtls-feed16"
    # destination_file = "vt.json" #테스트용 이름
    
    # client.set_bucket_versioning(bucket_name, VersioningConfig(status="Enabled"))

    adjusted_time  = datetime.now() + timedelta(hours=9)
    timestamp = adjusted_time.strftime("%Y%m%dT%H%M%S")  # 현재 시간 포맷
    object_name = f"{destination_file}_{timestamp}.json"  # 객체 이름에 타임스탬프 추가
    # object_name = destination_file
    # Upload the file, renaming it in the process,
    client.fput_object(
        bucket_name, object_name, source_file
    )
    
def get_metadata(client):
    bucket_name = "lake-uwb"
    # destination_file = "uwb-rtls.json"
    object_name = "vt.json"
        
    stat = client.stat_object(bucket_name, object_name)
    print(f"Object Name: {stat.object_name}")
    print(f"Last Modified: {stat.last_modified}")
    print(f"ETag: {stat.etag}")
    print(f"Size: {stat.size} bytes")
    print(f"Content Type: {stat.content_type}")
            
    # client.fget_object(
    #     bucket_name, object_name, './downloaded_logs.json'
    # )
    # with open('./downloaded_logs.json', 'r') as f:
    #     print(f.read())

def get_file_info(minio):
    bucket_name = "lake-uwb"
    object_name = "uwb-rtls-feed16_20241229T003757.json"
    # 오브젝트 가져오기
    response = minio.get_object(bucket_name, object_name)

    # JSON 데이터 읽기 및 파싱
    json_data = json.load(response)

    # 결과 출력
    print(json_data)
    print()
    prim_id = json_data['resource']
    ds0 = json_data['body']['datastreams'][0]['current_value']
    ds1 = json_data['body']['datastreams'][1]['current_value']
    ds2 = json_data['body']['datastreams'][2]['current_value']
    print(prim_id, ds0, ds1, ds2)

def main():
    # uwb_data.json 파일 생성, 한 번만 실행하면됨됨
    # create_data_file()
    
    client = Minio(f"10.32.174.125:9040",
        access_key="pRWLQmzIoCE5nUKyac1O",
        secret_key="8FpYdGdHL14opVBipvvGzjScTMNaQSHOjH9WaUZp",
        secure=False
    )
    # upload_file(client, 5)
    # get_metadata(client)
    get_file_info(client)

    
def create_data_file():
    uwb_data = {
        "body": {
            "id": "16",
            "datastreams": [
                {"id": "posX", "current_value": "22.62", "at": "2024-07-10 06:22:09.804"},
                {"id": "posY", "current_value": "-21.40", "at": "2024-07-10 06:22:09.804"},
                {"id": "clr", "current_value": "0.36", "at": "2024-07-10 06:22:09.804"},
                {"id": "numberOfAnchors", "current_value": "6", "at": "2024-07-10 06:22:09.804"}
            ],
            "uuid": "ecb96332-f4b2-11ed-af60-230463e1fe65",
            "address": "0x230463E1FE65",
            "extended_tag_position": {
                "master": "E8EB1B3BD8A8",
                "corrected": True,
                "slaves": [
                    {"address": "8034282787E9", "time": 0.806366695305684, "fp": -85.3, "rssi": -83.9},
                    {"address": "803428275EA7", "time": 0.806366678215719, "fp": -85.71, "rssi": -81.5},
                    {"address": "80342828007E", "time": 0.806366706818745, "fp": -97.08, "rssi": -90.36},
                    {"address": "80342827FDF2", "time": 0.806366700821338, "fp": -96.99, "rssi": -88.11},
                    {"address": "E8EB1B3BD8A8", "time": 0.806366678858048, "fp": -96.77, "rssi": -83.7}
                ]
            }
        },
        "resource": "/feeds/16"
    }

    output_file = "uwb_data.json"
    with open(output_file, "w") as f:
        json.dump(uwb_data, f, indent=4)
        print("파일 저장 완료")
        
if __name__ == "__main__":
    main()