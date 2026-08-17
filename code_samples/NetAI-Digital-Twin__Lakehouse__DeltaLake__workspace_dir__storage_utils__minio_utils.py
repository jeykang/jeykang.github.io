""" 
1. 버킷 생성
2. 파일 업로드
3. 파일 다운로드
"""
from minio import Minio
from minio.error import S3Error
import json
import os
from time import sleep

from dotenv import load_dotenv
load_dotenv(dotenv_path='.my_env')

def create_bucket(client: Minio, bucket_name: str):
    found = client.bucket_exists(bucket_name)
    if not found:
        client.make_bucket(bucket_name)
        print("Created bucket", bucket_name)
    else:
        print("Bucket", bucket_name, "already exists")
    print("========")
    sleep(1)

def test_minio_connection():
    try:
        # MinIO 클라이언트 초기화
        client = Minio(
            f"{os.getenv('LOCAL_IP_ADDRESS')}:9040",
            access_key="pRWLQmzIoCE5nUKyac1O",
            secret_key="8FpYdGdHL14opVBipvvGzjScTMNaQSHOjH9WaUZp",
            secure=False  # http 사용시 False
        )
        
        # 버킷 리스트 가져오기
        buckets = client.list_buckets()
        print("버킷 목록:")
        for bucket in buckets:
            print(f" - {bucket.name}")
            
        # 특정 버킷의 객체 리스트 가져오기
        objects = client.list_objects('python-test-bucket')
        print("\n'python-test-bucket' 버킷의 객체:")
        for obj in objects:
            print(f" - {obj.object_name}")
            
    except S3Error as e:
        print(f"에러 발생: {e}")
                
def main():
    # Create a client with the MinIO server playground, its access key
    # and secret key.
    client = Minio(f"{os.getenv('LOCAL_IP_ADDRESS')}:9040",
        access_key="pRWLQmzIoCE5nUKyac1O",
        secret_key="8FpYdGdHL14opVBipvvGzjScTMNaQSHOjH9WaUZp",
        secure=False
    )
    
    input_option = -1
    while input_option != 0:
        input_option = int(input("\nSelect an Option\n1: Create Bucket\n2: Upload File\n3: Download File\n4: Test MinIO Connection\n0: Exit\n--> "))

        if input_option == 1:
            bucket_name = input("Please Enter the Bucket Name: ")
            # The bucket to create
            create_bucket(client, bucket_name)
        elif input_option == 2:
            print("업로드파일 작성")
        elif input_option == 3:
            print("다운로드파일 작성")
        elif input_option == 4:
            test_minio_connection()
            
    # # The file to upload, change this path if needed
    # source_file = "sample_logs.json"

    # # The destination bucket and filename on the MinIO server
    # bucket_name = "deltalake-bucket"
    # destination_file = "logs_test.json"

    # # Make the bucket if it doesn't exist.
    # create_bucket(client, bucket_name)

    # # Upload the file, renaming it in the process,
    # client.fput_object(
    #     bucket_name, destination_file, source_file
    # )
    # print(
    #     source_file, "successfully uploaded as object",
    #     destination_file, "to bucket", bucket_name,
    # )
    
    # # Download the file
    # try:
    #     client.fget_object(
    #         bucket_name, destination_file, './downloaded_logs.json'
    #     )
    #     print(
    #         source_file, "successfully downloaded as",
    #         destination_file, "from bucket", bucket_name,
    #     )
    # except S3Error as exc:
    #     print("Error occurred while downloading.", exc)    

if __name__ == "__main__":
    try:
        main()
    except S3Error as exc:
        print("error occurred.", exc)