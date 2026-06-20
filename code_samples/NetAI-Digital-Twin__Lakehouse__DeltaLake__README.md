kafka, MinIO, PostgreSQL을 사용하여 DeltaLake사용

workspace_dir: workspace 컨테이너에 mount된 폴더.  
workspace 컨테이너는 직접 코드를 수정하고 gpu를 사용할 수 있는 컨테이너

docker-compose up  
docker-compose up --force-recreate

kafdrop: 9000  
minio-console: 9003  
pgAdmin(postgres): 5050  서버 -> 객체 -> 등록 -> 서버로 서버 등록해야 함
