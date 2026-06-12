#!/bin/bash

mkdir -p ./minio_data
mkdir -p ./python-scripts
mkdir -p ./trino_data
mkdir -p ./user_data

chmod -R 777 ./minio_data
chmod -R 777 ./python-scripts
chmod -R 777 ./trino_data
chmod -R 777 ./user_data

docker compose up -d

echo -e " - MinIO   : http://localhost:9001 [Example] (ID: AAAAA / PW: BBBBBBBB)"
echo -e " - Polaris : http://localhost:8181"
echo -e " - Trino   : http://localhost:8080"
echo -e " - Superset: http://localhost:8088"
echo -e " - Spark UI: http://localhost:4040 (Activated when the job runs)"

docker compose ps