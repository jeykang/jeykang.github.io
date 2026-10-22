# Lakehouse deployment

Helm chart to deploy the medallion lakehouse (Polaris + Postgres + MinIO + Spark) to
Kubernetes. All knobs are parameterized in `helm/lakehouse/values.yaml` (defaults mirror
`docker-compose.yml` / `.env.example`, so dev is unaffected); override per environment.

## Key property: deploy before or after the data lands
Ingestion is **register-in-place over data-at-rest** — the services come up empty and only
touch raw data when you run the ingestion jobs. So the SSD data can be uploaded first and
the lakehouse stood up later (or vice-versa). Register-in-place stores **absolute paths** in
Iceberg manifests, so the raw-data mount (`rawData.mountPath`) must stay **stable** after
registration.

## Deploy
```bash
# dev / self-contained (in-cluster MinIO + chart-managed secrets + PVCs)
helm upgrade --install lakehouse deploy/helm/lakehouse -n lakehouse --create-namespace

# prod (external secret + admin-bound raw-data PVC + real image)
kubectl create secret generic lakehouse-secrets -n lakehouse \
  --from-literal=POLARIS_CREDENTIAL=... --from-literal=AWS_ACCESS_KEY_ID=... # (see .env.example [SECRET] rows)
helm upgrade --install lakehouse deploy/helm/lakehouse -n lakehouse \
  -f deploy/helm/lakehouse/values-prod.yaml
```

## Run ingestion (after data is uploaded under `rawData.mountPath`)
```bash
POD=$(kubectl get pod -l app=spark -n lakehouse -o name | head -1)
kubectl exec -it $POD -n lakehouse -- spark-submit nvidia_ingestion/register_bronze.py
kubectl exec -it $POD -n lakehouse -- spark-submit nvidia_ingestion/run_gold_score.py --gold-axis camera
```

## Layout
- `helm/lakehouse/` — the chart (`values.yaml` = definable variables; `values-prod.yaml` = overlay).
- `docker/Dockerfile.spark` — the Spark image with ingestion code baked in (built by CI).
- Config contract + `[SECRET]` flags: repo-root `.env.example`. Config code: `kaist_ingestion/config.py`, `nvidia_ingestion/config.py` (env-driven, pyspark-free to import).
