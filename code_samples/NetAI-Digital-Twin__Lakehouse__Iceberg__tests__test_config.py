"""Config-resolution tests — run in CI WITHOUT pyspark or any live service.

Guards the generalization contract: (1) config is importable without a Spark
install, (2) defaults are the local/compose dev values, (3) every knob is
env-overridable (for K8s/prod), (4) every env var the code reads is documented
in .env.example (so deploys never miss a variable).
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_config_imports_without_pyspark():
    import importlib
    m = importlib.import_module("nvidia_ingestion.config")
    assert hasattr(m, "NvidiaPipelineConfig") and hasattr(m, "build_spark_session")


def test_defaults_are_compose_friendly(monkeypatch):
    for k in ("POLARIS_URI", "AWS_S3_ENDPOINT", "NVIDIA_NS_BRONZE", "SPARK_CATALOG_NAME"):
        monkeypatch.delenv(k, raising=False)
    from nvidia_ingestion.config import NvidiaPipelineConfig
    c = NvidiaPipelineConfig()
    assert c.catalog.uri == "http://polaris:8181/api/catalog"
    assert c.storage.endpoint == "http://minio:9000"
    assert c.nvidia.namespace_bronze == "nvidia_bronze"
    assert c.spark_catalog_name == "iceberg"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("POLARIS_URI", "http://polaris.lakehouse.svc:8181/api/catalog")
    monkeypatch.setenv("AWS_S3_ENDPOINT", "http://minio.lakehouse.svc:9000")
    monkeypatch.setenv("NVIDIA_NS_BRONZE", "prod_bronze")
    monkeypatch.setenv("SPARK_CATALOG_NAME", "prod_cat")
    monkeypatch.setenv("NVIDIA_SOURCE_PATH", "/data/ssd_batch_01")
    from nvidia_ingestion.config import NvidiaPipelineConfig
    c = NvidiaPipelineConfig()
    assert c.catalog.uri.endswith("svc:8181/api/catalog")
    assert c.storage.endpoint.endswith("svc:9000")
    assert c.nvidia.namespace_bronze == "prod_bronze"
    assert c.spark_catalog_name == "prod_cat"
    assert c.nvidia.source_path == "/data/ssd_batch_01"


def test_env_example_documents_every_knob():
    env_example = (ROOT / ".env.example").read_text()
    src = ((ROOT / "kaist_ingestion" / "config.py").read_text()
           + (ROOT / "nvidia_ingestion" / "config.py").read_text())
    used = set(re.findall(r'_env\(\s*"([A-Z0-9_]+)"', src))
    missing = sorted(v for v in used if v not in env_example)
    assert not missing, f"env vars read by config but missing from .env.example: {missing}"
