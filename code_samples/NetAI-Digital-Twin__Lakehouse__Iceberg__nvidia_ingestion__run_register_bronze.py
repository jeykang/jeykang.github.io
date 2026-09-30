"""Thin launcher so spark-submit runs register_bronze as a package module
(its relative imports need it loaded as nvidia_ingestion.register_bronze)."""
from nvidia_ingestion.register_bronze import main
if __name__ == "__main__":
    main()
