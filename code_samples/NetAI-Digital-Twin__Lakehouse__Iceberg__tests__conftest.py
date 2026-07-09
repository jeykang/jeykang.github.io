import pathlib
import sys

# Make the repo root importable so `nvidia_ingestion` / `kaist_ingestion` resolve
# regardless of pytest's rootdir insertion mode.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
