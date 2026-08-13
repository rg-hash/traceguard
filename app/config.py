from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "generated"
ARTIFACT_DIR = ROOT / "artifacts"
DATASET_PATH = DATA_DIR / "incidents.jsonl"
MODEL_PATH = ARTIFACT_DIR / "models.joblib"
