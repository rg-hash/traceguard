import sys
from pathlib import Path

# Permit `python scripts/<script>.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import DATASET_PATH
from app.data import write_dataset

write_dataset(DATASET_PATH)
print(f"Wrote reproducible demo data to {DATASET_PATH}")
