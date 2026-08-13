from app.config import DATASET_PATH
from app.data import write_dataset

write_dataset(DATASET_PATH)
print(f"Wrote reproducible demo data to {DATASET_PATH}")
