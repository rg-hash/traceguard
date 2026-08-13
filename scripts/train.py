import joblib
from app.config import ARTIFACT_DIR, DATASET_PATH, MODEL_PATH
from app.data import load_dataset
from app.ml import train_models

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(train_models(load_dataset(DATASET_PATH)), MODEL_PATH)
print(f"Saved model artifact to {MODEL_PATH}")
