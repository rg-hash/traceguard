"""Holdout evaluation; avoids reporting metrics on the training data."""
from sklearn.metrics import classification_report, f1_score

from app.config import DATASET_PATH
from app.data import load_dataset
from app.ml import analyze, incident_text, train_models

rows = load_dataset(DATASET_PATH)
split = int(len(rows) * 0.75)
models = train_models(rows[:split])
holdout = rows[split:]
truth = [row["is_anomaly"] for row in holdout]
predicted = [int(models["anomaly_model"].predict_proba([incident_text(row)])[0][1] >= 0.5) for row in holdout]
anomalous = [row for row in holdout if row["is_anomaly"]]
top1 = sum(analyze(row, models, threshold=0.0)["root_cause"] == row["root_cause"] for row in anomalous) / len(anomalous)
print(classification_report(truth, predicted, digits=3))
print(f"Anomaly F1: {f1_score(truth, predicted):.3f}")
print(f"Root-cause top-1 accuracy: {top1:.3f}")
