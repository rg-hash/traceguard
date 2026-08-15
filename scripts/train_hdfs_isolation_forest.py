import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score
)
from sklearn.model_selection import train_test_split

DATASET_PATH = Path("data/processed/hdfs_incidents.jsonl")
MODEL_PATH = Path("artifacts/hdfs_isolation_forest.joblib")


# 1. Load HDFS incidents
rows = []

with DATASET_PATH.open(encoding="utf-8") as file:
    for line in file:
        if line.strip():
            rows.append(json.loads(line))

texts = [
    " ".join(event["message"] for event in row["events"])
    for row in rows
]

labels = np.array([row["is_anomaly"] for row in rows])


# 2. Create train, validation, and untouched test sets
x_train_all, x_test, y_train_all, y_test = train_test_split(
    texts,
    labels,
    test_size=0.20,
    random_state=7,
    stratify=labels
)

x_train_all, x_validation, y_train_all, y_validation = train_test_split(
    x_train_all,
    y_train_all,
    test_size=0.25,
    random_state=7,
    stratify=y_train_all
)

# Keep only normal logs for Isolation Forest training
x_train_normal = [
    text
    for text, label in zip(x_train_all, y_train_all)
    if label == 0
]

print(f"Normal incidents used for training: {len(x_train_normal)}")
print(f"Validation incidents: {len(x_validation)}")
print(f"Test incidents: {len(x_test)}")


# 3. Convert logs to TF-IDF features
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=5000
)

x_train_features = vectorizer.fit_transform(x_train_normal)
x_validation_features = vectorizer.transform(x_validation)
x_test_features = vectorizer.transform(x_test)


# 4. Train unsupervised anomaly detector
model = IsolationForest(
    n_estimators=300,
    contamination="auto",
    random_state=7,
    n_jobs=-1
)

model.fit(x_train_features)


# 5. Higher score = more anomalous
validation_scores = -model.score_samples(x_validation_features)
test_scores = -model.score_samples(x_test_features)


# 6. Select threshold only on validation data
candidate_thresholds = np.quantile(
    validation_scores,
    np.linspace(0.05, 0.95, 100)
)

best_threshold = None
best_f1 = -1

for threshold in candidate_thresholds:
    validation_predictions = (
        validation_scores >= threshold
    ).astype(int)

    score = f1_score(y_validation, validation_predictions)

    if score > best_f1:
        best_f1 = score
        best_threshold = threshold

print(f"\nSelected validation threshold: {best_threshold:.6f}")
print(f"Best validation F1: {best_f1:.3f}")


# 7. Final evaluation only on untouched test data
test_predictions = (
    test_scores >= best_threshold
).astype(int)

print("\nIsolation Forest Test Report:\n")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["Normal", "Anomaly"],
        digits=3
    )
)

print(f"Anomaly F1 Score: {f1_score(y_test, test_predictions):.3f}")
print(f"PR-AUC Score: {average_precision_score(y_test, test_scores):.3f}")


# 8. Save model and vectorizer
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "vectorizer": vectorizer,
        "model": model,
        "threshold": best_threshold,
        "model_type": "Isolation Forest",
        "training_data": "Normal HDFS blocks only"
    },
    MODEL_PATH
)

print(f"\nSaved model to: {MODEL_PATH}")