import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATASET_PATH = Path("data/processed/bgl_windows.jsonl")
MODEL_PATH = Path("artifacts/bgl_logistic_regression.joblib")


# 1. Load BGL incident windows
rows = []

with DATASET_PATH.open(encoding="utf-8") as file:
    for line in file:
        if line.strip():
            rows.append(json.loads(line))


# 2. Convert each BGL window into one text input
texts = [
    " ".join(event["message"] for event in row["events"])
    for row in rows
]

labels = np.array([
    row["is_anomaly"]
    for row in rows
])

print(f"Total BGL windows: {len(rows)}")
print(f"Normal windows: {sum(labels == 0)}")
print(f"Anomaly windows: {sum(labels == 1)}")


# 3. Split data:
# 60% train, 20% validation, 20% untouched test
x_train_all, x_test, y_train_all, y_test = train_test_split(
    texts,
    labels,
    test_size=0.20,
    random_state=7,
    stratify=labels
)

x_train, x_validation, y_train, y_validation = train_test_split(
    x_train_all,
    y_train_all,
    test_size=0.25,
    random_state=7,
    stratify=y_train_all
)


# 4. TF-IDF + Logistic Regression baseline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=50_000
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=7
        )
    )
])


# 5. Train only on training data
model.fit(x_train, y_train)


# 6. Select threshold using validation data only
validation_scores = model.predict_proba(x_validation)[:, 1]

best_threshold = None
best_f1 = -1

for threshold in np.linspace(0.05, 0.95, 91):
    validation_predictions = (
        validation_scores >= threshold
    ).astype(int)

    score = f1_score(
        y_validation,
        validation_predictions
    )

    if score > best_f1:
        best_f1 = score
        best_threshold = threshold

print(f"\nSelected validation threshold: {best_threshold:.2f}")
print(f"Validation F1: {best_f1:.3f}")


# 7. Evaluate once on untouched test data
test_scores = model.predict_proba(x_test)[:, 1]

test_predictions = (
    test_scores >= best_threshold
).astype(int)

print("\nBGL Logistic Regression Test Report:\n")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["Normal", "Anomaly"],
        digits=3
    )
)

print(f"Anomaly Precision: {precision_score(y_test, test_predictions):.3f}")
print(f"Anomaly Recall: {recall_score(y_test, test_predictions):.3f}")
print(f"Anomaly F1 Score: {f1_score(y_test, test_predictions):.3f}")
print(f"PR-AUC Score: {average_precision_score(y_test, test_scores):.3f}")


# 8. Save trained model, threshold, and metadata
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "threshold": best_threshold,
        "dataset": "LogHub BGL",
        "model_type": "TF-IDF + Logistic Regression",
        "window_size": 20
    },
    MODEL_PATH
)

print(f"\nSaved model to: {MODEL_PATH}")