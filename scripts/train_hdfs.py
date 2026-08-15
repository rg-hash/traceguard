import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score

DATASET_PATH = Path("data/processed/hdfs_incidents.jsonl")
MODEL_PATH = Path("artifacts/hdfs_anomaly_model.joblib")


# 1. Load JSONL incidents
rows = []

with DATASET_PATH.open(encoding="utf-8") as file:
    for line in file:
        if line.strip():
            rows.append(json.loads(line))


# 2. Convert every incident's many log lines into one text input
texts = []
labels = []

for row in rows:
    log_text = " ".join(
        event["message"]
        for event in row["events"]
    )

    texts.append(log_text)
    labels.append(row["is_anomaly"])


# 3. Keep 75% data for training and 25% unseen for testing
x_train, x_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.25,
    random_state=7,
    stratify=labels
)


# 4. Build ML pipeline
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


# 5. Train model
model.fit(x_train, y_train)


# 6. Evaluate model on unseen HDFS incidents
probabilities = model.predict_proba(x_test)[:, 1]
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(
        f"Threshold={threshold:.1f} | "
        f"Precision={precision:.3f} | "
        f"Recall={recall:.3f} | "
        f"F1={f1:.3f}"
    )

SELECTED_THRESHOLD = 0.4

predictions = (
    probabilities >= SELECTED_THRESHOLD
).astype(int)
print("\nClassification Report:\n")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Normal", "Anomaly"],
        digits=3
    )
)

print(f"Anomaly F1 Score: {f1_score(y_test, predictions):.3f}")
print(f"PR-AUC Score: {average_precision_score(y_test, probabilities):.3f}")


# 7. Save trained model
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"\nSaved model to: {MODEL_PATH}")