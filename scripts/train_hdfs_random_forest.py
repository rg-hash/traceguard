import json
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATASET_PATH = Path("data/processed/hdfs_incidents.jsonl")


# Load grouped HDFS incidents
rows = []

with DATASET_PATH.open(encoding="utf-8") as file:
    for line in file:
        if line.strip():
            rows.append(json.loads(line))


# Convert each incident's log events to text
texts = [
    " ".join(event["message"] for event in row["events"])
    for row in rows
]

labels = [row["is_anomaly"] for row in rows]


# Same split as Logistic Regression for fair comparison
x_train, x_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.25,
    random_state=7,
    stratify=labels
)


# TF-IDF features + Random Forest classifier
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=5000
        )
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=7,
            n_jobs=-1
        )
    )
])


# Train
model.fit(x_train, y_train)


# Evaluate
probabilities = model.predict_proba(x_test)[:, 1]

SELECTED_THRESHOLD = 0.4
predictions = (probabilities >= SELECTED_THRESHOLD).astype(int)

print("\nRandom Forest Classification Report:\n")
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


vectorizer = model.named_steps["tfidf"]
classifier = model.named_steps["classifier"]

feature_names = vectorizer.get_feature_names_out()
importances = classifier.feature_importances_

top_features = sorted(
    zip(feature_names, importances),
    key=lambda item: item[1],
    reverse=True
)[:30]

print("\nTop 30 influential features:\n")

for feature, importance in top_features:
    print(f"{feature:40} {importance:.6f}")