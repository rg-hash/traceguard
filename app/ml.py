"""Training and inference pipeline; intentionally simple baselines before deep models."""
from __future__ import annotations

from collections import Counter
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

CAUSE_LABELS = ["database", "network", "application"]


def incident_text(incident: dict[str, Any]) -> str:
    return " ".join(event["message"] for event in incident["events"])


def train_models(rows: list[dict[str, Any]]) -> dict[str, Any]:
    texts = [incident_text(row) for row in rows]
    anomaly_targets = [row["is_anomaly"] for row in rows]
    anomaly_model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=7)),
    ])
    anomaly_model.fit(texts, anomaly_targets)

    anomalous_rows = [row for row in rows if row["is_anomaly"]]
    cause_texts = [incident_text(row) for row in anomalous_rows]
    causes = [row["root_cause"] for row in anomalous_rows]
    cause_model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("classifier", RandomForestClassifier(n_estimators=150, random_state=7, class_weight="balanced")),
    ])
    cause_model.fit(cause_texts, causes)
    return {"anomaly_model": anomaly_model, "cause_model": cause_model, "version": "baseline-1.0"}


def retrieve_evidence(incident: dict[str, Any], predicted_cause: str) -> list[dict[str, str]]:
    keywords = set({"database": ["db", "postgres", "query"], "network": ["packet", "dns", "upstream"], "application": ["exception", "handler", "version"]}[predicted_cause])
    matches = []
    for event in incident["events"]:
        score = sum(keyword in event["message"].lower() for keyword in keywords)
        if score:
            matches.append((score, event))
    return [event for _, event in sorted(matches, key=lambda pair: pair[0], reverse=True)[:3]]


def analyze(incident: dict[str, Any], models: dict[str, Any], threshold: float = 0.70) -> dict[str, Any]:
    text = incident_text(incident)
    anomaly_probability = float(models["anomaly_model"].predict_proba([text])[0][1])
    cause_probabilities = models["cause_model"].predict_proba([text])[0]
    classes = models["cause_model"].named_steps["classifier"].classes_
    ordered = sorted(zip(classes, cause_probabilities), key=lambda item: item[1], reverse=True)
    root_cause, cause_confidence = ordered[0]
    evidence = retrieve_evidence(incident, root_cause)
    confidence = min(anomaly_probability, float(cause_confidence))
    decision = "ANALYZED" if confidence >= threshold and evidence else "NEEDS_HUMAN_REVIEW"
    return {
        "incident_id": incident["incident_id"], "decision": decision,
        "anomaly_probability": round(anomaly_probability, 4),
        "root_cause": root_cause if decision == "ANALYZED" else None,
        "root_cause_confidence": round(float(cause_confidence), 4),
        "ranked_causes": [{"cause": label, "probability": round(float(prob), 4)} for label, prob in ordered],
        "evidence": evidence,
        "model_version": models["version"],
        "reason": "Confidence or evidence support is insufficient; route to an operator." if decision != "ANALYZED" else "Classification is supported by retrieved log evidence.",
    }
