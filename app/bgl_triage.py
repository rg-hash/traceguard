from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from app.adaptation import select_balanced_support_set
from app.cross_domain import incident_to_cross_domain_text
from app.evidence import rank_evidence_lines


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BGL_DEVELOPMENT_PATH = (
    PROJECT_ROOT / "data/processed/bgl_development_windows.jsonl"
)

BGL_MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts/cross_domain"
    / "bgl_locked_holdout_support_600_seed_7.joblib"
)

BGL_SUPPORT_SIZE = 600
BGL_SUPPORT_SELECTION_SEED = 7
BGL_TOP_K = 3

BGL_WINDOW_NUMBER = re.compile(r"bgl-window-(\d+)$")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load BGL incidents stored as one JSON object per line."""
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def bgl_time_key(incident: dict[str, Any]) -> int:
    """
    Extract the chronological BGL window number from an incident ID.

    Example:
    bgl-window-001234 -> 1234
    """
    match = BGL_WINDOW_NUMBER.search(str(incident["incident_id"]))

    if not match:
        raise ValueError(
            f"Invalid BGL incident ID: {incident['incident_id']}"
        )

    return int(match.group(1))


class BGLTriageService:
    """
    Production wrapper around the locked-holdout BGL model.

    It uses only development-era labelled support evidence. The locked
    future holdout remains evaluation-only and is never loaded here.
    """

    def __init__(
        self,
        vectorizer: Any,
        classifier: Any,
        support_incidents: list[dict[str, Any]],
        anomaly_threshold: float,
        safety_policy: dict[str, Any],
    ) -> None:
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.support_incidents = support_incidents
        self.anomaly_threshold = anomaly_threshold
        self.safety_policy = safety_policy

        self.support_texts = [
            incident_to_cross_domain_text(incident)
            for incident in support_incidents
        ]

        self.support_labels = [
            int(incident["is_anomaly"])
            for incident in support_incidents
        ]

        # TF-IDF vectors are L2-normalized, so dot product equals
        # cosine similarity for these incident representations.
        self.support_matrix = self.vectorizer.transform(
            self.support_texts
        )

    def triage(
        self,
        incident_id: str,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Analyze caller-provided BGL logs safely.

        Returns LIKELY_ANOMALY / LIKELY_NORMAL only if all safety checks
        pass. Otherwise, returns NEEDS_HUMAN_REVIEW.
        """
        incident = {
            "incident_id": incident_id,
            "events": events,
        }

        query_text = incident_to_cross_domain_text(incident)

        if not query_text:
            raise ValueError("No usable log messages were provided.")

        query_matrix = self.vectorizer.transform([query_text])

        anomaly_probability = float(
            self.classifier.predict_proba(query_matrix)[0, 1]
        )

        predicted_label = int(
            anomaly_probability >= self.anomaly_threshold
        )

        classifier_confidence = (
            anomaly_probability
            if predicted_label == 1
            else 1.0 - anomaly_probability
        )

        similarities = (
            query_matrix @ self.support_matrix.T
        ).toarray().ravel()

        top_indices = similarities.argsort()[::-1][:BGL_TOP_K]
        top_scores = [float(similarities[index]) for index in top_indices]
        evidence_labels = [
            self.support_labels[index]
            for index in top_indices
        ]

        unanimous_evidence = len(set(evidence_labels)) == 1
        evidence_label = evidence_labels[0]
        classifier_evidence_agree = (
            predicted_label == evidence_label
        )

        top_similarity = top_scores[0]

        confidence_passed = (
            classifier_confidence
            >= float(self.safety_policy["confidence_threshold"])
        )

        similarity_passed = (
            top_similarity
            >= float(self.safety_policy["similarity_threshold"])
        )

        automated = (
            confidence_passed
            and similarity_passed
            and unanimous_evidence
            and classifier_evidence_agree
        )

        if automated:
            recommendation = (
                "LIKELY_ANOMALY"
                if predicted_label == 1
                else "LIKELY_NORMAL"
            )
        else:
            recommendation = "NEEDS_HUMAN_REVIEW"

        evidence = []

        for rank, index in enumerate(top_indices, start=1):
            historical_incident = self.support_incidents[index]
            historical_text = self.support_texts[index]
            historical_label = self.support_labels[index]

            evidence.append(
                {
                    "rank": rank,
                    "historical_incident_id": historical_incident[
                        "incident_id"
                    ],
                    "historical_label": (
                        "Anomaly"
                        if historical_label == 1
                        else "Normal"
                    ),
                    "similarity": round(
                        float(similarities[index]),
                        4,
                    ),
                    "cited_templates": rank_evidence_lines(
                        query_text=query_text,
                        evidence_text=historical_text,
                        max_lines=2,
                    ),
                }
            )

        decision_checks = {
            "classifier_confidence": round(classifier_confidence, 4),
            "minimum_confidence": float(
                self.safety_policy["confidence_threshold"]
            ),
            "confidence_passed": confidence_passed,
            "top_similarity": round(top_similarity, 4),
            "minimum_similarity": float(
                self.safety_policy["similarity_threshold"]
            ),
            "similarity_passed": similarity_passed,
            "top_3_evidence_unanimous": unanimous_evidence,
            "classifier_evidence_agree": classifier_evidence_agree,
        }

        return {
            "incident_id": incident_id,
            "recommendation": recommendation,
            "classifier_anomaly_probability": round(
                anomaly_probability,
                4,
            ),
            "classifier_prediction": (
                "Anomaly"
                if predicted_label == 1
                else "Normal"
            ),
            "policy": {
                "model": "TF-IDF + Logistic Regression",
                "support_size": len(self.support_incidents),
                "support_selection_seed": BGL_SUPPORT_SELECTION_SEED,
                "anomaly_threshold": self.anomaly_threshold,
                "requires_unanimous_top_3_evidence": True,
                "requires_classifier_evidence_agreement": True,
                **self.safety_policy,
            },
            "decision_checks": decision_checks,
            "evidence_labels": evidence_labels,
            "evidence": evidence,
        }


@lru_cache
def get_bgl_triage_service() -> BGLTriageService:
    """
    Load and reconstruct the evaluated BGL deployment configuration once.

    The support set is selected from the first 60% of chronological BGL
    development data using the same seed as the final model artifact.
    """
    if not BGL_MODEL_PATH.exists():
        raise RuntimeError(
            "BGL deployment model is missing. Expected: "
            f"{BGL_MODEL_PATH}"
        )

    model_bundle = joblib.load(BGL_MODEL_PATH)

    development_incidents = load_jsonl(BGL_DEVELOPMENT_PATH)
    development_incidents.sort(key=bgl_time_key)

    support_pool_end = int(len(development_incidents) * 0.60)
    support_pool = development_incidents[:support_pool_end]

    support_incidents = select_balanced_support_set(
        support_pool,
        total_size=int(
            model_bundle.get("support_size", BGL_SUPPORT_SIZE)
        ),
        seed=int(
            model_bundle.get(
                "support_selection_seed",
                BGL_SUPPORT_SELECTION_SEED,
            )
        ),
    )

    return BGLTriageService(
    vectorizer=model_bundle["vectorizer"],
    classifier=model_bundle["classifier"],
    support_incidents=support_incidents,
    anomaly_threshold=float(model_bundle["anomaly_threshold"]),
    safety_policy=dict(model_bundle["safety_policy"]),
)