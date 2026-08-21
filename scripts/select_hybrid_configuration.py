"""Select hybrid retrieval settings on development data only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.investigation import InvestigationService

DEVELOPMENT_PATH = PROJECT_ROOT / "data/evaluation/investigation_development.json"
KNOWLEDGE_PATH = PROJECT_ROOT / "data/knowledge/incident_knowledge.json"
DEPLOYMENTS_PATH = PROJECT_ROOT / "data/knowledge/deployments.json"
SEMANTIC_WEIGHTS = [0.40, 0.55, 0.70, 0.85]
EVIDENCE_THRESHOLDS = [0.20, 0.26, 0.32, 0.38]


def score(service: InvestigationService, cases: list[dict]) -> dict:
    known = [case for case in cases if case["expected_root_cause"]]
    unknown = [case for case in cases if not case["expected_root_cause"]]
    top_1 = unknown_abstentions = emitted = correct = 0
    for case in cases:
        result = service.investigate(
            incident_id=case["incident_id"], events=case["events"],
            triage_recommendation="NEEDS_HUMAN_REVIEW", triage_context={},
        )
        predicted = [item["cause"] for item in result["hypotheses"]]
        expected = case["expected_root_cause"]
        if expected:
            top_1 += bool(predicted) and predicted[0] == expected
            emitted += len(predicted)
            correct += sum(cause == expected for cause in predicted)
        else:
            unknown_abstentions += not predicted
    top_1_accuracy = top_1 / len(known) if known else 0.0
    abstention_rate = unknown_abstentions / len(unknown) if unknown else 0.0
    precision = correct / emitted if emitted else 0.0
    return {
        "top_1_accuracy": top_1_accuracy,
        "unknown_abstention_rate": abstention_rate,
        "hypothesis_precision": precision,
        "objective": 0.50 * abstention_rate + 0.35 * top_1_accuracy + 0.15 * precision,
    }


if __name__ == "__main__":
    cases = json.loads(DEVELOPMENT_PATH.read_text())
    service = InvestigationService(
        json.loads(KNOWLEDGE_PATH.read_text()),
        json.loads(DEPLOYMENTS_PATH.read_text()),
    )
    trials = []
    for weight in SEMANTIC_WEIGHTS:
        for threshold in EVIDENCE_THRESHOLDS:
            service.semantic_weight = weight
            service.lexical_weight = 1.0 - weight
            service.minimum_hypothesis_evidence = threshold
            trials.append({
                "semantic_weight": weight,
                "minimum_hypothesis_evidence": threshold,
                **score(service, cases),
            })
    trials.sort(key=lambda item: item["objective"], reverse=True)
    for trial in trials:
        print(json.dumps(trial, sort_keys=True))
    print("\nSELECTED_CONFIGURATION")
    print(json.dumps(trials[0], indent=2, sort_keys=True))
