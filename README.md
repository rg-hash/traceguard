# TraceGuard

TraceGuard is an evidence-grounded AIOps prototype for detecting anomalous log windows, ranking likely root causes, and safely escalating incidents when the available evidence is weak.

It is deliberately designed as a portfolio-quality project rather than a generic chat interface:

```text
log events -> feature extraction -> anomaly detector -> root-cause ranker
                                                |-> evidence retrieval -> confidence gate -> API response
```

## What is implemented

- A reproducible synthetic telecom-style incident dataset generator (safe to publish).
- Supervised anomaly-detection baselines: logistic regression and random forest.
- Root-cause ranking over `database`, `network`, and `application` failures.
- Deterministic, cited evidence retrieval from an incident's log window.
- A conservative abstention policy based on calibrated probability and evidence support.
- FastAPI service, Docker image, tests, and GitHub Actions quality gate.

The generated data is a starter dataset; it **does not** represent Oracle systems or production telecom data.

## Quick start

Requires Python 3.11+.

```bash
cd traceguard
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_demo_data.py
python scripts/train.py
uvicorn app.api:app --reload
```

Open `http://127.0.0.1:8000/docs`, or query an incident:

```bash
curl http://127.0.0.1:8000/incidents/demo-network-001/analyze
```

Run the verification suite:

```bash
pytest
python scripts/evaluate.py
```

## Data and evaluation plan

The data generator creates normal and anomalous windows with causal signatures, noisy distractor logs, timestamps, service names, severity, and a ground-truth root cause. It supports an end-to-end demo without proprietary information. For a research extension, adapt `app/data.py` to LogHub HDFS/BGL logs and retain the same train/test split and metrics. LogHub provides publicly accessible research log datasets. [LogHub](https://github.com/logpai/loghub)

Report these metrics in `docs/report.md` after training:

- anomaly detection: precision, recall, F1, PR-AUC;
- root cause: top-1 accuracy and mean reciprocal rank;
- retrieval: evidence recall@3;
- safety: coverage, abstention rate, and unsafe confident-answer rate;
- systems: p50/p95 API latency using a small load test.

## Responsible-AI design

TraceGuard does not execute remediation. A low-confidence classification or evidence score produces `NEEDS_HUMAN_REVIEW`, preserving the uncertainty rather than fabricating a root cause. API responses include quoted log evidence and model/version metadata for auditability.

## Repository layout

| Path | Purpose |
| --- | --- |
| `app/` | Dataset, ML pipeline, evidence layer, and API. |
| `scripts/` | Deterministic dataset generation, training, and evaluation. |
| `tests/` | Unit and API safety tests. |
| `docs/` | Architecture, experiment plan, and report template. |
| `.github/workflows/` | CI test and lint workflow. |

## 14-week extension plan

1. Weeks 1–2: reproduce baseline results and create a data card.
2. Weeks 3–4: improve anomaly features and run leakage checks.
3. Weeks 5–7: add a sequence model (DeepLog/Transformer) as a controlled comparison.
4. Weeks 8–9: add service-dependency graph features for root-cause ranking.
5. Weeks 10–11: test calibrated abstention thresholds and evidence-quality ablations.
6. Weeks 12–13: add PostgreSQL/OpenSearch ingestion and Prometheus metrics.
7. Week 14: finalize experiments, report, demo video, and GitHub release.
