# TraceGuard

**Evidence-grounded, safety-aware AIOps log triage for public HDFS and BlueGene/L (BGL) datasets.**

TraceGuard analyzes a group of system logs and returns one of three recommendations:

- `LIKELY_ANOMALY`
- `LIKELY_NORMAL`
- `NEEDS_HUMAN_REVIEW`

Unlike a basic anomaly classifier, TraceGuard does not rely only on a prediction score. It retrieves similar historical incidents, cites the normalized log templates behind its decision, and abstains when the model or evidence is not trustworthy enough.

> TraceGuard performs triage only. It does not automatically remediate systems and does not claim root-cause classification for datasets without root-cause labels.

---

## Why TraceGuard?

Operational log models can obtain high offline accuracy yet still be unsafe in production. A reliable AIOps system should answer:

1. What does the model predict?
2. What historical evidence supports that prediction?
3. Does the evidence agree with the model?
4. Is the system sufficiently confident to automate the decision?
5. When should the decision be deferred to an engineer?

TraceGuard implements this workflow as a deployable FastAPI application with PostgreSQL audit records, Prometheus metrics, an operator dashboard, and reproducible experiments.

---

## Key capabilities

- TF-IDF + Logistic Regression anomaly classification baseline
- Random Forest, Isolation Forest, and LSTM benchmark experiments
- Semantic retrieval with Sentence Transformers
- Lexical retrieval with TF-IDF
- Hybrid semantic + lexical HDFS evidence retrieval
- BGL evidence retrieval using a temporally valid 600-incident support set
- Normalization of volatile identifiers such as timestamps, node IDs, block IDs, IP addresses, ports, and long numbers
- Evidence citations: returned decisions include the relevant normalized log templates
- Safety-aware abstention instead of forced predictions
- PostgreSQL decision audit trail
- Prometheus request, latency, persistence, failure, and human-review metrics
- MLflow experiment tracking
- FastAPI documentation and an interactive browser dashboard
- Temporal evaluation, cross-domain transfer experiments, OOD detection, few-shot adaptation, and locked-holdout evaluation
- **Investigation Agent:** a LangGraph workflow that summarizes anomalous logs, retrieves incidents and runbooks, correlates deployments, ranks evidence-backed hypotheses, and returns a read-only debugging plan for engineer review

---

## Investigation Agent

When a triage result is `LIKELY_ANOMALY` or `NEEDS_HUMAN_REVIEW`, call
`POST /investigate` to create an engineering investigation plan. The agent is
strictly read-only: it cannot execute commands, restart services, roll back a
deployment, or change infrastructure.

```text
logs → incident summary → hybrid evidence retrieval → deployment context
     → deterministic hypothesis ranking → cited runbook checks → engineer review
```

The initial demo knowledge base includes database connection-pool, DNS/network,
and post-deployment regression incidents, plus their approved runbooks. It is
deliberately small and versioned so the workflow can be tested end-to-end before
connecting Jira, GitHub, Kubernetes, or an incident-management platform.

Example request:

```json
{
  "incident_id": "payment-incident-1",
  "triage_recommendation": "LIKELY_ANOMALY",
  "events": [{
    "timestamp": "2026-08-20T10:08:00Z",
    "service": "payment-api",
    "message": "ERROR database connection timeout; connection pool exhausted"
  }]
}
```

The response contains a timestamped incident summary, retrieved incident and
runbook IDs, ranked hypotheses, evidence-linked diagnostic checks, and the
fixed decision `ENGINEER_REVIEW_REQUIRED`.

---

## System architecture

```text
Operator pastes BGL logs
          │
          ▼
FastAPI: POST /triage/bgl
          │
          ▼
Normalize variable identifiers
          │
          ├── TF-IDF + Logistic Regression
          │       └── anomaly probability and predicted label
          │
          └── Retrieve top-3 historical BGL evidence incidents
                  └── cosine similarity over TF-IDF vectors
          │
          ▼
Safety policy
  - classifier confidence threshold
  - evidence similarity threshold
  - unanimous top-3 evidence labels
  - classifier/evidence agreement
          │
          ├── all checks pass → LIKELY_NORMAL or LIKELY_ANOMALY
          └── any check fails → NEEDS_HUMAN_REVIEW
          │
          ▼
Return cited evidence + persist decision in PostgreSQL
          │
          ▼
Dashboard, Prometheus metrics, and human-review queue
```

---

## Safety policy

A BGL decision is automated only when all of the following conditions hold:

1. The Logistic Regression classifier confidence meets the validation-selected threshold.
2. The closest retrieved BGL evidence exceeds the similarity threshold.
3. The top-three retrieved evidence incidents have the same label.
4. The model prediction and evidence label agree.

Otherwise, TraceGuard returns:

```json
{
  "recommendation": "NEEDS_HUMAN_REVIEW"
}
```

This is intentional abstention, not a system failure.

For the deployed 600-label BGL configuration:

```text
Classifier confidence threshold: 0.50
Evidence similarity threshold: 0.3558
Top-k evidence required: 3
Classifier/evidence agreement: required
Unanimous evidence labels: required
```

---

## Evaluation highlights

### HDFS hybrid evidence retrieval

HDFS uses a chronological 60/20/20 train/validation/test split.

| Metric | Final temporal test result |
|---|---:|
| Precision@1, same-label evidence | 0.9200 |
| Precision@3, same-label evidence | 0.9242 |
| Automated-decision coverage | 83.0% |
| Automated-decision accuracy | 99.4% |
| Unsafe confident-decision rate | 0.6% |
| Hybrid retrieval latency | 3.58 ms/query |

The HDFS retriever uses a hybrid score:

```text
0.75 × semantic similarity + 0.25 × lexical TF-IDF similarity
```

### BGL locked future holdout

For BGL, the raw timeline was divided before final evaluation:

```text
Development timeline: first 80% of BGL windows
Locked future holdout: final 20% of BGL windows
```

The locked holdout is temporally later and disjoint from development data.

The final evaluation used 600 labelled BGL support incidents and five predeclared support-selection seeds.

| Metric | Mean ± sample standard deviation |
|---|---:|
| Classifier F1 | 0.9267 ± 0.0297 |
| PR-AUC | 0.9828 ± 0.0050 |
| Selective automation coverage | 56.83% ± 6.17% |
| Automated-decision accuracy | 98.36% ± 0.87% |
| Unsafe automated-decision rate | 1.64% ± 0.87% |

Interpretation:

> On a temporally later, disjoint BGL holdout, TraceGuard automated approximately 56.8% of triage decisions with 98.4% accuracy among automated decisions and a 1.6% unsafe-decision rate across five predeclared support selections.

The five runs share the same locked test timeline. The reported variation reflects support-set selection sensitivity, not five independent test populations.

---

## Cross-domain finding

TraceGuard explicitly tested whether an HDFS-trained model transfers safely to BGL logs.

It did not.

```text
HDFS → BGL zero-shot transfer:
- BGL F1 without abstention: 0.5673
- Naive abstention coverage: 91.9%
- Unsafe automated-decision rate: 61.4%
```

This is an important negative result: confidence alone is not enough when the incoming log distribution differs from the training domain.

An out-of-distribution guard based on source-evidence similarity detected BGL as outside the HDFS domain and abstained on all BGL cases rather than producing unsafe automated decisions.

---

## Quick start

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure PostgreSQL

Create a local database and user, then create a local environment file:

```bash
cp .env.example .env
```

Update `.env`:

```bash
DATABASE_URL='postgresql://YOUR_USERNAME:YOUR_PASSWORD@localhost:5432/traceguard'
TRACEGUARD_PORT=8000
```

Do not commit `.env`.

### 4. Start the full local application

```bash
./scripts/run_local.sh
```

Open:

```text
Dashboard: http://127.0.0.1:8000/dashboard/
API docs:  http://127.0.0.1:8000/docs
Metrics:   http://127.0.0.1:8000/metrics
```

---

## Product usage

### Analyze BGL logs using the dashboard

1. Open `http://127.0.0.1:8000/dashboard/`.
2. Enter an incident ID.
3. Paste one BGL log message per line.
4. Select **Analyze safely**.
5. Inspect:
   - anomaly probability,
   - prediction,
   - safety checks,
   - top-three historical evidence incidents,
   - cited normalized templates.
6. Refresh the human-review queue to view abstained incidents.

When pasting logs from LogHub BGL, do not include the leading alert label because it is the offline dataset label.

### Analyze BGL logs using the API

```bash
curl -sS -X POST http://127.0.0.1:8000/triage/bgl \
  -H "Content-Type: application/json" \
  -d '{
    "incident_id": "manual-bgl-normal-001",
    "events": [
      {
        "message": "2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected"
      }
    ]
  }'
```

Example response fields:

```json
{
  "recommendation": "LIKELY_NORMAL",
  "classifier_anomaly_probability": 0.0861,
  "classifier_prediction": "Normal",
  "decision_checks": {
    "confidence_passed": true,
    "similarity_passed": true,
    "top_3_evidence_unanimous": true,
    "classifier_evidence_agree": true
  },
  "evidence": [
    {
      "rank": 1,
      "historical_label": "Normal",
      "similarity": 0.9597,
      "cited_templates": [
        {
          "template": "<date> <node> <timestamp> <node> ras kernel info instruction cache parity error corrected"
        }
      ]
    }
  ],
  "decision_record_id": 1
}
```

### Review persisted decisions

```bash
curl -sS \
  "http://127.0.0.1:8000/decisions?source=bgl_evidence_grounded_triage&limit=20"
```

### Review abstained incidents only

```bash
curl -sS \
  "http://127.0.0.1:8000/decisions?source=bgl_evidence_grounded_triage&recommendation=NEEDS_HUMAN_REVIEW&limit=20"
```

---

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Application health check |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/docs` | Interactive FastAPI documentation |
| `GET` | `/dashboard/` | Operator dashboard |
| `POST` | `/analyze` | Original generic ML analysis endpoint |
| `POST` | `/retrieve/hdfs` | HDFS hybrid evidence retrieval |
| `POST` | `/triage/bgl` | Safe BGL evidence-grounded triage |
| `GET` | `/decisions` | Persisted decision history and review queue |

---

## Prometheus metrics

Example BGL metrics:

```text
traceguard_bgl_triage_requests_total
traceguard_bgl_triage_latency_seconds
traceguard_bgl_triage_failures_total
traceguard_bgl_decisions_persisted_total
traceguard_bgl_human_reviews_total
```

Example query:

```bash
curl -s http://127.0.0.1:8000/metrics | grep "^traceguard_bgl"
```

---

## Data

TraceGuard uses public systems-log datasets:

- HDFS logs
- BlueGene/L (BGL) logs
- OpenStack logs were explored but excluded from the supervised benchmark because the processed dataset contained too few anomaly windows for meaningful evaluation.

Raw large datasets, generated indices, and trained artifacts should not be committed to GitHub. Reproduce them locally with the scripts in `scripts/`.

Relevant preparation and evaluation scripts include:

```text
scripts/prepare_bgl.py
scripts/prepare_bgl_locked_holdout.py
scripts/train_bgl.py
scripts/build_bgl_retrieval_index.py
scripts/build_hdfs_hybrid_index.py
scripts/evaluate_hdfs_hybrid.py
scripts/evaluate_bgl_locked_holdout.py
scripts/evaluate_hdfs_to_bgl_transfer.py
scripts/evaluate_hdfs_to_bgl_ood_guard.py
scripts/evaluate_hdfs_to_bgl_few_shot_adaptation.py
scripts/evaluate_bgl_development_selection.py
```

---

## Repository structure

```text
app/
  api.py                 FastAPI endpoints and application lifecycle
  bgl_triage.py          Deployed BGL classifier, retrieval, and safety policy
  hybrid_retrieval.py    HDFS hybrid semantic + lexical retriever
  retrieval.py           Semantic retrieval implementation
  evidence.py            Evidence-template citation logic
  log_normalization.py   HDFS log normalization
  cross_domain.py        HDFS and BGL cross-domain normalization
  adaptation.py          Reproducible balanced support-set sampling
  ood.py                 Similarity-based OOD detection helpers
  risk_control.py        Wilson-bound risk-control helpers
  database.py            PostgreSQL persistence and review-queue queries
  metrics.py             Prometheus metrics

scripts/
  run_local.sh                      One-command local startup
  evaluate_bgl_locked_holdout.py    Final locked-holdout BGL evaluation
  create_bgl_human_review_demo.py   Creates an abstention demonstration request

static/
  index.html              Browser dashboard

tests/
  test_hdfs_retrieval_api.py
  test_bgl_triage_api.py
  test_cross_domain.py
  test_ood.py
  test_domain_ensemble.py
  test_temporal_backtesting.py
```

---

## Testing

Run the focused BGL API test:

```bash
pytest -q tests/test_bgl_triage_api.py
```

Run the HDFS retrieval API test:

```bash
pytest -q tests/test_hdfs_retrieval_api.py
```

Run all tests:

```bash
pytest -q
```

---

## Reproducibility and limitations

- All major experiments use fixed, reported random seeds.
- BGL final evaluation uses a temporally later locked holdout, not a random split.
- The deployed BGL model uses seed `7`; five-seed results are reported for robustness.
- Retrieval evidence is historical similarity evidence, not a causal explanation.
- The system does not infer root causes where datasets do not provide verified root-cause labels.
- HDFS and BGL have different log structures; cross-domain results show that naive transfer can be unsafe.
- The local dashboard is intended for demonstration and development. Production deployment should add authentication, secret management, structured log ingestion, database connection pooling, and persistent Prometheus scraping.

---

## Project claim

> TraceGuard evaluates evidence-grounded abstention for log-based anomaly triage across public AIOps datasets. It combines classical ML, retrieval-based evidence, temporal evaluation, out-of-distribution detection, and selective prediction to automate only decisions that meet a conservative safety policy.

---

## Author

Riddhi Goyal  
GitHub: [rg-hash](https://github.com/rg-hash)
