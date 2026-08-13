TraceGuard is an **AI system for investigating software/network incidents from log messages**.

In simple terms: when a system fails, it reads the logs, decides if there is a real problem, predicts the likely cause, shows the logs supporting that conclusion, and asks a human to review if it is not confident.

```text
System logs
    ↓
Is something unusual happening?
    ↓
What is the likely cause?
    ↓
Which log lines support that conclusion?
    ↓
Confident → report diagnosis
Uncertain → send to human reviewer
```

## What each block does

| Block | Non-technical meaning | Technical implementation |
|---|---|---|
| Synthetic data generator | Creates safe fake incident logs so you can develop publicly without exposing Oracle/customer information. | `app/data.py` creates 240 JSONL log windows with normal, database, network, and application-failure patterns. |
| Data loader | Reads available incidents for training or analysis. | Loads `data/generated/incidents.jsonl`; automatically creates it if missing. |
| Feature extraction | Converts human-readable logs into a form an ML model can process. | Joins incident log messages and uses TF-IDF to represent important words and phrases numerically. |
| Anomaly detector | Answers: “Does this set of logs look abnormal?” | Logistic Regression predicts anomaly probability. |
| Root-cause ranker | Answers: “If abnormal, is the likely issue database, network, or application?” | Random Forest classifies anomalous logs and returns ranked probabilities for all three causes. |
| Evidence retrieval | Shows why the system reached its conclusion instead of returning a black-box answer. | Finds relevant log messages using cause-specific keywords such as `packet`, `dns`, and `upstream` for network incidents. |
| Confidence gate | Prevents confident-looking but unsupported diagnoses. | The system combines anomaly and root-cause confidence. If confidence is below 70% or no supporting evidence is found, it returns `NEEDS_HUMAN_REVIEW`. |
| FastAPI backend | Makes the intelligence usable by a dashboard, automation pipeline, or another service. | `app/api.py` exposes REST endpoints and automatically creates Swagger documentation. |
| Tests | Ensures expected safety and API behavior keep working after changes. | Pytest tests network diagnosis, normal-log abstention, health endpoint, and 404 handling. |
| Evaluation script | Measures whether the models work on data they were not trained on. | Uses a 75/25 holdout split and reports anomaly F1 and root-cause Top-1 accuracy. |
| Docker | Packages the app consistently so it can run on any machine/server. | `Dockerfile` runs the API in a Python 3.11 container. |
| GitHub Actions CI | Automatically checks code whenever you push to GitHub. | Installs packages, runs tests, then evaluates the model. |

## Example flow

Imagine these logs arrive:

```text
INFO request completed
ERROR packet loss detected
ERROR dns resolution failure
WARN upstream connection reset
INFO health check passed
```

TraceGuard would return something similar to:

```json
{
  "decision": "ANALYZED",
  "anomaly_probability": 0.97,
  "root_cause": "network",
  "root_cause_confidence": 0.91,
  "evidence": [
    "packet loss detected",
    "dns resolution failure",
    "upstream connection reset"
  ]
}
```

If it receives only unclear logs, it will not invent an answer:

```json
{
  "decision": "NEEDS_HUMAN_REVIEW",
  "root_cause": null,
  "reason": "Confidence or evidence support is insufficient; route to an operator."
}
```

## Important files

- [app/data.py](/Users/riddhigoyal/Downloads/airproj/traceguard/app/data.py) — creates and loads safe demo incidents
- [app/ml.py](/Users/riddhigoyal/Downloads/airproj/traceguard/app/ml.py) — ML models, ranking, evidence, and confidence logic
- [app/api.py](/Users/riddhigoyal/Downloads/airproj/traceguard/app/api.py) — API endpoints
- [scripts/train.py](/Users/riddhigoyal/Downloads/airproj/traceguard/scripts/train.py) — trains and saves models
- [scripts/evaluate.py](/Users/riddhigoyal/Downloads/airproj/traceguard/scripts/evaluate.py) — reports model metrics
- [docs/architecture.md](/Users/riddhigoyal/Downloads/airproj/traceguard/docs/architecture.md) — system diagram and safety boundary


It demonstrates that you understand more than LLM APIs:

- traditional ML and measurable baselines;
- anomaly detection and classification;
- reliability and root-cause analysis;
- explainability through cited evidence;
- AI safety via confidence-based abstention;
- API, Docker, testing, CI/CD, and reproducible experiments.

The key portfolio claim is:

> “I built an evidence-grounded AIOps system that detects log anomalies, ranks root causes, and abstains when it cannot support a diagnosis with operational evidence.”

Currently it uses synthetic data as a safe working prototype. next upgrade would be evaluating it on a public LogHub dataset, then adding a small dashboard and a Transformer/DeepLog baseline comparison.
