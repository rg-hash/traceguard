# TraceGuard: Complete Project Journey

## 1. Where the project started

The original idea was to build an **AI system that helps engineers understand system failures from logs**.

Software systems generate many logs:

```text
INFO request completed
WARN retry attempt failed
ERROR database timeout
FATAL service crashed
```

During an incident, engineers must decide:

- Is there actually a problem?
- Which logs matter?
- Is this unusual compared with historical logs?
- Should the system alert an engineer?
- Can the system safely make a recommendation?

The initial project was named **TraceGuard** because it should “guard” engineers from unsafe AI predictions by showing evidence and abstaining when uncertain.

The project already existed at:

```text
/Users/riddhigoyal/Desktop/learnings/airproj/traceguard
```

Its first scope was:

```text
Logs → anomaly prediction → evidence → safe recommendation
```

Not:

```text
Logs → automatic production fix
```

That safety boundary has remained throughout the project.

---

# 2. Initial research phase: HDFS log triage

The first public dataset used was the **HDFS dataset** from LogHub.

HDFS means Hadoop Distributed File System. It produces logs related to large-scale storage and distributed-system behavior.

The initial question was:

> Can an ML model identify whether a group of logs is normal or anomalous?

The initial flow was:

```text
HDFS log window
      ↓
Normalize logs
      ↓
Convert log text into ML features
      ↓
Predict normal or anomaly
      ↓
Retrieve similar historical evidence
      ↓
Return prediction or request human review
```

## Important initial files

| File | What it does |
|---|---|
| `app/data.py` | Loads or generates log-incident data. |
| `app/log_normalization.py` | Removes changing values such as timestamps, node IDs, block IDs, ports, IP addresses, and large numbers. |
| `app/ml.py` | Trains and runs anomaly-detection models. |
| `app/evidence.py` | Finds relevant log lines that support a result. |
| `app/hybrid_retrieval.py` | Combines semantic and lexical log retrieval. |
| `app/risk_control.py` | Controls when the system is allowed to automate a recommendation. |
| `app/api.py` | Exposes the project through FastAPI REST endpoints. |
| `scripts/train_hdfs.py` | Trains HDFS models. |
| `scripts/evaluate_hdfs_hybrid.py` | Evaluates HDFS hybrid evidence retrieval. |
| `data/processed/hdfs_incidents.jsonl` | Processed HDFS incident data. |
| `data/index/hdfs_hybrid_train_documents.jsonl` | Historical HDFS training evidence index. |

## Initial ML approach

The main baseline used:

```text
TF-IDF + Logistic Regression
```

### TF-IDF

TF-IDF converts text into numbers based on important words and phrases.

For example:

```text
database connection timeout
```

gets a different numeric representation than:

```text
request completed successfully
```

### Logistic Regression

It predicts the probability that a log incident is anomalous.

Example:

```json
{
  "anomaly_probability": 0.91
}
```

But TraceGuard did not trust this probability alone.

---

# 3. Why evidence retrieval was added

A basic anomaly model can say:

```text
This is an anomaly: 91% confidence
```

But an engineer will ask:

> Why?

So the project added evidence retrieval.

The system searches historical incidents and returns similar logs.

Example:

```text
Current incident:
database connection timeout

Similar historical incident:
database connection timeout
connection pool exhausted
retry failed
```

This created the first TraceGuard safety idea:

```text
Model prediction
      +
Historical evidence
      +
Confidence threshold
      =
Safe recommendation
```

The system only automates a recommendation when:

1. The model is confident.
2. Historical evidence is similar enough.
3. The top historical evidence items agree.
4. The model and evidence agree.

Otherwise, it returns:

```text
NEEDS_HUMAN_REVIEW
```

This is called **abstention**.

Instead of pretending to know everything, the system says:

> “I do not have enough reliable support for an automated recommendation.”

---

# 4. Why the project moved from HDFS to BGL

After HDFS, the project moved to the **BlueGene/L (BGL)** dataset.

BGL is a public supercomputer log dataset. It is useful because it contains a chronological timeline of logs and anomalies.

The question became more advanced:

> Does the model remain reliable when future logs differ from earlier logs?

This is important because real production systems change over time:

```text
New software release
New infrastructure
New workload
New failure patterns
New log format
```

## Important BGL files

| File | What it does |
|---|---|
| `app/bgl_triage.py` | Production-style BGL triage service. |
| `app/temporal_backtesting.py` | Tests the model over time instead of using random data splits. |
| `app/ood.py` | Detects out-of-distribution inputs. |
| `app/cross_domain.py` | Tests transfer between HDFS and BGL. |
| `app/adaptation.py` | Few-shot support-set selection and adaptation. |
| `app/domain_ensemble.py` | Tests domain-combination/ensemble experiments. |
| `scripts/prepare_bgl.py` | Converts raw BGL logs into usable windows. |
| `scripts/prepare_bgl_locked_holdout.py` | Creates the final future BGL evaluation split. |
| `scripts/evaluate_bgl_locked_holdout.py` | Evaluates the final BGL model. |
| `scripts/evaluate_hdfs_to_bgl_transfer.py` | Tests HDFS-to-BGL transfer. |
| `scripts/evaluate_hdfs_to_bgl_ood_guard.py` | Evaluates OOD safety behavior. |
| `data/processed/bgl_development_windows.jsonl` | BGL development dataset. |
| `data/processed/bgl_locked_temporal_holdout.jsonl` | Future BGL holdout dataset. |

---

# 5. What the BGL research showed

The project found an important negative result:

```text
HDFS-trained model → BGL logs
```

did not transfer safely.

The zero-shot HDFS-to-BGL result was poor:

```text
BGL F1 without abstention: approximately 0.57
Unsafe automated decisions: approximately 61%
```

This was a very important finding.

It means:

> A model trained on one kind of system log should not blindly be trusted on another kind of system log.

The project then added an **OOD guard**.

OOD means Out-of-Distribution.

```text
New log looks unlike training evidence
        ↓
Do not trust normal confidence score
        ↓
Abstain and request human review
```

This gave the project a strong responsible-AI angle.

## BGL final research result

On the BGL temporal locked holdout, the existing triage system achieved approximately:

```text
Classifier F1: 92.67%
Selective automation coverage: 56.83%
Automated-decision accuracy: 98.36%
Unsafe automated-decision rate: 1.64%
```

Meaning:

```text
The system automated only some cases,
but when it automated, it was highly accurate.
```

This is safer than forcing a decision for every incident.

---

# 6. Original product state before the Investigation Agent

At this stage, TraceGuard could do this:

```text
Paste BGL logs
      ↓
Predict normal/anomaly
      ↓
Retrieve similar historical BGL evidence
      ↓
Apply safety checks
      ↓
Return:
LIKELY_NORMAL
LIKELY_ANOMALY
NEEDS_HUMAN_REVIEW
```

It also already had:

- FastAPI backend
- PostgreSQL decision storage
- Prometheus metrics
- MLflow experiment tracking
- Docker setup
- Pytest tests
- Static web dashboard
- BGL triage API endpoint
- Human-review queue

The main API endpoint was:

```text
POST /triage/bgl
```

The main UI was:

```text
http://127.0.0.1:8000/dashboard
```

But there was a product limitation:

> It could say “this is probably abnormal,” but it could not help an engineer debug the issue.

That is where the Investigation Agent idea came from.

---

# 7. The new idea: TraceGuard Investigation Agent

We asked:

> Once TraceGuard detects an anomaly, can it help the engineer investigate safely?

The new product flow became:

```text
Logs
  ↓
Existing HDFS/BGL anomaly triage
  ↓
LIKELY_ANOMALY or NEEDS_HUMAN_REVIEW
  ↓
Investigation Agent
  ↓
Find similar incidents, runbooks, and deployment context
  ↓
Rank likely root-cause categories
  ↓
Suggest evidence-linked debugging checks
  ↓
Engineer reviews and confirms outcome
```

The important rule was:

```text
The agent can investigate.
The agent cannot automatically fix production.
```

It cannot:

```text
restart services
roll back deployments
run shell commands
change configurations
modify databases
delete Kubernetes pods
```

Every investigation result ends with:

```text
ENGINEER_REVIEW_REQUIRED
```

---

# 8. Files added for the Investigation Agent

## A. Knowledge base files

We first created a small, controlled operational knowledge base.

| File | Purpose |
|---|---|
| `data/knowledge/incident_knowledge.json` | Stores historical incidents and approved runbooks. |
| `data/knowledge/deployments.json` | Stores demo deployment metadata for deployment correlation. |

### `incident_knowledge.json`

This includes demo historical incidents such as:

```text
INC-DB-001
Payment API database connection-pool exhaustion
```

```text
INC-NET-002
Checkout API DNS failure
```

```text
INC-DEPLOY-003
Orders API regression after deployment
```

It also includes approved runbooks:

```text
RUNBOOK-DB-01
Database connectivity investigation
```

```text
RUNBOOK-NET-01
DNS and downstream dependency investigation
```

```text
RUNBOOK-DEPLOY-01
Post-deployment regression investigation
```

The agent suggests debugging steps only from these approved runbooks.

---

## B. Investigation Agent implementation

| File | Purpose |
|---|---|
| `app/investigation.py` | Main LangGraph Investigation Agent. |

This file contains the complete agent workflow.

```text
summarize_incident
      ↓
retrieve_evidence
      ↓
correlate_deployments
      ↓
rank_hypotheses
      ↓
create_investigation_plan
      ↓
ENGINEER_REVIEW_REQUIRED
```

### What each LangGraph node does

| Agent node | Purpose |
|---|---|
| `summarize_incident` | Counts errors, identifies the affected service, extracts dominant failures, and records the incident time range. |
| `retrieve_evidence` | Searches historical incidents and approved runbooks. |
| `correlate_deployments` | Finds recent deployments for the affected service. |
| `rank_hypotheses` | Scores possible root causes. |
| `create_investigation_plan` | Retrieves safe debugging checks from approved runbooks. |

---

# 9. Initial hypothesis-ranking logic

The agent initially used transparent scoring:

```text
45% log-pattern match
40% historical incident evidence
15% deployment context
```

Example:

```text
Current logs:
database timeout
connection pool exhausted

Historical evidence:
INC-DB-001

Deployment context:
payment-api deployment occurred recently
```

Possible result:

```json
{
  "cause": "database_connection_pool_exhaustion",
  "confidence": 0.78,
  "evidence_ids": ["INC-DB-001"]
}
```

This is better than an LLM randomly guessing because every hypothesis must connect to evidence.

---

# 10. Why hybrid retrieval was needed

The first Investigation Agent used only TF-IDF retrieval.

It worked for exact language:

```text
DNS resolution failure
```

But it failed for paraphrased language:

```text
resolver returned no address
service discovery could not find endpoint
name lookup failed
```

The original holdout result was:

```text
Known-cause Top-1 accuracy: 66.67%
Unknown-cause abstention: 100%
Unsupported-hypothesis rate: 0%
```

It missed all network cases because their words did not exactly match the runbook wording.

This was a useful failure.

It showed:

> TF-IDF is safe but too literal for real-world log-language variation.

---

# 11. Hybrid semantic + lexical retrieval upgrade

We upgraded `app/investigation.py`.

The new retrieval method combines:

```text
70% semantic similarity
+
30% TF-IDF lexical similarity
=
Hybrid similarity score
```

## New technology

```text
Sentence Transformer:
all-MiniLM-L6-v2
```

The semantic model understands similar meanings.

Example:

```text
“DNS resolution failure”
```

and:

```text
“resolver returned no address”
```

are semantically related even if the exact words differ.

## Updated `app/investigation.py`

The file now:

- Loads the Sentence Transformer model
- Generates embeddings for incidents and runbooks
- Calculates semantic similarity
- Calculates lexical TF-IDF similarity
- Combines both scores
- Returns evidence with all three scores:

```json
{
  "hybrid_similarity": 0.54,
  "semantic_similarity": 0.69,
  "lexical_similarity": 0.18
}
```

The agent also uses:

```text
minimum_hypothesis_evidence = 0.26
```

This means weak evidence does not become a root-cause hypothesis.

---

# 12. Connecting the agent to FastAPI

We updated:

| File | Changes |
|---|---|
| `app/api.py` | Added Investigation Agent API models and endpoints. |

New endpoints:

```text
POST /investigate
POST /investigations/feedback
GET  /investigations/feedback
```

## `POST /investigate`

This receives:

- Incident ID
- Log events
- Existing BGL/HDFS triage recommendation
- Existing anomaly probability
- Existing BGL historical evidence
- Existing safety checks

It returns:

```text
Incident summary
Historical incident/runbook evidence
Deployment context
Ranked hypotheses
Approved debugging checks
Human-review requirement
```

## Why preserve the BGL result?

The Investigation Agent does not replace the anomaly model.

It preserves the original BGL decision:

```text
BGL anomaly probability
BGL classifier prediction
BGL safety checks
BGL retrieved evidence
```

Then it adds debugging evidence.

```text
BGL evidence:
Why was this incident considered anomalous?

Investigation evidence:
Why does the agent recommend these debugging checks?
```

---

# 13. Human feedback and PostgreSQL

We updated:

| File | Changes |
|---|---|
| `app/database.py` | Added engineer-feedback table and functions. |

New PostgreSQL table:

```text
investigation_feedback
```

It stores:

```text
incident ID
triage recommendation
reviewed hypothesis
hypothesis accepted/rejected
confirmed root cause
resolution
usefulness rating
reviewer note
timestamp
```

New database functions:

```python
save_investigation_feedback()
list_investigation_feedback()
```

This creates the feedback loop:

```text
Agent suggests hypothesis
      ↓
Engineer confirms/rejects it
      ↓
Actual resolution is saved
      ↓
Verified information can later become future evidence
```

The system does not retrain automatically from feedback. That would be unsafe without review.

---

# 14. Testing files added

| File | Purpose |
|---|---|
| `tests/test_investigation.py` | Tests Investigation Agent logic and API contracts. |

The tests verify:

- Database logs produce the expected hypothesis.
- Runbook evidence is retrieved.
- `/investigate` preserves BGL triage context.
- Feedback can be stored.
- Feedback records can be retrieved.

Final focused test result:

```text
4 passed
```

---

# 15. Evaluation files added

We added several evaluation scripts.

| File | What it does |
|---|---|
| `scripts/evaluate_investigator.py` | Initial controlled workflow benchmark. |
| `scripts/evaluate_investigator_baselines.py` | Compares keyword-only, retrieval-only, and full agent approaches. |
| `scripts/evaluate_investigator_holdout.py` | Measures known-cause accuracy, hypothesis precision, and safe abstention. |
| `scripts/select_hybrid_configuration.py` | Selects hybrid semantic/lexical settings using development data only. |

## Evaluation datasets

| File | Purpose |
|---|---|
| `data/evaluation/investigation_benchmark.json` | Small initial functional benchmark. |
| `data/evaluation/investigation_development.json` | Development set for tuning hybrid retrieval. |
| `data/evaluation/investigation_holdout.json` | Earlier stress-test set; no longer treated as final because it influenced design. |
| `data/evaluation/investigation_locked_holdout.json` | Final locked evaluation dataset. |

---

# 16. Why development and locked holdout were separated

This is important for research credibility.

Correct process:

```text
Development data
    ↓
Choose semantic weight and safety threshold
    ↓
Freeze configuration
    ↓
Run final locked holdout once
```

The selected development configuration was:

```text
Semantic weight: 0.70
Lexical weight: 0.30
Minimum evidence threshold: 0.26
```

Development-set result:

```text
Top-1 accuracy: 100%
Hypothesis precision: 100%
Unknown abstention: 100%
```

Then the final locked holdout was tested.

---

# 17. Final locked holdout results

Final results:

```text
Known-cause Top-1 accuracy: 88.89%
Known-cause Top-3 accuracy: 88.89%
Hypothesis precision: 100%
Mean hypotheses per known case: 0.89
Unknown-cause abstention rate: 100%
Unknown unsupported-hypothesis rate: 0%
```

Meaning:

```text
9 known incident cases
8 correctly diagnosed
1 safely abstained

3 unknown incident types
3 safely abstained
0 unsafe guesses
```

The one known failure was:

```text
database connection-pool exhaustion
→ no hypothesis returned
```

This was a conservative abstention, not an incorrect diagnosis.

This is a good result because it shows the project values safety:

```text
Unsafe system:
Wrong confident answer

TraceGuard:
No strong evidence → human review
```

---

# 18. Dashboard UI upgrade

The project already had:

| File | Purpose |
|---|---|
| `static/index.html` | Original BGL triage dashboard. |

Originally it could:

```text
Paste BGL logs
      ↓
Call /triage/bgl
      ↓
Show prediction, safety checks, evidence, review queue
```

We extended this file with the Investigation Agent UI.

The dashboard now includes:

```text
BGL triage
      ↓
Investigate triaged incident
      ↓
Show incident summary
      ↓
Show root-cause hypotheses
      ↓
Show historical evidence
      ↓
Show approved runbook checks
      ↓
Engineer feedback form
      ↓
Save feedback to PostgreSQL
```

It also has:

```text
Investigate unclassified logs
```

This is important because:

```text
BGL model = research model trained on supercomputer logs
```

while:

```text
payment-api / checkout-api / orders-api logs
```

are application-oriented demonstration examples.

The UI clearly marks these manually submitted application logs as:

```text
operator_reported_unclassified_logs
```

with:

```text
NEEDS_HUMAN_REVIEW
```

It does not falsely claim that the BGL model was trained on payment-service logs.

---

# 19. Final project architecture

```text
                 ┌──────────────────────────┐
                 │ HDFS / BGL Research Data │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │ Anomaly Triage Layer     │
                 │ TF-IDF + Logistic Reg.   │
                 │ Evidence + Abstention    │
                 └────────────┬─────────────┘
                              │
              LIKELY_ANOMALY / NEEDS_HUMAN_REVIEW
                              │
                              ▼
                 ┌──────────────────────────┐
                 │ LangGraph Investigator   │
                 │                          │
                 │ • Summarize logs         │
                 │ • Retrieve evidence      │
                 │ • Correlate deployment   │
                 │ • Rank hypotheses        │
                 │ • Select runbook checks  │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │ Engineer Review          │
                 │                          │
                 │ • Accept/reject theory   │
                 │ • Confirm root cause     │
                 │ • Record resolution      │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │ PostgreSQL Feedback      │
                 │ Historical Evidence      │
                 └──────────────────────────┘
```

---

# 20. Final project purpose

TraceGuard is not just an anomaly detector.

It is now:

> **An evidence-grounded, safety-aware AI incident investigation copilot that detects suspicious logs, retrieves historical operational evidence, ranks root-cause hypotheses, suggests approved debugging checks, and requires engineer review before any action.**

The strongest part of the project is that it combines:

```text
Traditional ML
+
Log retrieval
+
Semantic embeddings
+
LangGraph agent orchestration
+
Human-in-the-loop safety
+
Temporal evaluation
+
OOD testing
+
Database feedback loop
+
FastAPI backend
+
Dashboard UI
+
Automated tests
+
Research-grade evaluation
```

That makes it a strong AI/ML systems project for a master’s application.