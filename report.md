# TraceGuard Progress Report

## 1. Project objective

TraceGuard is an evidence-grounded AIOps project for detecting abnormal behaviour in system logs and safely triaging incidents.

Target pipeline:

```text
Logs
  ↓
Group related events into incidents
  ↓
Detect Normal vs Anomaly
  ↓
Identify root cause when labels/evidence exist
  ↓
Retrieve supporting log evidence
  ↓
Return diagnosis or NEEDS_HUMAN_REVIEW
```

The project is designed to show AI/ML depth, reliable evaluation, and production-oriented engineering—not just an LLM chatbot.

---

## 2. Initial project implementation

Built the initial TraceGuard structure:

```text
app/
  data.py       → synthetic incident generation
  ml.py         → anomaly model, root-cause model, evidence retrieval
  api.py        → FastAPI endpoints

scripts/
  generate_demo_data.py
  train.py
  evaluate.py

tests/
  test_ml.py
  test_api.py

docs/
  architecture.md
  data-card.md
  report.md
```

Supporting engineering pieces were also added:

- FastAPI API design
- Docker and Docker Compose
- GitHub Actions CI workflow
- pytest tests
- data-card and architecture documentation
- confidence-based human-review safety decision

---

## 3. Synthetic-data prototype

The initial dataset was generated safely inside the repository.

It contained:

```text
240 incidents
120 normal incidents
120 anomalous incidents
```

Anomalous incidents were assigned one of three synthetic root causes:

```text
database
network
application
```

Example network signatures:

```text
packet loss detected
dns resolution failure
upstream connection reset
```

Example database signatures:

```text
db connection pool exhausted
postgres timeout
query latency elevated
```

This synthetic dataset was useful for validating the complete architecture, but it was not sufficient for an advanced ML portfolio claim.

---

## 4. Public real-data integration: LogHub HDFS

You added the public LogHub HDFS v1 dataset. LogHub provides publicly accessible datasets for AI-driven log analytics, including HDFS, BGL, OpenStack, and Android logs. [LogHub](https://github.com/logpai/loghub)

Files used:

```text
data/raw/HDFS_v1/HDFS.log
data/raw/HDFS_v1/preprocessed/anomaly_label.csv
data/raw/HDFS_v1/preprocessed/Event_traces.csv
```

Dataset observations:

```text
Raw HDFS.log size: approximately 1.5 GB
Label file rows: 575,062 including header
```

The raw log lines contain HDFS block IDs:

```text
blk_-1608999687919862906
```

The core preprocessing insight was:

```text
Many log lines with the same BlockId
        ↓
One HDFS block / incident
        ↓
Normal or Anomaly label from anomaly_label.csv
```

You created a balanced real-data subset:

```text
1,000 normal HDFS blocks
1,000 anomalous HDFS blocks
Total: 2,000 incidents
```

---

## 5. HDFS Model 1: TF-IDF + Logistic Regression

### Method

```text
All logs from one BlockId
        ↓
Join messages into text
        ↓
TF-IDF unigram + bigram features
        ↓
Logistic Regression
        ↓
Anomaly probability
```

The model was trained on a stratified subset and evaluated on held-out HDFS blocks.

### Baseline result

At threshold `0.5`:

```text
Anomaly F1: 0.871
PR-AUC:      0.958
```

### Threshold analysis

You tested operating thresholds:

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.3 | 0.694 | 0.972 | 0.810 |
| 0.4 | 0.815 | 0.932 | 0.869 |
| 0.5 | 0.940 | 0.812 | 0.871 |
| 0.6 | 0.987 | 0.620 | 0.762 |
| 0.7 | 0.992 | 0.488 | 0.654 |

### Decision

You selected `0.4` as a practical incident-alerting threshold.

Reason:

```text
Threshold 0.4
→ catches 93.2% of anomalies
→ retains 81.5% anomaly precision
→ has near-best F1
```

This demonstrates an important production ML skill: selecting thresholds based on operational cost, rather than automatically accepting `0.5`.

---

## 6. HDFS Model 2: TF-IDF + Random Forest

### Method

```text
Grouped HDFS log text
        ↓
TF-IDF features
        ↓
Random Forest classifier
        ↓
Normal vs Anomaly
```

### Result

```text
Anomaly precision: 1.000
Anomaly recall:    0.996
Anomaly F1:        0.998
PR-AUC:            1.000
```

### Feature audit

You checked the most influential features.

Meaningful high-importance features included:

```text
trying
in volumemap
blockinfo not
error
unexpected
packetresponder for
delete block
addstoredblock request
not found
warn
```

This was a useful leakage check. The model’s important features were mostly HDFS error/failure patterns, not unique block IDs or explicit labels.

### Important limitation

This score is exceptionally high. It is valid for the sampled random HDFS holdout split, but it does not prove production generalisation because:

- training and test blocks can share highly similar event templates;
- HDFS failures may have recognisable signatures;
- the benchmark uses only one system/dataset.

---

## 7. HDFS Model 3: Isolation Forest

### Purpose

Isolation Forest is an unsupervised approach.

```text
Logistic Regression / Random Forest
→ use Normal and Anomaly labels during training

Isolation Forest
→ trains only on Normal logs
→ flags unfamiliar behaviour as anomalous
```

### Evaluation design

This experiment used a better evaluation protocol:

```text
Training set
Validation set → threshold selection
Untouched test set → final metric
```

### Result

```text
Anomaly precision: 0.508
Anomaly recall:    0.920
Anomaly F1:        0.655
PR-AUC:            0.685
```

### Interpretation

Isolation Forest caught most anomalies:

```text
184 of 200 anomalies detected
```

But it also generated many false alerts:

```text
Normal recall: 0.110
```

Conclusion:

> The unsupervised model had high sensitivity but a large false-alert burden. It is useful when labelled anomaly data is unavailable, but supervised models performed much better on this HDFS benchmark.

---

## 8. HDFS Model 4: Sequence-aware Bidirectional LSTM

### Purpose

The earlier models treat incidents primarily as text/features.

The LSTM model uses the order of event templates:

```text
E5 → E22 → E11 → E9 → E26
```

This is stronger because event order can matter in operational failures.

### Dataset

Used:

```text
data/raw/HDFS_v1/preprocessed/Event_traces.csv
```

The dataset supplies:

```text
BlockId
Success / Fail label
Ordered event-template sequence
```

### Implementation details

The model:

```text
Streams Event_traces.csv row by row
        ↓
Uses reservoir sampling
        ↓
Selects 5,000 Success + 5,000 Fail sequences
        ↓
Uses 60% train / 20% validation / 20% untouched test
        ↓
Embeds event-template IDs
        ↓
Bidirectional LSTM learns sequence behaviour
```

### Result

```text
Test incidents: 2,000

Anomaly precision: 1.000
Anomaly recall:    0.997
Anomaly F1:        0.998
PR-AUC:            1.000
Selected threshold: 0.91
```

### Interpretation

The HDFS event-template sequences strongly separate Success and Fail examples under a random holdout split.

Important caveat:

- The LSTM used 10,000 total samples.
- Earlier text baselines used 2,000 total samples.
- Therefore, the models are not yet a fully fair head-to-head comparison.

---

## 9. Current HDFS comparison

| Model | Learning type | Dataset setup | F1 | PR-AUC |
|---|---|---|---:|---:|
| Logistic Regression | Supervised | 2,000-block text subset | 0.869 | 0.958 |
| Random Forest | Supervised | 2,000-block text subset | 0.998 | 1.000 |
| Isolation Forest | Unsupervised, normal-only | 2,000-block text subset | 0.655 | 0.685 |
| Bidirectional LSTM | Supervised sequence model | 10,000 event traces | 0.998 | 1.000 |

---

## 10. OpenStack cross-domain attempt

You added public OpenStack files:

```text
data/raw/OpenStack/openstack_normal1.log
data/raw/OpenStack/openstack_normal2.log
data/raw/OpenStack/openstack_abnormal.log
data/raw/OpenStack/anomaly_labels.txt
```

OpenStack labels are based on injected anomaly VM instances.

### Initial preprocessing result

Grouping all logs by VM instance created only:

```text
8 incidents total
```

A supervised evaluation produced a test set with only:

```text
1 normal incident
1 anomalous incident
```

The resulting PR-AUC of `1.000` was not meaningful because the test set had only two samples.

Correct decision: do not report that score.

### Windowed OpenStack preprocessing

You then changed the unit of analysis:

```text
One VM instance
        ↓
Overlapping log windows
        ↓
One window = one sample
```

Using:

```text
Window size: 10 events
Stride: 5 events
```

produced:

```text
7,323 total OpenStack windows
7,307 normal windows
16 anomaly windows
1,874 VM instances
1,870 normal VMs
4 anomalous VMs
```

### OpenStack conclusion

OpenStack is currently not suitable as a strong independent supervised benchmark because it has only:

```text
4 independently labelled anomaly VM instances
```

Overlapping windows from the same VM do not create new independent anomaly examples.

Correct claim:

> I adapted the TraceGuard ingestion and incident-windowing pipeline from HDFS to OpenStack cloud logs. Because OpenStack contained only four independently labelled anomaly VM sources, I did not report window-level metrics as robust generalisation evidence.

This is scientifically honest and strengthens the project.

---

## 11. What has been demonstrated

You have demonstrated:

1. Public log ingestion and preprocessing.
2. Memory-safe processing of large log datasets.
3. Grouping raw logs into incident-level ML samples.
4. Supervised anomaly classification.
5. Unsupervised anomaly detection.
6. Sequence-aware deep learning for logs.
7. Threshold selection and alerting trade-offs.
8. Feature-importance inspection and leakage awareness.
9. Validation/test split reasoning.
10. Cross-domain preprocessing challenges and dataset-quality assessment.

---

## 12. What has not yet been demonstrated

You should not claim these yet:

- real HDFS root-cause prediction;
- robust cross-system generalisation;
- production readiness;
- semantic evidence retrieval;
- hybrid retrieval or reranking;
- calibrated probability estimates;
- abstention coverage on real logs;
- PostgreSQL/MLflow/Prometheus integration;
- live API ingestion or queue-based processing.

The real HDFS dataset provides anomaly labels, not database/network/application root-cause labels.

---

## 13. Accurate current project claim

> Developed TraceGuard, an AIOps log anomaly-detection prototype. Built reproducible preprocessing pipelines for public LogHub HDFS and OpenStack logs; evaluated supervised, unsupervised, and sequence-aware methods on HDFS. A TF-IDF Logistic Regression baseline achieved 0.958 PR-AUC, while a Bidirectional LSTM achieved 0.998 anomaly F1 on a 10,000-sequence random holdout benchmark. Performed threshold analysis to balance missed incidents against false alerts and documented dataset limitations to avoid overstated cross-domain claims.

---

## 14. Remaining roadmap

```text
1. Make HDFS model comparison fully fair:
   same sample size, same split, same metrics

2. Add BGL:
   second quantitative public dataset

3. Evaluate distribution shift:
   HDFS-trained method versus BGL/OpenStack methodology

4. Add semantic evidence retrieval:
   embeddings, vector database, hybrid retrieval, reranking

5. Add safety evaluation:
   calibration, abstention coverage, unsafe confident-answer rate

6. Add production components:
   FastAPI ingestion, PostgreSQL, MLflow, Prometheus, Docker, CI/CD

7. Write final research report:
   methodology, results, limitations, ablations, and failure analysis
```
