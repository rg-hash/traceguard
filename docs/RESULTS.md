# TraceGuard: Evidence-Grounded Log Triage Results

## Summary

TraceGuard is an evidence-grounded AIOps prototype for log-based incident triage. Rather than returning only an anomaly label, it retrieves similar historic incidents, cites matching normalized log templates, and abstains when the retrieved evidence is inconsistent.

The central question was:

> Can a log-triage system automate only high-confidence decisions while exposing the historical evidence that supports each recommendation?

## Datasets

- **HDFS**: public Hadoop Distributed File System logs from LogHub.
- **BGL**: public BlueGene/L supercomputer logs from LogHub.
- **OpenStack**: ingested and windowed, but excluded from quantitative benchmarking because it contained only four anomalous VM groups and was too imbalanced for a credible supervised comparison.

Raw logs were converted into structured incident/window JSONL files. Large raw log files were streamed during preprocessing rather than loaded fully into memory.

## Evaluation Design

For semantic and hybrid retrieval experiments, incidents were sorted chronologically and split into:

- First 60%: historic training evidence corpus
- Next 20%: validation data for selecting safety policy
- Final 20%: untouched temporal test data

This avoids the overly optimistic results caused by random splits of temporally related log windows.

## Methods

### BGL semantic retrieval

- Embedding model: `all-MiniLM-L6-v2`
- Evidence corpus: 6,000 historic windows
- Retrieval: top-3 cosine-similar historic windows
- Safety policy: automate only when all top-3 evidence labels agree and similarity is at least `0.90`

### HDFS hybrid retrieval

- Semantic retrieval: sentence embeddings using `all-MiniLM-L6-v2`
- Lexical retrieval: TF-IDF with unigram and bigram features
- Hybrid score: `0.75 × semantic similarity + 0.25 × lexical similarity`
- Log normalization removed block IDs, IP addresses, ports, timestamps, and long numeric identifiers while preserving error templates.
- Safety policy: automate only when all top-3 evidence labels agree.

## Final Temporal-Test Results

| Dataset | Test incidents | Evidence P@1 | Evidence P@3 | Coverage | Automated accuracy | Unsafe decision rate | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| BGL semantic retrieval | 2,000 | 84.85% | 84.95% | 46.10% | 98.26% | 1.74% | 3.92 ms/query |
| HDFS hybrid retrieval | 400 | 92.00% | 92.42% | 83.00% | 99.40% | 0.60% | 3.58 ms/query |

For HDFS, 332 of 400 final temporal-test incidents were automated, with two unsafe confident decisions. For BGL, the safety policy was more conservative under distribution shift and automated 922 of 2,000 incidents.

## Key Findings

1. Random splits were misleadingly optimistic because nearby log windows can share failure bursts and templates.
2. Exact technical vocabulary matters: HDFS hybrid retrieval outperformed pure semantic retrieval.
3. Abstention is essential. The system refuses automation when top evidence conflicts, even if the best retrieved match has a high score.
4. TraceGuard exposes normalized log-template citations, allowing an operator to inspect why a recommendation was made.

## Operational Prototype

The FastAPI endpoint `POST /retrieve/hdfs` returns:

- a triage recommendation;
- top hybrid similarity;
- historical incident IDs and labels;
- cited matching log templates;
- a database audit-record ID.

Recommendations and evidence are persisted in PostgreSQL. Prometheus metrics expose request counts, retrieval latency, failures, and successful persistence. The retrieval index is preloaded during API startup; a measured warm API request completed in approximately 78 ms including PostgreSQL persistence.

## Reproducibility

- `scripts/build_bgl_retrieval_index.py`
- `scripts/calibrate_bgl_retrieval.py`
- `scripts/evaluate_bgl_retrieval.py`
- `scripts/build_hdfs_hybrid_index.py`
- `scripts/calibrate_hdfs_hybrid.py`
- `scripts/evaluate_hdfs_hybrid.py`
- `scripts/log_final_retrieval_runs.py`

MLflow experiment records are stored locally under `artifacts/mlruns`.

## Limitations

This project evaluates binary anomaly-label agreement, not true root-cause accuracy, because the public HDFS and BGL incident records used here do not provide reliable root-cause labels. Results should not be interpreted as automatic remediation capability. Thresholds and safety policies are dataset-specific and must be recalibrated before deployment to a new environment.