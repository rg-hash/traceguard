# TraceGuard: Experimental Results

## Research question

Can an evidence-grounded incident-triage system safely automate log-anomaly decisions, abstain when evidence is insufficient, and remain reliable under temporal and cross-domain distribution shift?

## Datasets

| Dataset | Purpose | Protocol |
|---|---|---|
| LogHub HDFS | In-domain evidence retrieval | 2,000 incident-level HDFS block traces; chronological 60/20/20 split |
| LogHub BGL development | Model selection and temporal backtesting | Early 80% of the raw BGL timeline; 10,000 balanced sampled windows |
| LogHub BGL locked holdout | Final evaluation | Last 20% of the raw BGL timeline; 2,000 balanced windows; disjoint and temporally later than all development data |
| OpenStack | Exploratory analysis only | Excluded from supervised evaluation because only 16 of 7,323 windows were anomalous |

The BGL evaluation datasets are balanced by design. Results therefore compare normal and anomalous triage fairly, but do not estimate production anomaly prevalence.

## Methods

TraceGuard combines:

1. Log normalization to remove variable identifiers such as block IDs, IP addresses, timestamps, node IDs, and long numeric values.
2. TF-IDF + Logistic Regression anomaly classification.
3. Top-3 historical evidence retrieval using cosine similarity.
4. Evidence agreement: the top retrieved evidence must be unanimous and agree with the model decision.
5. Abstention: incidents that fail safety conditions are sent to human review.
6. Conservative validation risk control using a Bonferroni-adjusted one-sided Wilson upper confidence bound.

The system never performs automated remediation.

## HDFS hybrid retrieval result

The HDFS hybrid retriever combined semantic embeddings with TF-IDF lexical matching.

| Metric | Final temporal-test result |
|---|---:|
| Precision@1 | 0.9200 |
| Precision@3 | 0.9242 |
| Automated coverage | 83.0% |
| Automated-decision accuracy | 99.4% |
| Unsafe automated-decision rate | 0.6% |
| Retrieval latency | 3.58 ms/query |

## Cross-domain findings

HDFS-to-BGL zero-shot transfer performed poorly:

- BGL zero-shot anomaly F1: 0.5673
- Unsafe automated-decision rate with naive source-calibrated abstention: 61.43%

An OOD similarity guard detected that BGL was unlike HDFS evidence and abstained on all BGL inputs, eliminating unsafe automation but providing no automated coverage.

Few-shot experiments across five support-set seeds showed that naive HDFS+BGL data mixing caused negative transfer for ordinary anomaly classification. A validation-selected probability ensemble was also unstable across seeds.

These negative results are retained as evidence that source-domain knowledge should not be blindly transferred across log domains.

## BGL temporal backtest

Using 600 labelled BGL support incidents, TraceGuard was evaluated across three expanding chronological development folds and five support-selection seeds.

| Fold | F1 | PR-AUC | Coverage | Unsafe rate | Safety violations |
|---|---:|---:|---:|---:|---:|
| Fold 1 | 0.5097 ± 0.1934 | 0.5619 ± 0.0846 | 77.26% ± 4.69% | 1.50% ± 0.38% | 0/5 |
| Fold 2 | 0.8673 ± 0.0139 | 0.9192 ± 0.0316 | 56.82% ± 2.74% | 2.18% ± 1.02% | 0/5 |
| Fold 3 | 0.8815 ± 0.0125 | 0.9382 ± 0.0370 | 43.94% ± 6.75% | 5.06% ± 3.67% | 1/5 |

This backtest showed that BGL logs are temporally non-stationary. Therefore, the locked BGL holdout was reserved for one final frozen-method evaluation.

## Final locked BGL holdout

Final method:

- BGL-only TF-IDF + Logistic Regression
- 600 labelled BGL support incidents
- Five predeclared support-selection seeds: 7, 21, 42, 84, 123
- Top-3 evidence agreement and abstention
- Risk-calibrated policy selected only from development validation data

| Seed | F1 | PR-AUC | Coverage | Automated accuracy | Unsafe rate |
|---:|---:|---:|---:|---:|---:|
| 7 | 0.9385 | 0.9854 | 66.50% | 98.35% | 1.65% |
| 21 | 0.9335 | 0.9806 | 49.35% | 98.78% | 1.22% |
| 42 | 0.9383 | 0.9841 | 56.50% | 99.20% | 0.80% |
| 84 | 0.8746 | 0.9755 | 56.60% | 96.91% | 3.09% |
| 123 | 0.9488 | 0.9883 | 55.20% | 98.55% | 1.45% |
| **Mean ± sample standard deviation** | **0.9267 ± 0.0297** | **0.9828 ± 0.0050** | **56.83% ± 6.17%** | **98.36% ± 0.87%** | **1.64% ± 0.87%** |

Pooled across the five support selections, TraceGuard made 94 unsafe automated decisions out of 5,683 automated decisions: 1.65%.

## Final claim

> On a temporally later, disjoint BGL holdout, TraceGuard’s 600-label evidence-grounded abstention policy automated 56.8% of incident-triage decisions with 98.4% automated-decision accuracy and a 1.6% unsafe-decision rate across five predeclared support selections.

## Limitations

- The BGL evaluation sets are balanced, so operational prevalence performance is not measured.
- HDFS and BGL provide anomaly labels, not reliable root-cause labels; TraceGuard does not claim root-cause classification accuracy.
- Evidence retrieval supports triage decisions but does not establish causality.
- Cross-domain transfer remains difficult; the project reports negative-transfer findings rather than claiming universal generalization.
- The system recommends review or triage only. It does not automatically change infrastructure.

## Reproducibility

```bash
pytest -q tests/test_bgl_locked_holdout.py
pytest -q tests/test_risk_control.py
pytest -q tests/test_temporal_backtesting.py

python scripts/evaluate_bgl_locked_holdout.py --seed 7
python scripts/evaluate_bgl_locked_holdout.py --seed 21
python scripts/evaluate_bgl_locked_holdout.py --seed 42
python scripts/evaluate_bgl_locked_holdout.py --seed 84
python scripts/evaluate_bgl_locked_holdout.py --seed 123