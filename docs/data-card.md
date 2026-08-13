# Demo incident dataset card

## Motivation

The included data lets contributors develop and test the complete pipeline without accessing customer, employee, or operational telemetry.

## Composition

`scripts/generate_demo_data.py` creates 240 JSONL incident windows. Each contains eight timestamped log events, service, severity, anomaly label, and a root-cause label. Half are normal; anomalous examples cycle across database, network, and application root causes.

## Collection and preprocessing

No records are collected from people or systems. The generator inserts causal templates plus normal-event distractors. The baseline joins each window's messages and applies TF-IDF.

## Uses and non-uses

Use it for pipeline validation, code review, UI demos, and baseline-method development. Do **not** use reported results as evidence of real-world fault-detection performance, nor train a production model from this data.

## Public-data extension

When adapting to LogHub, preserve source licensing and citations; hash/redact identifiers, use temporal splits, document parsing rules, and assess whether labels are complete. Never commit confidential logs, hostnames, IP addresses, tokens, customer data, or incident tickets.

## Risks

The templates are intentionally separable, so models can overfit to keywords. Mitigate this with unseen-template tests, noise injection, cross-dataset evaluation, calibration checks, and mandatory human review when evidence is incomplete.
