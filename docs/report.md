# TraceGuard experiment report

## Research question

Does evidence-grounded abstention reduce unsupported root-cause claims while retaining useful incident-analysis coverage?

## Protocol

- Use a time/order-preserving 75/25 train-test split.
- Compare TF-IDF + logistic regression with a sequence-model extension.
- Keep the test set untouched while choosing the confidence threshold on validation data.
- Report confidence intervals or repeat runs for stochastic models.

## Results table

| Model | Anomaly F1 | RCA top-1 | Evidence recall@3 | Coverage | Unsafe confident answers |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF baseline | TBD | TBD | TBD | TBD | TBD |
| Sequence-model extension | TBD | TBD | TBD | TBD | TBD |

## Limitations

The starter data contains designed causal signatures and is therefore easier than operational telemetry. Results must not be presented as real telecom production performance. Validate any extension on a public dataset such as LogHub, document dataset licensing, and analyze failures by service, severity, and unseen template.
