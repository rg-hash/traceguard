# Architecture and safety boundary

```text
                 +--------------------+
                 | log/metric ingestion|
                 +----------+---------+
                            |
                    feature extraction
                            |
            +---------------+----------------+
            |                                |
     anomaly classifier                root-cause ranker
            |                                |
            +-------------+------------------+
                          |
              evidence retrieval from logs
                          |
          confidence + evidence policy gate
                 /                    \
          ANALYZED              NEEDS_HUMAN_REVIEW
```

No remediation is automatically executed. Future tool calls must require an authenticated human approval, structured action schemas, audit logs, least-privilege access, and rollback paths.
