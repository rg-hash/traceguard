from prometheus_client import Counter, Histogram


HDFS_RETRIEVAL_REQUESTS = Counter(
    "traceguard_hdfs_retrieval_requests_total",
    "Number of HDFS evidence-retrieval requests by recommendation.",
    ["recommendation"],
)

HDFS_RETRIEVAL_LATENCY_SECONDS = Histogram(
    "traceguard_hdfs_retrieval_latency_seconds",
    "End-to-end latency of HDFS evidence retrieval.",
    buckets=(0.01, 0.05, 0.10, 0.25, 0.50, 1.0, 2.5, 5.0),
)

HDFS_RETRIEVAL_FAILURES = Counter(
    "traceguard_hdfs_retrieval_failures_total",
    "Number of HDFS evidence-retrieval failures by failure type.",
    ["reason"],
)

HDFS_DECISIONS_PERSISTED = Counter(
    "traceguard_hdfs_decisions_persisted_total",
    "Number of HDFS retrieval decisions successfully stored in PostgreSQL.",
)