import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.tracking import log_retrieval_run


def main() -> None:
    bgl_run_id = log_retrieval_run(
        experiment_name="TraceGuard Evidence Retrieval",
        run_name="bgl_semantic_temporal_final",
        parameters={
            "dataset": "BGL",
            "retrieval_method": "semantic_embeddings",
            "embedding_model": "all-MiniLM-L6-v2",
            "semantic_weight": 1.00,
            "lexical_weight": 0.00,
            "split_strategy": "chronological_60_20_20",
            "training_evidence_windows": 6000,
            "final_test_windows": 2000,
            "evidence_policy": "unanimous_top_3",
            "minimum_similarity": 0.90,
        },
        metrics={
            "precision_at_1_same_label": 0.8485,
            "precision_at_3_same_label": 0.8495,
            "abstention_coverage": 0.4610,
            "automated_decision_accuracy": 0.9826,
            "unsafe_confident_decision_rate": 0.0174,
            "unsafe_confident_decisions": 16,
            "retrieval_latency_ms": 3.92,
        },
    )

    hdfs_run_id = log_retrieval_run(
        experiment_name="TraceGuard Evidence Retrieval",
        run_name="hdfs_hybrid_temporal_final",
        parameters={
            "dataset": "HDFS",
            "retrieval_method": "hybrid_semantic_tfidf",
            "embedding_model": "all-MiniLM-L6-v2",
            "semantic_weight": 0.75,
            "lexical_weight": 0.25,
            "split_strategy": "chronological_60_20_20",
            "training_evidence_incidents": 1200,
            "final_test_incidents": 400,
            "evidence_policy": "unanimous_top_3",
            "minimum_hybrid_score": 0.05,
        },
        metrics={
            "precision_at_1_same_label": 0.9200,
            "precision_at_3_same_label": 0.9242,
            "abstention_coverage": 0.8300,
            "automated_decision_accuracy": 0.9940,
            "unsafe_confident_decision_rate": 0.0060,
            "unsafe_confident_decisions": 2,
            "hybrid_retrieval_latency_ms": 3.58,
        },
    )

    print(f"Logged BGL final run: {bgl_run_id}")
    print(f"Logged HDFS final run: {hdfs_run_id}")
    print("Tracking directory: artifacts/mlruns")


if __name__ == "__main__":
    main()