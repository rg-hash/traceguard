from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedEvidence:
    """One historical log window retrieved as supporting evidence."""

    incident_id: str
    label: int
    score: float
    text: str


class SemanticRetriever:
    """
    Finds historically similar log windows using sentence embeddings.

    Important: the index will contain only training data, never validation
    or test data. This prevents evaluation data leakage.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.documents: list[dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None

    def build_index(self, documents: list[dict[str, Any]]) -> None:
        """
        Convert training log windows into normalized semantic vectors.

        Each document must contain:
        - incident_id
        - text
        - label
        """
        if not documents:
            raise ValueError("Cannot build a retrieval index from zero documents.")

        self.documents = documents

        texts = [document["text"] for document in documents]

        self.embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def search(self, query_text: str, top_k: int = 3) -> list[RetrievedEvidence]:
        """
        Return the top-k most semantically similar training incidents.
        """
        if self.embeddings is None:
            raise RuntimeError("Build the retrieval index before searching.")

        if not query_text.strip():
            raise ValueError("Query text cannot be empty.")

        query_embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        similarity_scores = self.embeddings @ query_embedding

        top_indices = np.argsort(similarity_scores)[::-1][:top_k]

        results = []

        for index in top_indices:
            document = self.documents[int(index)]

            results.append(
                RetrievedEvidence(
                    incident_id=str(document["incident_id"]),
                    label=int(document["label"]),
                    score=float(similarity_scores[index]),
                    text=str(document["text"]),
                )
            )

        return results

    def search_batch(
        self,
        query_texts: list[str],
        top_k: int = 3,
        batch_size: int = 64,
    ) -> list[list[RetrievedEvidence]]:
        """
        Retrieve evidence for many queries efficiently.

        This is used for evaluation. It embeds queries in batches instead of
        running the embedding model separately for every log window.
        """
        if self.embeddings is None:
            raise RuntimeError("Build the retrieval index before searching.")

        if not query_texts:
            return []

        query_embeddings = self.model.encode(
            query_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=True,
        )

        all_results = []

        for query_embedding in query_embeddings:
            similarity_scores = self.embeddings @ query_embedding
            top_indices = np.argsort(similarity_scores)[::-1][:top_k]

            matches = []

            for index in top_indices:
                document = self.documents[int(index)]

                matches.append(
                    RetrievedEvidence(
                        incident_id=str(document["incident_id"]),
                        label=int(document["label"]),
                        score=float(similarity_scores[index]),
                        text=str(document["text"]),
                    )
                )

            all_results.append(matches)

        return all_results

    def evidence_summary(
        self,
        query_text: str,
        top_k: int = 3,
        minimum_similarity: float = 0.55,
    ) -> dict[str, Any]:
        """
        Summarize retrieved evidence and decide whether it is strong enough
        to support an automated recommendation.
        """
        matches = self.search(query_text=query_text, top_k=top_k)

        top_score = matches[0].score
        anomaly_votes = sum(match.label == 1 for match in matches)
        anomaly_vote_ratio = anomaly_votes / len(matches)

        has_strong_evidence = top_score >= minimum_similarity

        if not has_strong_evidence:
            recommendation = "NEEDS_HUMAN_REVIEW"
        elif anomaly_votes > len(matches) / 2:
            recommendation = "LIKELY_ANOMALY"
        else:
            recommendation = "LIKELY_NORMAL"

        return {
            "recommendation": recommendation,
            "top_similarity": round(top_score, 4),
            "anomaly_vote_ratio": round(anomaly_vote_ratio, 4),
            "has_strong_evidence": has_strong_evidence,
            "matches": matches,
        }

    