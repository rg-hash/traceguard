from __future__ import annotations

from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from app.retrieval import RetrievedEvidence


class HybridRetriever:
    """
    Combines semantic embeddings with lexical TF-IDF retrieval.

    Semantic retrieval helps with similar meanings.
    Lexical retrieval rewards exact operational terms, error names,
    component names, and repeated log templates.
    """

    def __init__(
        self,
        semantic_weight: float = 0.60,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        if not 0.0 <= semantic_weight <= 1.0:
            raise ValueError("semantic_weight must be between 0 and 1.")

        self.semantic_weight = semantic_weight
        self.lexical_weight = 1.0 - semantic_weight

        self.model = SentenceTransformer(model_name)
        self.vectorizer = TfidfVectorizer(
            lowercase=False,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            norm="l2",
        )

        self.documents: list[dict[str, Any]] = []
        self.semantic_embeddings: np.ndarray | None = None
        self.tfidf_matrix = None

    def build_index(self, documents: list[dict[str, Any]]) -> None:
        """Build both semantic and lexical indexes from training evidence."""
        if not documents:
            raise ValueError("Cannot build a retrieval index from zero documents.")

        self.documents = documents
        texts = [str(document["text"]) for document in documents]

        self.semantic_embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def score_batch(
        self,
        query_texts: list[str],
        batch_size: int = 64,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return semantic and lexical similarity matrices for many queries.

        Row i corresponds to query i.
        Column j corresponds to training evidence document j.
        """
        if self.semantic_embeddings is None or self.tfidf_matrix is None:
            raise RuntimeError("Build the retrieval index before scoring.")

        if not query_texts:
            raise ValueError("query_texts cannot be empty.")

        semantic_queries = self.model.encode(
            query_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=True,
        )

        semantic_scores = semantic_queries @ self.semantic_embeddings.T

        lexical_queries = self.vectorizer.transform(query_texts)
        lexical_scores = (
            self.tfidf_matrix @ lexical_queries.T
        ).T.toarray()

        return semantic_scores, lexical_scores

    def search(self, query_text: str, top_k: int = 3) -> list[RetrievedEvidence]:
        """Return top-k evidence windows ranked by hybrid similarity."""
        if self.semantic_embeddings is None or self.tfidf_matrix is None:
            raise RuntimeError("Build the retrieval index before searching.")

        if not query_text.strip():
            raise ValueError("Query text cannot be empty.")

        semantic_query = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        semantic_scores = self.semantic_embeddings @ semantic_query

        lexical_query = self.vectorizer.transform([query_text])
        lexical_scores = (self.tfidf_matrix @ lexical_query.T).toarray().ravel()

        hybrid_scores = (
            self.semantic_weight * semantic_scores
            + self.lexical_weight * lexical_scores
        )

        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        results = []

        for index in top_indices:
            document = self.documents[int(index)]

            results.append(
                RetrievedEvidence(
                    incident_id=str(document["incident_id"]),
                    label=int(document["label"]),
                    score=float(hybrid_scores[index]),
                    text=str(document["text"]),
                )
            )

        return results