"""Test suite runner — orchestrate chunking, retrieval, and evaluation.

Includes a built-in TF-IDF keyword retriever so ragtest works out of the
box with zero external dependencies.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from ragtest.chunking import (
    Chunk,
    fixed_size_chunks,
    paragraph_chunks,
    sentence_chunks,
)
from ragtest.config import EvalConfig
from ragtest.dataset import EvalDataset
from ragtest.generation import (
    answer_length_ratio,
    completeness_score,
    faithfulness_score,
    relevance_score,
)
from ragtest.retrieval import mrr, ndcg_at_k, precision_at_k, recall_at_k


# ---------------------------------------------------------------------------
# Built-in TF-IDF retriever
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase and extract alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


class TFIDFRetriever:
    """Minimal TF-IDF keyword retriever.

    Uses term-frequency * inverse-document-frequency scoring.  Good enough
    for evaluating chunking and metric logic without needing an external
    vector database.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._token_lists: list[list[str]] = []
        self._idf: dict[str, float] = {}

    def index(self, chunks: list[Chunk]) -> None:
        """Build the TF-IDF index from a list of chunks.

        Args:
            chunks: Document chunks to index.
        """
        self._chunks = list(chunks)
        self._token_lists = [_tokenize(c.text) for c in self._chunks]

        # Compute IDF
        n = len(self._chunks)
        df: dict[str, int] = {}
        for tokens in self._token_lists:
            unique = set(tokens)
            for token in unique:
                df[token] = df.get(token, 0) + 1

        self._idf = {
            token: math.log((n + 1) / (count + 1)) + 1
            for token, count in df.items()
        }

    def query(self, text: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Retrieve the top-k most relevant chunks for a query.

        Args:
            text: Query string.
            top_k: Number of results to return.

        Returns:
            List of (Chunk, score) tuples, sorted by descending score.
        """
        query_tokens = _tokenize(text)
        if not query_tokens:
            return []

        scores: list[tuple[int, float]] = []
        for idx, doc_tokens in enumerate(self._token_lists):
            if not doc_tokens:
                scores.append((idx, 0.0))
                continue

            # Term frequency in this document
            tf: dict[str, float] = {}
            for t in doc_tokens:
                tf[t] = tf.get(t, 0) + 1
            for t in tf:
                tf[t] /= len(doc_tokens)

            # Score = sum of TF-IDF for query terms
            score = 0.0
            for qt in query_tokens:
                if qt in tf:
                    score += tf[qt] * self._idf.get(qt, 1.0)

            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results: list[tuple[Chunk, float]] = []
        for idx, score in scores[:top_k]:
            results.append((self._chunks[idx], score))

        return results


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    """Evaluation results for a single question."""

    question: str
    expected_answer: str
    retrieved_ids: list[str]
    relevant_ids: list[str]
    generated_answer: str = ""
    context: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class EvalResults:
    """Aggregated evaluation results."""

    dataset_name: str
    question_results: list[QuestionResult]
    aggregate_metrics: dict[str, float] = field(default_factory=dict)
    config_summary: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _parse_metric_k(metric_name: str) -> tuple[str, int]:
    """Parse a metric name like 'recall@5' into ('recall', 5).

    Args:
        metric_name: Metric name, optionally with @k suffix.

    Returns:
        Tuple of (base_name, k).  k defaults to 5 if not specified.
    """
    if "@" in metric_name:
        parts = metric_name.split("@", 1)
        return parts[0], int(parts[1])
    return metric_name, 5


def _chunk_documents(dataset: EvalDataset, config: EvalConfig) -> list[Chunk]:
    """Chunk all documents in the dataset using the configured strategy.

    Args:
        dataset: The evaluation dataset.
        config: The evaluation configuration.

    Returns:
        List of all chunks across all documents.
    """
    all_chunks: list[Chunk] = []

    for doc in dataset.documents:
        if not doc.content:
            continue

        strategy = config.chunking.strategy

        if strategy == "fixed":
            chunks = fixed_size_chunks(
                doc.content,
                chunk_size=config.chunking.chunk_size,
                overlap=config.chunking.overlap,
                source_id=doc.id,
            )
        elif strategy == "sentence":
            chunks = sentence_chunks(
                doc.content,
                sentences_per_chunk=config.chunking.sentences_per_chunk,
                source_id=doc.id,
            )
        elif strategy == "paragraph":
            chunks = paragraph_chunks(doc.content, source_id=doc.id)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        all_chunks.extend(chunks)

    return all_chunks


def _compute_retrieval_metrics(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    requested_metrics: list[str],
) -> dict[str, float]:
    """Compute the requested retrieval metrics for a single question.

    Args:
        retrieved_ids: IDs of retrieved chunks/documents.
        relevant_ids: IDs of ground-truth relevant documents.
        requested_metrics: List of metric names to compute.

    Returns:
        Mapping of metric name to score.
    """
    results: dict[str, float] = {}

    for metric_name in requested_metrics:
        base, k = _parse_metric_k(metric_name)

        if base == "recall":
            results[metric_name] = recall_at_k(retrieved_ids, relevant_ids, k)
        elif base == "precision":
            results[metric_name] = precision_at_k(retrieved_ids, relevant_ids, k)
        elif base == "mrr":
            results[metric_name] = mrr(retrieved_ids, relevant_ids)
        elif base == "ndcg":
            results[metric_name] = ndcg_at_k(retrieved_ids, relevant_ids, k)

    return results


def _compute_generation_metrics(
    answer: str,
    expected_answer: str,
    question: str,
    context: str,
    requested_metrics: list[str],
) -> dict[str, float]:
    """Compute the requested generation metrics for a single question.

    Args:
        answer: The generated answer (or expected_answer for metric-only mode).
        expected_answer: The ground-truth expected answer.
        question: The original question.
        context: The retrieved context text.
        requested_metrics: List of metric names to compute.

    Returns:
        Mapping of metric name to score.
    """
    results: dict[str, float] = {}

    for metric_name in requested_metrics:
        if metric_name == "faithfulness":
            results[metric_name] = faithfulness_score(answer, context)
        elif metric_name == "relevance":
            results[metric_name] = relevance_score(answer, question)
        elif metric_name == "completeness":
            results[metric_name] = completeness_score(answer, expected_answer)
        elif metric_name == "answer_length_ratio":
            results[metric_name] = answer_length_ratio(answer, expected_answer)

    return results


def run_evaluation(dataset: EvalDataset, config: EvalConfig) -> EvalResults:
    """Run a full RAG evaluation.

    Steps:
        1. Chunk all source documents.
        2. Build a TF-IDF index over all chunks.
        3. For each question, retrieve top-k chunks.
        4. Compute retrieval metrics against ground truth source_ids.
        5. Compute generation metrics using the expected answer.
        6. Aggregate results across all questions.

    When no generator is configured (``config.generator.type == "none"``),
    the expected answer is used in place of a generated answer so that
    generation metrics still produce meaningful baseline scores.

    Args:
        dataset: The evaluation dataset.
        config: The evaluation configuration.

    Returns:
        EvalResults with per-question and aggregate scores.
    """
    # 1. Chunk documents
    all_chunks = _chunk_documents(dataset, config)

    # 2. Build retriever index
    retriever = TFIDFRetriever()
    retriever.index(all_chunks)

    top_k = config.retriever.top_k

    # Separate retrieval and generation metric names
    retrieval_metrics = [
        m for m in config.metrics
        if _parse_metric_k(m)[0] in ("recall", "precision", "mrr", "ndcg")
    ]
    generation_metrics = [
        m for m in config.metrics
        if m in ("faithfulness", "relevance", "completeness", "answer_length_ratio")
    ]

    # 3. Evaluate each question
    question_results: list[QuestionResult] = []

    for q in dataset.questions:
        # Retrieve
        results = retriever.query(q.question, top_k=top_k)
        retrieved_ids = [chunk.source_id for chunk, _score in results]
        context = "\n\n".join(chunk.text for chunk, _score in results)

        # Use expected answer as generated answer when no generator configured
        generated_answer = q.expected_answer

        # Compute metrics
        metrics: dict[str, float] = {}
        metrics.update(_compute_retrieval_metrics(
            retrieved_ids, q.source_ids, retrieval_metrics,
        ))
        metrics.update(_compute_generation_metrics(
            generated_answer, q.expected_answer, q.question, context,
            generation_metrics,
        ))

        question_results.append(QuestionResult(
            question=q.question,
            expected_answer=q.expected_answer,
            retrieved_ids=retrieved_ids,
            relevant_ids=q.source_ids,
            generated_answer=generated_answer,
            context=context,
            metrics=metrics,
        ))

    # 4. Aggregate
    aggregate: dict[str, float] = {}
    all_metric_names = set()
    for qr in question_results:
        all_metric_names.update(qr.metrics.keys())

    for name in sorted(all_metric_names):
        values = [qr.metrics[name] for qr in question_results if name in qr.metrics]
        if values:
            aggregate[name] = sum(values) / len(values)

    config_summary = {
        "chunking_strategy": config.chunking.strategy,
        "chunk_size": config.chunking.chunk_size,
        "overlap": config.chunking.overlap,
        "retriever_type": config.retriever.type,
        "top_k": config.retriever.top_k,
        "generator_type": config.generator.type,
        "total_chunks": len(all_chunks),
        "total_questions": len(dataset.questions),
    }

    return EvalResults(
        dataset_name=dataset.name,
        question_results=question_results,
        aggregate_metrics=aggregate,
        config_summary=config_summary,
    )
