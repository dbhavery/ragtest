"""Tests for ragtest.runner — TF-IDF retriever and evaluation runner."""

import textwrap
from pathlib import Path

import pytest

from ragtest.config import ChunkingConfig, EvalConfig, GeneratorConfig, RetrieverConfig
from ragtest.dataset import Document, EvalDataset, Question, load_dataset
from ragtest.runner import TFIDFRetriever, run_evaluation
from ragtest.chunking import Chunk


class TestTFIDFRetriever:
    def test_basic_retrieval(self) -> None:
        retriever = TFIDFRetriever()
        chunks = [
            Chunk(text="How to reset your password in the settings menu.", source_id="faq"),
            Chunk(text="System requirements include 8GB RAM and Python 3.11.", source_id="manual"),
            Chunk(text="Contact support for billing questions.", source_id="support"),
        ]
        retriever.index(chunks)
        results = retriever.query("reset password", top_k=2)
        assert len(results) == 2
        # The FAQ chunk should rank first
        assert results[0][0].source_id == "faq"

    def test_empty_query(self) -> None:
        retriever = TFIDFRetriever()
        retriever.index([Chunk(text="some text")])
        assert retriever.query("", top_k=5) == []

    def test_top_k_limit(self) -> None:
        retriever = TFIDFRetriever()
        chunks = [Chunk(text=f"Document {i}") for i in range(10)]
        retriever.index(chunks)
        results = retriever.query("document", top_k=3)
        assert len(results) == 3


class TestRunEvaluation:
    def test_end_to_end(self) -> None:
        """Run evaluation with in-memory dataset — no file I/O."""
        dataset = EvalDataset(
            name="test",
            documents=[
                Document(
                    id="faq",
                    path="faq.txt",
                    content="To reset your password, go to Settings > Security > Reset Password. "
                            "This will send a verification email to your registered address.",
                ),
                Document(
                    id="manual",
                    path="manual.txt",
                    content="System requirements: Python 3.11 or higher, 8GB RAM minimum, "
                            "NVIDIA GPU recommended for best performance.",
                ),
            ],
            questions=[
                Question(
                    question="How do I reset my password?",
                    expected_answer="Go to Settings > Security > Reset Password",
                    source_ids=["faq"],
                ),
                Question(
                    question="What are the system requirements?",
                    expected_answer="Python 3.11+, 8GB RAM, NVIDIA GPU recommended",
                    source_ids=["manual"],
                ),
            ],
        )

        config = EvalConfig(
            retriever=RetrieverConfig(type="tfidf", top_k=3),
            chunking=ChunkingConfig(strategy="paragraph"),
            metrics=["recall@3", "precision@3", "mrr", "faithfulness", "relevance", "completeness"],
        )

        results = run_evaluation(dataset, config)

        assert results.dataset_name == "test"
        assert len(results.question_results) == 2
        assert len(results.aggregate_metrics) > 0

        # Retrieval should find the right documents
        for qr in results.question_results:
            assert len(qr.metrics) > 0

        # Aggregate metrics should be between 0 and 1 for score-based metrics
        for name, score in results.aggregate_metrics.items():
            if name != "answer_length_ratio":
                assert 0.0 <= score <= 1.0, f"{name} = {score} out of range"

    def test_from_yaml(self, tmp_path: Path) -> None:
        """Run evaluation from a YAML dataset file."""
        doc = tmp_path / "doc.txt"
        doc.write_text(
            "To reset your password, navigate to Settings, then Security, "
            "then click Reset Password. A confirmation email will be sent.",
            encoding="utf-8",
        )

        dataset_file = tmp_path / "dataset.yaml"
        dataset_file.write_text(textwrap.dedent("""\
            name: yaml-test
            documents:
              - path: doc.txt
                id: doc1
            questions:
              - question: "How do I reset my password?"
                expected_answer: "Go to Settings > Security > Reset Password"
                source_ids: ["doc1"]
        """), encoding="utf-8")

        dataset = load_dataset(dataset_file)
        config = EvalConfig(
            retriever=RetrieverConfig(top_k=3),
            chunking=ChunkingConfig(strategy="fixed", chunk_size=200, overlap=20),
            metrics=["recall@3", "mrr", "completeness"],
        )

        results = run_evaluation(dataset, config)
        assert results.dataset_name == "yaml-test"
        assert len(results.question_results) == 1
        assert "completeness" in results.aggregate_metrics
