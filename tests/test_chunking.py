"""Tests for ragtest.chunking — chunking strategies."""

import pytest

from ragtest.chunking import (
    compare_strategies,
    fixed_size_chunks,
    paragraph_chunks,
    sentence_chunks,
)


class TestFixedSizeChunks:
    def test_basic_chunking(self) -> None:
        text = "a" * 100
        chunks = fixed_size_chunks(text, chunk_size=30, overlap=0)
        assert len(chunks) == 4  # 100/30 = 3.33, ceil to 4
        assert chunks[0].char_count == 30
        assert chunks[-1].char_count == 10

    def test_overlap(self) -> None:
        text = "a" * 100
        chunks = fixed_size_chunks(text, chunk_size=50, overlap=10)
        # Step = 50-10 = 40. Starts: 0,40,80 -> 3 chunks
        assert len(chunks) == 3

    def test_empty_text(self) -> None:
        assert fixed_size_chunks("", chunk_size=10) == []

    def test_invalid_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            fixed_size_chunks("text", chunk_size=0)

    def test_overlap_equals_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="overlap.*must be less than"):
            fixed_size_chunks("text", chunk_size=10, overlap=10)

    def test_source_id_propagated(self) -> None:
        chunks = fixed_size_chunks("hello world", chunk_size=5, overlap=0, source_id="doc1")
        assert all(c.source_id == "doc1" for c in chunks)


class TestSentenceChunks:
    def test_basic_sentences(self) -> None:
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = sentence_chunks(text, sentences_per_chunk=2)
        assert len(chunks) == 2

    def test_single_sentence_per_chunk(self) -> None:
        text = "One. Two. Three."
        chunks = sentence_chunks(text, sentences_per_chunk=1)
        assert len(chunks) == 3

    def test_empty_text(self) -> None:
        assert sentence_chunks("", sentences_per_chunk=3) == []

    def test_invalid_sentences_per_chunk(self) -> None:
        with pytest.raises(ValueError, match="sentences_per_chunk must be positive"):
            sentence_chunks("text", sentences_per_chunk=0)


class TestParagraphChunks:
    def test_basic_paragraphs(self) -> None:
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = paragraph_chunks(text)
        assert len(chunks) == 3
        assert chunks[0].text == "First paragraph."
        assert chunks[1].text == "Second paragraph."

    def test_empty_text(self) -> None:
        assert paragraph_chunks("") == []

    def test_single_paragraph(self) -> None:
        text = "Just one paragraph with no double newlines."
        chunks = paragraph_chunks(text)
        assert len(chunks) == 1


class TestCompareStrategies:
    def test_default_strategies(self) -> None:
        text = "Word. " * 200  # ~1200 chars
        stats = compare_strategies(text)
        assert "fixed_500" in stats
        assert "sentence_5" in stats
        assert "paragraph" in stats
        assert all(s.chunk_count > 0 for s in stats.values())

    def test_custom_strategies(self) -> None:
        text = "Sentence one. Sentence two. Sentence three."
        stats = compare_strategies(text, strategies={
            "tiny_fixed": {"type": "fixed", "chunk_size": 10, "overlap": 0},
        })
        assert "tiny_fixed" in stats
        assert stats["tiny_fixed"].chunk_count > 0
