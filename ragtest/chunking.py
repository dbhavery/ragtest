"""Chunking strategies — split documents into chunks for retrieval.

All functions are pure Python with no external dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    """A single chunk of text produced by a chunking strategy."""

    text: str
    source_id: str = ""
    index: int = 0

    @property
    def char_count(self) -> int:
        """Number of characters in the chunk."""
        return len(self.text)

    @property
    def word_count(self) -> int:
        """Number of whitespace-delimited words in the chunk."""
        return len(self.text.split())


@dataclass
class ChunkingStats:
    """Summary statistics for a set of chunks."""

    strategy: str
    chunk_count: int
    avg_char_count: float
    min_char_count: int
    max_char_count: int
    avg_word_count: float
    total_chars: int


def fixed_size_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    source_id: str = "",
) -> list[Chunk]:
    """Split text into fixed-size character chunks with optional overlap.

    Args:
        text: The full text to chunk.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.
        source_id: Identifier for the source document.

    Returns:
        List of Chunk objects.

    Raises:
        ValueError: If chunk_size <= 0 or overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
        return []

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    start = 0
    idx = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append(Chunk(text=chunk_text, source_id=source_id, index=idx))
        idx += 1
        start += step

    return chunks


def sentence_chunks(
    text: str,
    sentences_per_chunk: int = 5,
    source_id: str = "",
) -> list[Chunk]:
    """Split text into chunks of N sentences each.

    Sentence boundaries are detected using a simple regex that splits on
    `.`, `!`, or `?` followed by whitespace or end-of-string.

    Args:
        text: The full text to chunk.
        sentences_per_chunk: Number of sentences per chunk.
        source_id: Identifier for the source document.

    Returns:
        List of Chunk objects.

    Raises:
        ValueError: If sentences_per_chunk <= 0.
    """
    if sentences_per_chunk <= 0:
        raise ValueError(
            f"sentences_per_chunk must be positive, got {sentences_per_chunk}"
        )

    if not text.strip():
        return []

    # Split on sentence-ending punctuation followed by whitespace or EOS.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    # Filter out empty strings
    sentences = [s for s in sentences if s.strip()]

    if not sentences:
        return []

    chunks: list[Chunk] = []
    for i in range(0, len(sentences), sentences_per_chunk):
        batch = sentences[i : i + sentences_per_chunk]
        chunk_text = " ".join(batch)
        chunks.append(Chunk(text=chunk_text, source_id=source_id, index=len(chunks)))

    return chunks


def paragraph_chunks(
    text: str,
    source_id: str = "",
) -> list[Chunk]:
    """Split text into chunks by paragraphs (double newlines).

    Args:
        text: The full text to chunk.
        source_id: Identifier for the source document.

    Returns:
        List of Chunk objects.
    """
    if not text.strip():
        return []

    paragraphs = re.split(r"\n\s*\n", text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    return [
        Chunk(text=p, source_id=source_id, index=i)
        for i, p in enumerate(paragraphs)
    ]


def _compute_stats(strategy_name: str, chunks: list[Chunk]) -> ChunkingStats:
    """Compute summary statistics for a list of chunks.

    Args:
        strategy_name: Name of the chunking strategy.
        chunks: List of chunks to summarize.

    Returns:
        A ChunkingStats instance.
    """
    if not chunks:
        return ChunkingStats(
            strategy=strategy_name,
            chunk_count=0,
            avg_char_count=0.0,
            min_char_count=0,
            max_char_count=0,
            avg_word_count=0.0,
            total_chars=0,
        )

    char_counts = [c.char_count for c in chunks]
    word_counts = [c.word_count for c in chunks]

    return ChunkingStats(
        strategy=strategy_name,
        chunk_count=len(chunks),
        avg_char_count=sum(char_counts) / len(char_counts),
        min_char_count=min(char_counts),
        max_char_count=max(char_counts),
        avg_word_count=sum(word_counts) / len(word_counts),
        total_chars=sum(char_counts),
    )


def compare_strategies(
    text: str,
    strategies: dict[str, dict[str, int | str]] | None = None,
    source_id: str = "",
) -> dict[str, ChunkingStats]:
    """Run multiple chunking strategies and compare their statistics.

    Args:
        text: The text to chunk.
        strategies: Mapping of strategy names to their parameters.  If None,
            uses sensible defaults for all three strategies.
        source_id: Identifier for the source document.

    Returns:
        Mapping of strategy name to ChunkingStats.

    Example::

        stats = compare_strategies(
            text,
            strategies={
                "fixed_500": {"type": "fixed", "chunk_size": 500, "overlap": 50},
                "sentence_3": {"type": "sentence", "sentences_per_chunk": 3},
                "paragraph": {"type": "paragraph"},
            },
        )
    """
    if strategies is None:
        strategies = {
            "fixed_500": {"type": "fixed", "chunk_size": 500, "overlap": 50},
            "sentence_5": {"type": "sentence", "sentences_per_chunk": 5},
            "paragraph": {"type": "paragraph"},
        }

    results: dict[str, ChunkingStats] = {}

    for name, params in strategies.items():
        strategy_type = str(params.get("type", "fixed"))

        if strategy_type == "fixed":
            chunks = fixed_size_chunks(
                text,
                chunk_size=int(params.get("chunk_size", 500)),
                overlap=int(params.get("overlap", 50)),
                source_id=source_id,
            )
        elif strategy_type == "sentence":
            chunks = sentence_chunks(
                text,
                sentences_per_chunk=int(params.get("sentences_per_chunk", 5)),
                source_id=source_id,
            )
        elif strategy_type == "paragraph":
            chunks = paragraph_chunks(text, source_id=source_id)
        else:
            raise ValueError(f"Unknown chunking strategy type: {strategy_type}")

        results[name] = _compute_stats(name, chunks)

    return results
