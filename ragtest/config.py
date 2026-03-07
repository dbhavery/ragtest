"""Eval suite configuration — load and validate YAML-based eval configs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RetrieverConfig:
    """Configuration for the retrieval backend."""

    type: str = "tfidf"  # "tfidf" (built-in), "chromadb", or "custom"
    top_k: int = 5
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratorConfig:
    """Configuration for the answer generation backend."""

    type: str = "none"  # "none", "ollama", "openai", or "custom"
    model: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkingConfig:
    """Configuration for document chunking strategy."""

    strategy: str = "fixed"  # "fixed", "sentence", "paragraph"
    chunk_size: int = 500
    overlap: int = 50
    sentences_per_chunk: int = 5


@dataclass
class EvalConfig:
    """Top-level evaluation configuration."""

    dataset_path: str = ""
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    metrics: list[str] = field(default_factory=lambda: [
        "recall@5",
        "precision@5",
        "mrr",
        "ndcg@5",
        "faithfulness",
        "relevance",
        "completeness",
    ])


def load_config(path: str | Path) -> EvalConfig:
    """Load an EvalConfig from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A fully populated EvalConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file is malformed.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(raw).__name__}")

    retriever_raw = raw.get("retriever", {})
    generator_raw = raw.get("generator", {})
    chunking_raw = raw.get("chunking", {})

    retriever = RetrieverConfig(
        type=retriever_raw.get("type", "tfidf"),
        top_k=retriever_raw.get("top_k", 5),
        params=retriever_raw.get("params", {}),
    )

    generator = GeneratorConfig(
        type=generator_raw.get("type", "none"),
        model=generator_raw.get("model", ""),
        params=generator_raw.get("params", {}),
    )

    chunking = ChunkingConfig(
        strategy=chunking_raw.get("strategy", "fixed"),
        chunk_size=chunking_raw.get("chunk_size", 500),
        overlap=chunking_raw.get("overlap", 50),
        sentences_per_chunk=chunking_raw.get("sentences_per_chunk", 5),
    )

    metrics = raw.get("metrics", [
        "recall@5",
        "precision@5",
        "mrr",
        "ndcg@5",
        "faithfulness",
        "relevance",
        "completeness",
    ])

    return EvalConfig(
        dataset_path=raw.get("dataset_path", ""),
        retriever=retriever,
        generator=generator,
        chunking=chunking,
        metrics=metrics,
    )
