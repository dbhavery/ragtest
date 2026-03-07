"""Eval dataset loader — parse and validate question-answer-source triples."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Document:
    """A source document in the eval dataset."""

    id: str
    path: str
    content: str = ""


@dataclass
class Question:
    """A single evaluation question with ground truth."""

    question: str
    expected_answer: str
    source_ids: list[str] = field(default_factory=list)


@dataclass
class EvalDataset:
    """A complete evaluation dataset with documents and questions."""

    name: str
    documents: list[Document]
    questions: list[Question]

    @property
    def document_map(self) -> dict[str, Document]:
        """Return a mapping of document ID to Document."""
        return {doc.id: doc for doc in self.documents}


def load_dataset(path: str | Path, base_dir: str | Path | None = None) -> EvalDataset:
    """Load an evaluation dataset from a YAML file.

    The YAML file should have this structure::

        name: my-eval
        documents:
          - path: docs/manual.md
            id: manual
        questions:
          - question: "How do I reset my password?"
            expected_answer: "Go to Settings > Security > Reset Password"
            source_ids: ["manual"]

    Document paths are resolved relative to *base_dir* (which defaults to the
    directory containing the YAML file).  If a document file exists on disk its
    content is read automatically; otherwise ``content`` is left empty.

    Args:
        path: Path to the YAML dataset file.
        base_dir: Base directory for resolving document paths.  Defaults to
            the parent directory of *path*.

    Returns:
        A validated EvalDataset.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If required fields are missing or malformed.
    """
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    if base_dir is None:
        base_dir = dataset_path.parent

    base_dir = Path(base_dir)

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Dataset file must contain a YAML mapping, got {type(raw).__name__}")

    name = raw.get("name", "")
    if not name:
        raise ValueError("Dataset must have a 'name' field")

    # --- documents -----------------------------------------------------------
    raw_docs: list[dict[str, Any]] = raw.get("documents", [])
    if not raw_docs:
        raise ValueError("Dataset must have at least one document")

    documents: list[Document] = []
    for i, doc_raw in enumerate(raw_docs):
        if "id" not in doc_raw:
            raise ValueError(f"Document at index {i} is missing 'id'")
        if "path" not in doc_raw:
            raise ValueError(f"Document at index {i} is missing 'path'")

        doc_file = base_dir / doc_raw["path"]
        content = ""
        if doc_file.exists():
            content = doc_file.read_text(encoding="utf-8")

        documents.append(Document(
            id=doc_raw["id"],
            path=doc_raw["path"],
            content=content,
        ))

    # --- questions -----------------------------------------------------------
    raw_questions: list[dict[str, Any]] = raw.get("questions", [])
    if not raw_questions:
        raise ValueError("Dataset must have at least one question")

    questions: list[Question] = []
    doc_ids = {d.id for d in documents}
    for i, q_raw in enumerate(raw_questions):
        if "question" not in q_raw:
            raise ValueError(f"Question at index {i} is missing 'question'")
        if "expected_answer" not in q_raw:
            raise ValueError(f"Question at index {i} is missing 'expected_answer'")

        source_ids = q_raw.get("source_ids", [])
        for sid in source_ids:
            if sid not in doc_ids:
                raise ValueError(
                    f"Question at index {i} references unknown source_id '{sid}'. "
                    f"Known IDs: {sorted(doc_ids)}"
                )

        questions.append(Question(
            question=q_raw["question"],
            expected_answer=q_raw["expected_answer"],
            source_ids=source_ids,
        ))

    return EvalDataset(name=name, documents=documents, questions=questions)
