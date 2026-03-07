"""Tests for ragtest.dataset — YAML dataset loading and validation."""

import textwrap
from pathlib import Path

import pytest

from ragtest.dataset import load_dataset


@pytest.fixture()
def tmp_dataset(tmp_path: Path) -> Path:
    """Create a minimal valid dataset YAML in a temp directory."""
    doc_file = tmp_path / "doc.txt"
    doc_file.write_text("This is the document content about password resets.", encoding="utf-8")

    dataset_file = tmp_path / "dataset.yaml"
    dataset_file.write_text(textwrap.dedent("""\
        name: test-eval
        documents:
          - path: doc.txt
            id: doc1
        questions:
          - question: "How do I reset my password?"
            expected_answer: "Go to Settings > Security > Reset Password"
            source_ids: ["doc1"]
    """), encoding="utf-8")
    return dataset_file


class TestLoadDataset:
    def test_valid_dataset(self, tmp_dataset: Path) -> None:
        ds = load_dataset(tmp_dataset)
        assert ds.name == "test-eval"
        assert len(ds.documents) == 1
        assert ds.documents[0].id == "doc1"
        assert "password resets" in ds.documents[0].content
        assert len(ds.questions) == 1
        assert ds.questions[0].question == "How do I reset my password?"

    def test_document_map(self, tmp_dataset: Path) -> None:
        ds = load_dataset(tmp_dataset)
        dmap = ds.document_map
        assert "doc1" in dmap
        assert dmap["doc1"].id == "doc1"

    def test_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path/dataset.yaml")

    def test_missing_name(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text("documents:\n  - path: x\n    id: x\nquestions:\n  - question: q\n    expected_answer: a\n", encoding="utf-8")
        with pytest.raises(ValueError, match="name"):
            load_dataset(f)

    def test_missing_documents(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text("name: test\nquestions:\n  - question: q\n    expected_answer: a\n", encoding="utf-8")
        with pytest.raises(ValueError, match="at least one document"):
            load_dataset(f)

    def test_unknown_source_id(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.txt"
        doc.write_text("content", encoding="utf-8")
        f = tmp_path / "bad.yaml"
        f.write_text(textwrap.dedent("""\
            name: test
            documents:
              - path: doc.txt
                id: doc1
            questions:
              - question: q
                expected_answer: a
                source_ids: ["nonexistent"]
        """), encoding="utf-8")
        with pytest.raises(ValueError, match="unknown source_id"):
            load_dataset(f)
