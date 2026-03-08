# ragtest
[![CI](https://github.com/dbhavery/ragtest/actions/workflows/ci.yml/badge.svg)](https://github.com/dbhavery/ragtest/actions/workflows/ci.yml)

**RAG evaluation suite -- benchmark retrieval accuracy, generation quality, and chunking strategies.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/ragtest.svg)](https://pypi.org/project/ragtest/)

---

## What is ragtest?

ragtest is a pytest-style evaluation suite for Retrieval-Augmented Generation (RAG) pipelines. Define your eval dataset as YAML, point ragtest at your documents, and get retrieval and generation quality scores instantly.

- **Zero external dependencies for core metrics** -- pure Python math, no vector DB or LLM needed.
- **Built-in TF-IDF retriever** -- works out of the box for baseline evaluation.
- **Heuristic generation metrics** -- faithfulness, relevance, and completeness without LLM calls.
- **Three chunking strategies** -- fixed-size, sentence-based, and paragraph-based with comparison tools.
- **Rich terminal output** -- color-coded scores with aggregate summaries.
- **HTML reports** -- shareable evaluation reports.

---

## Quick Start

### Install

```bash
pip install ragtest
```

Or from source:

```bash
git clone https://github.com/dbhavery/ragtest.git
cd ragtest
pip install -e ".[dev]"
```

### Define your eval dataset

Create `eval.yaml`:

```yaml
name: my-knowledge-base
dataset_path: eval.yaml

documents:
  - path: docs/manual.md
    id: manual
  - path: docs/faq.md
    id: faq

questions:
  - question: "How do I reset my password?"
    expected_answer: "Go to Settings > Security > Reset Password"
    source_ids: ["faq"]
  - question: "What are the system requirements?"
    expected_answer: "Python 3.11+, 8GB RAM, NVIDIA GPU recommended"
    source_ids: ["manual"]

retriever:
  type: tfidf
  top_k: 5

chunking:
  strategy: fixed
  chunk_size: 500
  overlap: 50

generator:
  type: none

metrics:
  - recall@5
  - precision@5
  - mrr
  - ndcg@5
  - faithfulness
  - relevance
  - completeness
```

### Run evaluation

```bash
ragtest run eval.yaml --verbose
```

Output:

```
ragtest evaluation: my-knowledge-base

        Configuration
 Key              Value
 Chunking         fixed (size=500, overlap=50)
 Retriever        tfidf (top_k=5)
 Generator        none
 Total chunks     12
 Total questions  2

        Aggregate Metrics
 Metric         Score    Rating
 recall@5       0.8500   GOOD
 precision@5    0.4000   POOR
 mrr            1.0000   GOOD
 faithfulness   0.9200   GOOD
 relevance      0.7500   FAIR
 completeness   0.8800   GOOD
```

### Generate HTML report

```bash
ragtest run eval.yaml --output html --html-path report.html
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `ragtest run <config.yaml>` | Run full evaluation from YAML config |
| `ragtest chunk <document>` | Test chunking strategies on a document |
| `ragtest metrics` | List all available metrics |

### Options

| Flag | Description |
|------|-------------|
| `--verbose / -v` | Show per-question detail |
| `--output terminal/html` | Output format (default: terminal) |
| `--html-path <path>` | Path for HTML report |
| `--strategy fixed/sentence/paragraph/all` | Chunking strategy (for `chunk` command) |
| `--size <int>` | Chunk size in characters (default: 500) |
| `--overlap <int>` | Chunk overlap in characters (default: 50) |

---

## Metrics Reference

### Retrieval Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| `recall@k` | \|retrieved intersect relevant\| / \|relevant\| | Fraction of relevant docs found in top-k |
| `precision@k` | \|retrieved intersect relevant\| / k | Fraction of top-k that are relevant |
| `mrr` | 1 / rank_of_first_relevant | Reciprocal rank of first relevant result |
| `ndcg@k` | DCG@k / IDCG@k | Normalized discounted cumulative gain |

### Generation Metrics

| Metric | Method | Description |
|--------|--------|-------------|
| `faithfulness` | Token overlap (answer vs. context) | Does the answer only use info from context? |
| `relevance` | Token overlap (question vs. answer) | Does the answer address the question? |
| `completeness` | Token overlap (expected vs. actual) | Does it cover all expected points? |
| `answer_length_ratio` | len(answer) / len(expected) | Verbosity comparison |

All generation metrics are heuristic-based (keyword overlap). No LLM calls required.

---

## Chunking Strategies

| Strategy | Method | Best for |
|----------|--------|----------|
| `fixed` | Split by character count with overlap | Uniform chunk sizes |
| `sentence` | Split by sentence boundaries | Preserving sentence coherence |
| `paragraph` | Split by double newlines | Preserving paragraph structure |

Compare strategies on any document:

```bash
ragtest chunk mydoc.txt --strategy all
```

---

## Architecture

```
ragtest/
  __init__.py       Package root
  cli.py            Click CLI entry point
  config.py         YAML config loader (EvalConfig dataclass)
  dataset.py        Eval dataset loader (documents + questions)
  retrieval.py      Pure-Python IR metrics (recall, precision, MRR, NDCG)
  generation.py     Heuristic generation metrics (faithfulness, relevance, completeness)
  chunking.py       Chunking strategies (fixed, sentence, paragraph)
  runner.py         Test runner + built-in TF-IDF retriever
  report.py         Rich terminal + HTML report output
```

### Key design decisions

- **Pure Python core** -- retrieval and generation metrics have zero external dependencies. Only the CLI layer uses `click` and `rich`.
- **Built-in TF-IDF retriever** -- no vector database needed for basic evaluation. Swap in ChromaDB or FAISS for production use.
- **YAML-first** -- datasets and configs are YAML files, easy to version control alongside your docs.
- **Pluggable** -- the `EvalConfig` supports `custom` retriever and generator types for extension.

---

## Development

```bash
git clone https://github.com/dbhavery/ragtest.git
cd ragtest
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT License. Copyright (c) 2026 Donald Havery. See [LICENSE](LICENSE) for details.
