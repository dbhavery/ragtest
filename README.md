# RagTest

RAG evaluation framework that measures retrieval precision, recall, and answer quality -- separating "did we find the right documents?" from "did we generate a good answer?"

## Why I Built This

Most RAG evaluation tools test the pipeline end-to-end: ask a question, check the answer. That tells you the pipeline is broken, but not *where*. If retrieval returns the wrong chunks, no generator can save it. If retrieval is perfect but generation hallucinates, you're debugging the wrong component. I needed metrics that isolate retrieval quality from generation quality.

## What It Does

- **Retrieval metrics** -- recall@k, precision@k, MRR, NDCG@k measure whether the right documents are found
- **Generation metrics** -- faithfulness, relevance, completeness measure answer quality against golden answers
- **Zero external dependencies for core metrics** -- pure Python math, no vector DB or LLM needed for evaluation
- **3 chunking strategies** with comparison tools -- fixed-size, sentence-based, paragraph-based; run all three on the same document to see which works best
- **HTML reports** -- shareable evaluation reports with color-coded score ratings

## Key Technical Decisions

- **Metric-first design over end-to-end testing** -- separating retrieval quality from generation quality pinpoints where the pipeline breaks. A low recall@5 with high faithfulness means your chunking is wrong, not your LLM.
- **YAML golden sets over database** -- version-controllable, diffable, shareable. Golden answer sets live next to the documents they evaluate, reviewable in PRs.
- **Built-in TF-IDF retriever** -- works out of the box for baseline evaluation without requiring ChromaDB or FAISS setup. Swap in your production retriever when ready.
- **Heuristic generation metrics** -- token overlap for faithfulness/relevance/completeness instead of LLM-as-judge. No API costs, deterministic results, runs in milliseconds.

## Quick Start

```bash
pip install ragtest

# Define an eval dataset
cat > eval.yaml << 'EOF'
name: knowledge-base-eval
documents:
  - path: docs/manual.md
    id: manual
questions:
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
metrics:
  - recall@5
  - precision@5
  - faithfulness
  - completeness
EOF

# Run evaluation
ragtest run eval.yaml --verbose

# Compare chunking strategies
ragtest chunk mydoc.txt --strategy all

# Generate HTML report
ragtest run eval.yaml --output html --html-path report.html
```

## Lessons Learned

**RAG evaluation without human baselines is circular.** Automated metrics only become meaningful after you've manually built golden answer sets -- expected answers written by a human who knows the documents. I tried bootstrapping golden sets with LLM-generated answers and the metrics just measured "does model A agree with model B?" which tells you nothing about correctness. The manual effort of writing 20-50 golden answers per document set is unavoidable and is the most valuable part of the evaluation pipeline.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

MIT License. See [LICENSE](LICENSE).
