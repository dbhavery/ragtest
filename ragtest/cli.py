"""Click CLI — ragtest command-line interface."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from ragtest.chunking import (
    compare_strategies,
    fixed_size_chunks,
    paragraph_chunks,
    sentence_chunks,
)
from ragtest.config import load_config
from ragtest.dataset import load_dataset
from ragtest.report import generate_html_report, print_summary
from ragtest.runner import run_evaluation


@click.group()
@click.version_option(package_name="ragtest")
def main() -> None:
    """ragtest -- RAG evaluation suite.

    Benchmark retrieval accuracy, generation quality, and chunking strategies
    with a pytest-style test runner for RAG pipelines.
    """


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Show per-question detail.")
@click.option(
    "--output",
    "-o",
    type=click.Choice(["terminal", "html"]),
    default="terminal",
    help="Output format.",
)
@click.option(
    "--html-path",
    type=click.Path(),
    default="ragtest-report.html",
    help="Path for the HTML report (only used with --output html).",
)
def run(config_path: str, verbose: bool, output: str, html_path: str) -> None:
    """Run a full RAG evaluation from a YAML config file.

    CONFIG_PATH is the path to a ragtest YAML config (see examples/).
    """
    console = Console()

    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error loading config:[/red] {exc}")
        raise SystemExit(1) from exc

    # Resolve dataset path relative to config file
    config_dir = Path(config_path).parent
    dataset_path = config_dir / config.dataset_path if config.dataset_path else None

    if dataset_path is None or not dataset_path.exists():
        console.print(f"[red]Dataset not found:[/red] {config.dataset_path}")
        raise SystemExit(1)

    try:
        dataset = load_dataset(dataset_path, base_dir=config_dir)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error loading dataset:[/red] {exc}")
        raise SystemExit(1) from exc

    console.print(f"\n[bold]Running evaluation:[/bold] {dataset.name}")
    console.print(f"  Documents: {len(dataset.documents)}")
    console.print(f"  Questions: {len(dataset.questions)}")
    console.print()

    results = run_evaluation(dataset, config)

    if output == "html":
        report_path = generate_html_report(results, html_path)
        console.print(f"[green]HTML report saved to:[/green] {report_path}")
    else:
        print_summary(results, verbose=verbose)


@main.command()
@click.argument("document", type=click.Path(exists=True))
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["fixed", "sentence", "paragraph", "all"]),
    default="all",
    help="Chunking strategy to test.",
)
@click.option("--size", type=int, default=500, help="Chunk size for fixed strategy.")
@click.option("--overlap", type=int, default=50, help="Overlap for fixed strategy.")
@click.option(
    "--sentences",
    type=int,
    default=5,
    help="Sentences per chunk for sentence strategy.",
)
def chunk(
    document: str, strategy: str, size: int, overlap: int, sentences: int
) -> None:
    """Test chunking strategies on a document.

    DOCUMENT is the path to a text file to chunk.
    """
    console = Console()

    text = Path(document).read_text(encoding="utf-8")

    if strategy == "all":
        stats = compare_strategies(
            text,
            strategies={
                "fixed": {"type": "fixed", "chunk_size": size, "overlap": overlap},
                "sentence": {"type": "sentence", "sentences_per_chunk": sentences},
                "paragraph": {"type": "paragraph"},
            },
        )

        table = Table(title="Chunking Strategy Comparison", border_style="blue")
        table.add_column("Strategy", style="bold")
        table.add_column("Chunks", justify="right")
        table.add_column("Avg Chars", justify="right")
        table.add_column("Min Chars", justify="right")
        table.add_column("Max Chars", justify="right")
        table.add_column("Avg Words", justify="right")
        table.add_column("Total Chars", justify="right")

        for name, s in stats.items():
            table.add_row(
                name,
                str(s.chunk_count),
                f"{s.avg_char_count:.0f}",
                str(s.min_char_count),
                str(s.max_char_count),
                f"{s.avg_word_count:.0f}",
                str(s.total_chars),
            )

        console.print(table)
    else:
        if strategy == "fixed":
            chunks = fixed_size_chunks(text, chunk_size=size, overlap=overlap)
        elif strategy == "sentence":
            chunks = sentence_chunks(text, sentences_per_chunk=sentences)
        else:
            chunks = paragraph_chunks(text)

        console.print(f"\n[bold]{strategy}[/bold] chunking: {len(chunks)} chunks\n")
        for c in chunks:
            preview = c.text[:80].replace("\n", " ")
            console.print(f"  [{c.index}] ({c.char_count} chars) {preview}...")


@main.command()
def metrics() -> None:
    """List all available evaluation metrics."""
    console = Console()

    table = Table(title="Available Metrics", border_style="green")
    table.add_column("Metric", style="bold")
    table.add_column("Type")
    table.add_column("Description")

    retrieval = [
        ("recall@k", "Retrieval", "Fraction of relevant docs found in top-k results"),
        ("precision@k", "Retrieval", "Fraction of top-k results that are relevant"),
        ("mrr", "Retrieval", "Reciprocal rank of first relevant result"),
        ("ndcg@k", "Retrieval", "Normalized discounted cumulative gain at k"),
    ]

    generation = [
        ("faithfulness", "Generation", "Fraction of answer tokens grounded in context"),
        ("relevance", "Generation", "Fraction of question tokens echoed in answer"),
        ("completeness", "Generation", "Fraction of expected answer tokens covered"),
        (
            "answer_length_ratio",
            "Generation",
            "Ratio of answer length to expected length",
        ),
    ]

    for name, mtype, desc in retrieval + generation:
        table.add_row(name, f"[cyan]{mtype}[/cyan]", desc)

    console.print(table)


if __name__ == "__main__":
    main()
