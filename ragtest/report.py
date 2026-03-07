"""Report output — rich terminal tables and optional HTML reports."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from ragtest.runner import EvalResults


def _score_color(score: float) -> str:
    """Return a Rich color name based on score thresholds.

    Args:
        score: Metric score in [0.0, 1.0].

    Returns:
        Color string: "green" (>0.8), "yellow" (0.5-0.8), "red" (<0.5).
    """
    if score > 0.8:
        return "green"
    if score >= 0.5:
        return "yellow"
    return "red"


def _format_score(score: float) -> str:
    """Format a score to 4 decimal places.

    Args:
        score: Numeric score.

    Returns:
        Formatted string.
    """
    return f"{score:.4f}"


def print_summary(results: EvalResults, verbose: bool = False) -> None:
    """Print evaluation results to the terminal using Rich tables.

    Args:
        results: The evaluation results to display.
        verbose: If True, show per-question detail in addition to aggregates.
    """
    console = Console()

    # Header
    console.print()
    console.print(f"[bold]ragtest[/bold] evaluation: [cyan]{results.dataset_name}[/cyan]")
    console.print()

    # Config summary
    cs = results.config_summary
    config_table = Table(title="Configuration", show_header=False, border_style="dim")
    config_table.add_column("Key", style="dim")
    config_table.add_column("Value")
    config_table.add_row("Chunking", f"{cs.get('chunking_strategy', '?')} (size={cs.get('chunk_size', '?')}, overlap={cs.get('overlap', '?')})")
    config_table.add_row("Retriever", f"{cs.get('retriever_type', '?')} (top_k={cs.get('top_k', '?')})")
    config_table.add_row("Generator", str(cs.get("generator_type", "none")))
    config_table.add_row("Total chunks", str(cs.get("total_chunks", 0)))
    config_table.add_row("Total questions", str(cs.get("total_questions", 0)))
    console.print(config_table)
    console.print()

    # Per-question detail (verbose mode)
    if verbose and results.question_results:
        detail_table = Table(title="Per-Question Results", border_style="blue")
        detail_table.add_column("#", style="dim", width=4)
        detail_table.add_column("Question", max_width=50)

        # Gather all metric names from the first question
        metric_names = sorted(results.question_results[0].metrics.keys())
        for name in metric_names:
            detail_table.add_column(name, justify="right", width=12)

        for i, qr in enumerate(results.question_results, 1):
            row = [str(i), qr.question[:50]]
            for name in metric_names:
                score = qr.metrics.get(name, 0.0)
                color = _score_color(score)
                row.append(f"[{color}]{_format_score(score)}[/{color}]")
            detail_table.add_row(*row)

        console.print(detail_table)
        console.print()

    # Aggregate summary
    if results.aggregate_metrics:
        agg_table = Table(title="Aggregate Metrics", border_style="green")
        agg_table.add_column("Metric", style="bold")
        agg_table.add_column("Score", justify="right")
        agg_table.add_column("Rating", justify="center")

        for name, score in sorted(results.aggregate_metrics.items()):
            color = _score_color(score)
            if score > 0.8:
                rating = f"[green]GOOD[/green]"
            elif score >= 0.5:
                rating = f"[yellow]FAIR[/yellow]"
            else:
                rating = f"[red]POOR[/red]"
            agg_table.add_row(name, f"[{color}]{_format_score(score)}[/{color}]", rating)

        console.print(agg_table)
        console.print()


def generate_html_report(results: EvalResults, output_path: str | Path) -> Path:
    """Generate an HTML evaluation report.

    Args:
        results: The evaluation results.
        output_path: Path for the output HTML file.

    Returns:
        The path to the generated HTML file.
    """
    output_path = Path(output_path)

    def score_css(score: float) -> str:
        if score > 0.8:
            return "color: #22c55e;"
        if score >= 0.5:
            return "color: #eab308;"
        return "color: #ef4444;"

    # Build metric columns from first question result
    metric_names = sorted(
        results.question_results[0].metrics.keys()
    ) if results.question_results else []

    # Per-question rows
    question_rows = ""
    for i, qr in enumerate(results.question_results, 1):
        cells = f"<td>{i}</td><td>{_html_escape(qr.question)}</td>"
        for name in metric_names:
            score = qr.metrics.get(name, 0.0)
            cells += f'<td style="{score_css(score)}">{_format_score(score)}</td>'
        question_rows += f"<tr>{cells}</tr>\n"

    # Aggregate rows
    agg_rows = ""
    for name, score in sorted(results.aggregate_metrics.items()):
        agg_rows += (
            f'<tr><td>{_html_escape(name)}</td>'
            f'<td style="{score_css(score)}">{_format_score(score)}</td></tr>\n'
        )

    # Metric headers
    metric_headers = "".join(f"<th>{_html_escape(n)}</th>" for n in metric_names)

    cs = results.config_summary

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ragtest report — {_html_escape(results.dataset_name)}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: #0f172a; color: #e2e8f0; padding: 2rem; }}
        h1 {{ color: #38bdf8; margin-bottom: 0.5rem; }}
        h2 {{ color: #94a3b8; margin: 2rem 0 1rem; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
        th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid #1e293b; }}
        th {{ background: #1e293b; color: #94a3b8; font-weight: 600; }}
        tr:hover {{ background: #1e293b; }}
        .config-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                        gap: 1rem; margin-bottom: 2rem; }}
        .config-card {{ background: #1e293b; padding: 1rem; border-radius: 0.5rem; }}
        .config-card .label {{ color: #64748b; font-size: 0.85rem; }}
        .config-card .value {{ color: #e2e8f0; font-size: 1.1rem; font-weight: 600; }}
        .footer {{ color: #475569; margin-top: 2rem; font-size: 0.85rem; }}
    </style>
</head>
<body>
    <h1>ragtest evaluation report</h1>
    <p style="color: #64748b;">Dataset: {_html_escape(results.dataset_name)}</p>

    <h2>Configuration</h2>
    <div class="config-grid">
        <div class="config-card">
            <div class="label">Chunking</div>
            <div class="value">{cs.get('chunking_strategy', '?')} (size={cs.get('chunk_size', '?')})</div>
        </div>
        <div class="config-card">
            <div class="label">Retriever</div>
            <div class="value">{cs.get('retriever_type', '?')} (top_k={cs.get('top_k', '?')})</div>
        </div>
        <div class="config-card">
            <div class="label">Total Chunks</div>
            <div class="value">{cs.get('total_chunks', 0)}</div>
        </div>
        <div class="config-card">
            <div class="label">Total Questions</div>
            <div class="value">{cs.get('total_questions', 0)}</div>
        </div>
    </div>

    <h2>Aggregate Metrics</h2>
    <table>
        <thead><tr><th>Metric</th><th>Score</th></tr></thead>
        <tbody>{agg_rows}</tbody>
    </table>

    <h2>Per-Question Results</h2>
    <table>
        <thead><tr><th>#</th><th>Question</th>{metric_headers}</tr></thead>
        <tbody>{question_rows}</tbody>
    </table>

    <div class="footer">Generated by ragtest v0.1.0</div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _html_escape(text: str) -> str:
    """Escape HTML special characters.

    Args:
        text: Raw string.

    Returns:
        HTML-safe string.
    """
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
