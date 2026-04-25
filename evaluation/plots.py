"""Text-based visualisation helpers for StratArena benchmark results.

These helpers work with zero extra dependencies; a ``matplotlib`` path can be
added later.  All functions accept the list of dicts returned by
:func:`evaluation.benchmarks.benchmark_all_tasks`.
"""
from __future__ import annotations

from typing import Any


_FILL = "█"
_EMPTY = "░"
_BAR_WIDTH = 30


def _bar(value: float, width: int = _BAR_WIDTH) -> str:
    """Return an ASCII progress bar for *value* in [0, 1]."""
    filled = max(0, min(width, round(value * width)))
    return _FILL * filled + _EMPTY * (width - filled)


def print_score_chart(
    rows: list[dict[str, Any]],
    title: str = "StratArena Benchmark Results",
) -> None:
    """Print a horizontal bar chart of per-task scores.

    Example output::

        StratArena Benchmark Results
        ============================================================
            easy │ ████████████████░░░░░░░░░░░░░░ │ 0.5600
          medium │ ██████████████████░░░░░░░░░░░░ │ 0.6100
            hard │ ████████████░░░░░░░░░░░░░░░░░░ │ 0.4200
        ------------------------------------------------------------
         average │ ████████████████░░░░░░░░░░░░░░ │ 0.5300
    """
    print(f"\n{title}")
    print("=" * (14 + _BAR_WIDTH + 12))
    for row in rows:
        task = str(row.get("task", "?"))
        score = float(row.get("score", 0.0))
        print(f"  {task:>8} │ {_bar(score)} │ {score:.4f}")
    if rows:
        avg = sum(float(r.get("score", 0.0)) for r in rows) / len(rows)
        print("-" * (14 + _BAR_WIDTH + 12))
        print(f"  {'average':>8} │ {_bar(avg)} │ {avg:.4f}")
    print()


def _fmt_cell(val: Any, width: int) -> str:
    """Format a single table cell, using 4dp for floats."""
    text = f"{val:.4f}" if isinstance(val, float) else str(val)
    return f"{text:<{width}}"


def print_metric_table(rows: list[dict[str, Any]]) -> None:
    """Print a plain-text table of all metrics for each task.

    Columns are determined dynamically from the keys of the first row.
    """
    if not rows:
        return

    headers = list(rows[0].keys())
    col_widths = {
        h: max(
            len(h),
            *(len(_fmt_cell(r.get(h, ""), 0)) for r in rows),
        )
        for h in headers
    }

    separator = "-+-".join("-" * col_widths[h] for h in headers)
    header_line = " | ".join(f"{h:<{col_widths[h]}}" for h in headers)
    print(header_line)
    print(separator)
    for row in rows:
        line = " | ".join(_fmt_cell(row.get(h, ""), col_widths[h]) for h in headers)
        print(line)
    print()


def print_full_report(rows: list[dict[str, Any]]) -> None:
    """Print both the score chart and the full metric table."""
    print_score_chart(rows)
    print_metric_table(rows)
