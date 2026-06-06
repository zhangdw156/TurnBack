#!/usr/bin/env python3
"""Estimate TurnBack vLLM evaluation progress with the resume completion rule.

A sample is complete only if the result JSONL contains a completed record with a
matching ``sample_id`` and source signature, parser status ``ok``, no error, and
a finite similarity. Malformed JSONL lines and failed records are reported but do
not count as completed.

Examples:
    uv run python scripts/estimate_eval_progress.py
    uv run python scripts/estimate_eval_progress.py --model qwen3-4b-thinking-2507
    uv run python scripts/estimate_eval_progress.py --city Toronto_Canada --difficulty easy
    uv run python scripts/estimate_eval_progress.py --json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

try:
    from turnback_eval_common import (
        EvalSample,
        filter_samples,
        load_completed_records,
        load_samples,
        result_path_for,
        safe_name,
        split_csv,
        summarize_completed,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.turnback_eval_common import (
        EvalSample,
        filter_samples,
        load_completed_records,
        load_samples,
        result_path_for,
        safe_name,
        split_csv,
        summarize_completed,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FILE = REPO_ROOT / "data" / "turnback_10pct.jsonl"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


@dataclass(frozen=True)
class ModelProgress:
    model: str
    result_path: str
    completed: int
    observed: int
    bad_jsonl_lines: int
    pending: int
    total: int
    progress: float
    mean_similarity_done: float | None
    by_city: dict[str, dict[str, Any]]
    by_difficulty: dict[str, dict[str, Any]]


def discover_models(results_dir: Path, data_file: Path) -> list[str]:
    if not results_dir.exists():
        return []
    data_name = f"{safe_name(data_file.stem)}.jsonl"
    models = []
    for path in sorted(results_dir.glob(f"*/{data_name}")):
        if path.is_file():
            models.append(path.parent.name)
    return models


def progress_for_model(model: str, samples: list[EvalSample], data_file: Path, results_dir: Path) -> ModelProgress:
    result_path = result_path_for(results_dir, model, data_file)
    completed, observed, bad_lines = load_completed_records(result_path, samples)
    summary = summarize_completed(samples, completed)
    overall = summary["overall"]
    return ModelProgress(
        model=model,
        result_path=str(result_path),
        completed=int(overall["completed"]),
        observed=observed,
        bad_jsonl_lines=bad_lines,
        pending=int(overall["pending"]),
        total=int(overall["total"]),
        progress=float(overall["progress"]),
        mean_similarity_done=overall["mean_similarity_done"],
        by_city=summary["by_city"],
        by_difficulty=summary["by_difficulty"],
    )


def progress_style(progress: float) -> str:
    if progress >= 1.0:
        return "green"
    if progress > 0:
        return "yellow"
    return "red"


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_optional_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def render_table(progress_items: list[ModelProgress]) -> None:
    console = Console()
    table = Table(title="TurnBack evaluation progress", box=box.SIMPLE_HEAVY)
    table.add_column("model", overflow="fold")
    table.add_column("progress", justify="right")
    table.add_column("done/total", justify="right")
    table.add_column("observed", justify="right")
    table.add_column("bad jsonl", justify="right")
    table.add_column("mean sim", justify="right")
    table.add_column("result", overflow="fold")
    for item in progress_items:
        bar = ProgressBar(total=1.0, completed=item.progress, width=18)
        table.add_row(
            item.model,
            Text.assemble(bar, " ", Text(format_pct(item.progress), style=progress_style(item.progress))),
            f"{item.completed:,}/{item.total:,}",
            f"{item.observed:,}",
            f"{item.bad_jsonl_lines:,}",
            format_optional_score(item.mean_similarity_done),
            item.result_path,
        )
    console.print(Panel(table, title="Resume-compatible completion rule", subtitle="completed = matching id+signature, parser ok, finite similarity"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--model", action="append", default=[], help="Model/result directory to inspect; can repeat or use comma lists.")
    parser.add_argument("--city", default="", help="Comma-separated city filter.")
    parser.add_argument("--difficulty", default="", help="Comma-separated difficulty filter.")
    parser.add_argument("--id", dest="ids", action="append", default=[], help="Sample id filter; can repeat or use comma lists.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a Rich table.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.data_file.exists():
        parser.error(f"data file not found: {args.data_file}; run scripts/prepare_eval_data.sh first")
    ids = [item for raw in args.ids for item in split_csv(raw)]
    samples = filter_samples(
        load_samples(args.data_file),
        cities=split_csv(args.city),
        difficulties=split_csv(args.difficulty),
        ids=ids,
        offset=args.offset,
        limit=args.limit,
    )
    if not samples:
        parser.error("no samples selected")

    models = [item for raw in args.model for item in split_csv(raw)]
    if not models:
        models = discover_models(args.results_dir, args.data_file)
    if not models:
        models = ["qwen3-4b-thinking-2507"]

    items = [progress_for_model(model, samples, args.data_file, args.results_dir) for model in models]
    if args.json:
        print(json.dumps([asdict(item) for item in items], ensure_ascii=False, indent=2))
    else:
        render_table(items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
