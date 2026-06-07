#!/usr/bin/env python3
"""Warm TurnBack evaluator OSM graph caches without calling an LLM.

This script builds the same GraphML cache files that ``scripts/evaluate_vllm.py``
uses before executing parsed model instructions. Run it with low concurrency to
avoid overloading public Overpass endpoints, then run the evaluator with the same
``--graph-cache-dir``, ``--network-type``, and ``--min-dist``/``--dist`` value.

Examples:
    uv run python scripts/warm_eval_graph_cache.py --dist 3500 --jobs 2
    uv run python scripts/warm_eval_graph_cache.py --city Toronto_Canada --limit 10 --dry-run
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

try:  # Allow both `python scripts/warm_eval_graph_cache.py` and module-style imports.
    from turnback_eval_common import EvalSample, filter_samples, load_samples, safe_name, split_csv
except ModuleNotFoundError:  # pragma: no cover
    from scripts.turnback_eval_common import EvalSample, filter_samples, load_samples, safe_name, split_csv

from path_builder.datasets import get_start_end_points
from path_builder.execution import PathBuilder

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FILE = REPO_ROOT / "data" / "turnback_10pct.jsonl"
DEFAULT_GRAPH_CACHE_DIR = REPO_ROOT / "cache" / "turnback_eval_graphs"
DEFAULT_OSMNX_CACHE_DIR = REPO_ROOT / "osmnx_cache"


@dataclass(frozen=True)
class WarmResult:
    sample_id: str
    city: str
    difficulty: str
    route_id: int | str
    status: str
    cache_path: str
    error: str | None = None


def graph_cache_path(sample: EvalSample, graph_cache_dir: Path, network_type: str, dist: int) -> Path:
    """Return the cache path used by scripts/evaluate_vllm.py for the same sample."""
    return (
        graph_cache_dir
        / safe_name(sample.city)
        / safe_name(sample.difficulty)
        / safe_name(str(sample.route_id))
        / safe_name(network_type)
        / f"dist_{dist}"
        / "graph.graphml"
    )


def sample_center_latlon(sample: EvalSample) -> tuple[float, float]:
    _start, end = get_start_end_points(sample.route_geojson)
    if end is None:
        raise ValueError("route_geojson metadata has no end coordinate")
    end_lon, end_lat = float(end[0]), float(end[1])
    return end_lat, end_lon


def configure_osmnx(cache_dir: Path, timeout: int | None) -> None:
    import osmnx as ox

    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache_dir)
    if timeout is not None and hasattr(ox.settings, "requests_timeout"):
        ox.settings.requests_timeout = timeout


def warm_sample(sample: EvalSample, args: argparse.Namespace) -> WarmResult:
    cache_path = graph_cache_path(sample, args.graph_cache_dir, args.network_type, args.dist)
    if cache_path.exists() and not args.refresh:
        return WarmResult(sample.sample_id, sample.city, sample.difficulty, sample.route_id, "cached", str(cache_path))
    if args.dry_run:
        status = "refresh-needed" if cache_path.exists() else "missing"
        return WarmResult(sample.sample_id, sample.city, sample.difficulty, sample.route_id, status, str(cache_path))

    try:
        configure_osmnx(args.osmnx_cache_dir, args.overpass_timeout)
        builder = PathBuilder.from_osm(sample_center_latlon(sample), dist=args.dist, network_type=args.network_type)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        import osmnx as ox

        ox.save_graphml(builder.graph, filepath=cache_path)
        return WarmResult(sample.sample_id, sample.city, sample.difficulty, sample.route_id, "warmed", str(cache_path))
    except Exception as exc:  # noqa: BLE001 - cache warming should continue and report all failures.
        return WarmResult(
            sample.sample_id,
            sample.city,
            sample.difficulty,
            sample.route_id,
            "failed",
            str(cache_path),
            f"{type(exc).__name__}:{exc}",
        )


def write_manifest(path: Path, results: list[WarmResult], args: argparse.Namespace) -> None:
    payload: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "data_file": str(args.data_file),
        "graph_cache_dir": str(args.graph_cache_dir),
        "osmnx_cache_dir": str(args.osmnx_cache_dir),
        "network_type": args.network_type,
        "dist": args.dist,
        "jobs": args.jobs,
        "dry_run": args.dry_run,
        "refresh": args.refresh,
        "results": [asdict(result) for result in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--graph-cache-dir", type=Path, default=DEFAULT_GRAPH_CACHE_DIR)
    parser.add_argument("--osmnx-cache-dir", type=Path, default=DEFAULT_OSMNX_CACHE_DIR)
    parser.add_argument("--dist", type=int, default=3500, help="OSM graph radius in meters. Match evaluator --min-dist.")
    parser.add_argument("--network-type", default="walk")
    parser.add_argument("--jobs", type=int, default=2, help="Concurrent Overpass/OSM graph builders. Keep low for public Overpass.")
    parser.add_argument("--city", default="", help="Comma-separated city filter.")
    parser.add_argument("--difficulty", default="", help="Comma-separated difficulty filter.")
    parser.add_argument("--id", dest="ids", action="append", default=[], help="Sample id filter; can repeat or use comma lists.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--refresh", action="store_true", help="Rebuild graph files even when graph.graphml already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Show missing/cache paths without downloading from OSM.")
    parser.add_argument("--request-sleep", type=float, default=0.0, help="Sleep after each completed warm task.")
    parser.add_argument("--overpass-timeout", type=int, default=180, help="OSMnx requests_timeout in seconds when supported.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional JSON manifest path for per-sample warm results.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.dist < 1:
        parser.error("--dist must be >= 1")
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.offset < 0:
        parser.error("--offset must be >= 0")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
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

    print("TurnBack OSM graph cache warmup")
    print(f"  data_file       : {args.data_file}")
    print(f"  selected        : {len(samples)}")
    print(f"  graph_cache_dir : {args.graph_cache_dir}")
    print(f"  osmnx_cache_dir : {args.osmnx_cache_dir}")
    print(f"  network_type    : {args.network_type}")
    print(f"  dist            : {args.dist}")
    print(f"  jobs            : {args.jobs}")
    print(f"  dry_run         : {args.dry_run}")
    print(f"  refresh         : {args.refresh}")

    results: list[WarmResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_to_sample = {executor.submit(warm_sample, sample, args): sample for sample in samples}
        for future in tqdm(concurrent.futures.as_completed(future_to_sample), total=len(future_to_sample), desc="Warm graph cache"):
            result = future.result()
            results.append(result)
            if result.status == "failed":
                tqdm.write(f"FAILED {result.sample_id}: {result.error}")
            if args.request_sleep:
                time.sleep(args.request_sleep)

    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    print("Finished graph cache warmup")
    for status in sorted(counts):
        print(f"  {status:14s}: {counts[status]}")

    if args.manifest:
        write_manifest(args.manifest, results, args)
        print(f"  manifest      : {args.manifest}")

    return 1 if counts.get("failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
