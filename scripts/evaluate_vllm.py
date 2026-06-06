#!/usr/bin/env python3
"""Evaluate TurnBack-10pct with an OpenAI-compatible vLLM chat server.

Server workflow:

    uv sync
    bash scripts/prepare_eval_data.sh
    bash scripts/run_vllm_eval.sh \
      --model qwen3-4b-thinking-2507 \
      --base-url http://127.0.0.1:8000/v1 \
      --api-key EMPTY \
      --jobs 8

Resume is enabled by default. A sample is skipped only when the result JSONL has
an earlier completed record whose ``sample_id`` and source signature match the
current dataset row, whose parser status is ``ok``, and whose similarity is a
finite number. Interrupted or malformed JSONL lines, API errors, parse failures,
and execution failures are not treated as completed. Use ``--force`` only when
you intentionally want to discard previous results for this model/data file.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI
from tqdm import tqdm

try:  # Allow both `python scripts/evaluate_vllm.py` and module-style imports.
    from turnback_eval_common import (
        EvalSample,
        append_jsonl,
        filter_samples,
        load_completed_records,
        load_samples,
        result_path_for,
        safe_name,
        sha256_text,
        split_csv,
        stable_dumps,
        summarize_completed,
        summary_path_for,
        write_json_atomic,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.turnback_eval_common import (
        EvalSample,
        append_jsonl,
        filter_samples,
        load_completed_records,
        load_samples,
        result_path_for,
        safe_name,
        sha256_text,
        split_csv,
        stable_dumps,
        summarize_completed,
        summary_path_for,
        write_json_atomic,
    )

from path_builder.datasets import get_start_end_points
from path_builder.execution import PathBuilder, recommended_graph_dist
from path_builder.instructions import parse_instruction_lines
from path_builder.models import ExecutionState, SimilarityThresholds, SimilarityWeights
from path_builder.prompting import extract_instruction_lines_from_response
from path_builder.similarity import extract_linestring_coordinates, score_geojson_routes

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FILE = REPO_ROOT / "data" / "turnback_10pct.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results"
DEFAULT_GRAPH_CACHE_DIR = REPO_ROOT / "cache" / "turnback_eval_graphs"
DEFAULT_SYSTEM_PROMPT = (
    "You are a route-reversal assistant. Return only executable reverse pedestrian "
    "navigation instructions, one step per line, ending with an arrive instruction."
)
OUTPUT_RULES = (
    "\n\nOutput requirements:\n"
    "- Write only the reverse navigation instructions, one step per line.\n"
    "- The first movement step must start at the destination and head back toward the original start.\n"
    "- Keep distances in meters when possible.\n"
    "- End with an arrive instruction.\n"
    "- Do not use markdown bullets, code fences, or extra explanation.\n"
)
_thread_local = threading.local()


def load_similarity_config(path: Path | None) -> tuple[SimilarityWeights, SimilarityThresholds]:
    if path is None:
        return SimilarityWeights(), SimilarityThresholds()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SimilarityWeights(**payload.get("weights", {})), SimilarityThresholds(**payload.get("thresholds", {}))


def normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(content)


def extract_extra_message_field(message: Any, names: Iterable[str]) -> str | None:
    for name in names:
        value = getattr(message, name, None)
        if value:
            return normalize_content(value)
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        for name in names:
            value = extra.get(name)
            if value:
                return normalize_content(value)
    if isinstance(message, dict):
        for name in names:
            value = message.get(name)
            if value:
                return normalize_content(value)
    return None


def usage_to_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return {
        key: getattr(usage, key)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if hasattr(usage, key)
    }


def parse_extra_body(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise SystemExit("--extra-body-json must decode to a JSON object")
    return parsed


def get_client(args: argparse.Namespace) -> OpenAI:
    client = getattr(_thread_local, "client", None)
    key = (args.base_url, args.api_key, args.timeout)
    if client is None or getattr(_thread_local, "client_key", None) != key:
        client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout, max_retries=0)
        _thread_local.client = client
        _thread_local.client_key = key
    return client


def build_messages(sample: EvalSample, args: argparse.Namespace) -> list[dict[str, str]]:
    prompt = sample.prompt.rstrip() + OUTPUT_RULES
    messages: list[dict[str, str]] = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def chat_once(sample: EvalSample, args: argparse.Namespace, extra_body: dict[str, Any] | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": build_messages(sample, args),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body
    response = get_client(args).chat.completions.create(**kwargs)
    choice = response.choices[0]
    message = choice.message
    content = normalize_content(getattr(message, "content", ""))
    reasoning = extract_extra_message_field(message, ("reasoning", "reasoning_content"))
    return {
        "raw_output": content,
        "raw_reasoning": reasoning,
        "finish_reason": getattr(choice, "finish_reason", None),
        "usage": usage_to_dict(getattr(response, "usage", None)),
    }


def chat_with_retry(sample: EvalSample, args: argparse.Namespace, extra_body: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str | None]:
    last_error: str | None = None
    for attempt in range(args.retries + 1):
        try:
            return chat_once(sample, args, extra_body), None
        except Exception as exc:  # noqa: BLE001 - service errors should be checkpointed, not crash the run.
            last_error = repr(exc)
            if attempt < args.retries and args.retry_sleep > 0:
                time.sleep(args.retry_sleep)
    return None, last_error


def final_answer_region(raw_output: str, raw_reasoning: str | None, think_end_tag: str) -> str:
    text = raw_output or ""
    if raw_reasoning and raw_output:
        text = f"<think>\n{raw_reasoning}\n{think_end_tag}\n{raw_output}"
    if think_end_tag and think_end_tag in text:
        return text.rsplit(think_end_tag, 1)[1].strip()
    # Some reasoning servers emit XML-ish tags with different casing.
    parts = re.split(re.escape(think_end_tag), text, flags=re.IGNORECASE) if think_end_tag else [text]
    return parts[-1].strip()


def reverse_reference_geojson(route_geojson: dict[str, Any]) -> dict[str, Any]:
    lines = extract_linestring_coordinates(route_geojson)
    if not lines:
        raise ValueError("reference route_geojson does not contain a LineString")
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"turnback_reference": "original_route_reversed"},
                "geometry": {"type": "LineString", "coordinates": list(reversed(lines[0]))},
            }
        ],
    }


def graph_cache_path(sample: EvalSample, args: argparse.Namespace, dist: int) -> Path:
    return (
        args.graph_cache_dir
        / safe_name(sample.city)
        / safe_name(sample.difficulty)
        / safe_name(str(sample.route_id))
        / safe_name(args.network_type)
        / f"dist_{dist}"
        / "graph.graphml"
    )


def load_or_build_builder(sample: EvalSample, args: argparse.Namespace, center_latlon: tuple[float, float], dist: int) -> tuple[PathBuilder, str, str | None]:
    cache_path = graph_cache_path(sample, args, dist) if args.graph_cache_dir else None
    if cache_path and cache_path.exists() and not args.refresh_graph_cache:
        import osmnx as ox

        try:
            graph = ox.load_graphml(cache_path)
        except ValueError:
            graph = ox.load_graphml(cache_path, node_dtypes={"osmid": str})
        return PathBuilder(graph), "graph-cache", str(cache_path)

    import osmnx as ox

    ox.settings.use_cache = True
    ox.settings.cache_folder = str(REPO_ROOT / "osmnx_cache")
    builder = PathBuilder.from_osm(center_latlon, dist=dist, network_type=args.network_type)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(builder.graph, filepath=cache_path)
    return builder, "live-osm", str(cache_path) if cache_path else None


def base_record(sample: EvalSample, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "source_signature": sample.source_signature,
        "sample_index": sample.index,
        "city": sample.city,
        "difficulty": sample.difficulty,
        "route_id": sample.route_id,
        "model": args.model,
        "base_url": args.base_url,
        "created_at": datetime.now(UTC).isoformat(),
        "completed": False,
        "parser_status": None,
        "similarity": None,
        "scores": None,
        "error": None,
    }


def evaluate_one(
    sample: EvalSample,
    args: argparse.Namespace,
    *,
    weights: SimilarityWeights,
    thresholds: SimilarityThresholds,
    extra_body: dict[str, Any] | None,
) -> dict[str, Any]:
    record = base_record(sample, args)
    if args.save_prompts:
        record["messages"] = build_messages(sample, args)
        record["prompt_hash"] = sha256_text(stable_dumps(record["messages"]))

    fields, error = chat_with_retry(sample, args, extra_body)
    if fields is None:
        record.update({"parser_status": "api_error", "error": error or "api_error"})
        return record
    record.update(fields)

    answer_region = final_answer_region(fields.get("raw_output") or "", fields.get("raw_reasoning"), args.think_end_tag)
    cleaned_lines = extract_instruction_lines_from_response(answer_region)
    record["answer_region"] = answer_region
    record["cleaned_instructions"] = cleaned_lines
    if not cleaned_lines:
        record.update({"parser_status": "no_navigation_lines", "error": "failed to extract navigation instructions"})
        return record

    try:
        commands = parse_instruction_lines(cleaned_lines)
    except Exception as exc:  # noqa: BLE001
        record.update({"parser_status": "parse_error", "error": f"instruction_parse_error:{type(exc).__name__}:{exc}"})
        return record
    if not commands:
        record.update({"parser_status": "parse_error", "error": "no parsed navigation commands"})
        return record

    try:
        _start, end = get_start_end_points(sample.route_geojson)
        if end is None:
            raise ValueError("route_geojson metadata has no end coordinate")
        end_lon, end_lat = float(end[0]), float(end[1])
        center_latlon = (end_lat, end_lon)
        dist = max(args.min_dist, recommended_graph_dist(commands))
        builder, graph_source, graph_cache = load_or_build_builder(sample, args, center_latlon, dist)
        trace = builder.execute(
            commands,
            ExecutionState(current_coordinates=center_latlon, current_heading=0.0),
            executor=args.executor,
        )
        prediction_geojson = builder.trace_to_geojson(trace)
        reference_geojson = reverse_reference_geojson(sample.route_geojson)
        score = score_geojson_routes(
            prediction_geojson,
            reference_geojson,
            weights=weights,
            thresholds=thresholds,
        )
    except Exception as exc:  # noqa: BLE001
        record.update({"parser_status": "execution_error", "error": f"{type(exc).__name__}:{exc}"})
        return record

    if args.save_geojson:
        geojson_dir = args.output_dir / safe_name(args.model) / "geojson"
        geojson_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = geojson_dir / f"{safe_name(sample.sample_id)}.prediction.geojson"
        reference_path = geojson_dir / f"{safe_name(sample.sample_id)}.reference.geojson"
        write_json_atomic(prediction_path, prediction_geojson)
        write_json_atomic(reference_path, reference_geojson)
        record["prediction_geojson_path"] = str(prediction_path)
        record["reference_geojson_path"] = str(reference_path)

    record.update(
        {
            "completed": True,
            "parser_status": "ok",
            "error": None,
            "similarity": score.similarity,
            "scores": score.scores,
            "weights": score.weights,
            "thresholds": score.params,
            "graph_source": graph_source,
            "graph_cache_path": graph_cache,
            "executor": args.executor,
            "dist": dist,
            "waypoint_count": len(trace.waypoints),
            "segment_count": len(trace.segment_coordinates),
            "final_coordinates": list(trace.final_state.current_coordinates),
        }
    )
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL"), help="vLLM served model name; defaults to VLLM_MODEL.")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--city", default="", help="Comma-separated city filter, e.g. Toronto_Canada,Tokyo_23_wards.")
    parser.add_argument("--difficulty", default="", help="Comma-separated difficulty filter: easy,medium,hard.")
    parser.add_argument("--id", dest="ids", action="append", default=[], help="Sample id to evaluate; can repeat or use comma lists.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=int(os.getenv("JOBS", "8")), help="Concurrent sample workers.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--request-sleep", type=float, default=0.0)
    parser.add_argument("--extra-body-json", default=None)
    parser.add_argument("--think-end-tag", default="</think>")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "similarity.paper.json")
    parser.add_argument("--executor", choices=("greedy", "search", "hybrid"), default="hybrid")
    parser.add_argument("--network-type", default="walk")
    parser.add_argument("--min-dist", type=int, default=1000, help="Minimum OSM graph radius in meters.")
    parser.add_argument("--graph-cache-dir", type=Path, default=DEFAULT_GRAPH_CACHE_DIR)
    parser.add_argument("--refresh-graph-cache", action="store_true")
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--save-geojson", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Load data and show pending/resume state without calling the endpoint.")
    parser.add_argument("--print-sample-prompts", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Delete this model/data result JSONL before running.")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip completed matching samples. Enabled by default; use --no-resume only for debugging.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.model:
        parser.error("--model is required unless VLLM_MODEL is set")
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.offset < 0:
        parser.error("--offset must be >= 0")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if not args.data_file.exists():
        parser.error(f"data file not found: {args.data_file}; run scripts/prepare_eval_data.sh first")

    weights, thresholds = load_similarity_config(args.config if args.config and args.config.exists() else None)
    all_samples = load_samples(args.data_file)
    ids = [item for raw in args.ids for item in split_csv(raw)]
    samples = filter_samples(
        all_samples,
        cities=split_csv(args.city),
        difficulties=split_csv(args.difficulty),
        ids=ids,
        offset=args.offset,
        limit=args.limit,
    )
    if not samples:
        parser.error("no samples selected")

    result_path = result_path_for(args.output_dir, args.model, args.data_file)
    summary_path = summary_path_for(args.output_dir, args.model, args.data_file)
    if args.force and result_path.exists():
        result_path.unlink()

    if args.resume:
        completed, observed_count, bad_lines = load_completed_records(result_path, samples)
    else:
        completed, observed_count, bad_lines = {}, 0, 0
    pending = [sample for sample in samples if sample.sample_id not in completed]

    print("TurnBack vLLM evaluation")
    print(f"  model       : {args.model}")
    print(f"  base_url    : {args.base_url}")
    print(f"  data_file   : {args.data_file}")
    print(f"  output      : {result_path}")
    print(f"  jobs        : {args.jobs}")
    print(f"  resume      : {'enabled' if args.resume else 'disabled'}")
    print(f"  selected    : {len(samples)}")
    print(f"  observed    : {observed_count}")
    print(f"  completed   : {len(completed)}")
    print(f"  pending     : {len(pending)}")
    if bad_lines:
        print(f"  bad_jsonl   : {bad_lines} (ignored; not counted as completed)")

    for sample in samples[: args.print_sample_prompts]:
        print("=" * 80)
        print(f"{sample.sample_id} city={sample.city} difficulty={sample.difficulty}")
        print(sample.prompt.rstrip() + OUTPUT_RULES)

    if args.dry_run:
        print("Dry run: no endpoint calls made.")
        return 0
    if not pending:
        summary = summarize_completed(samples, completed)
        summary.update({"model": args.model, "data_file": str(args.data_file), "result_path": str(result_path)})
        write_json_atomic(summary_path, summary)
        print("No pending samples: existing results are complete for this selection.")
        print(f"Saved summary to: {summary_path}")
        return 0

    extra_body = parse_extra_body(args.extra_body_json)
    completed_now = dict(completed)
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_to_sample = {
            executor.submit(
                evaluate_one,
                sample,
                args,
                weights=weights,
                thresholds=thresholds,
                extra_body=extra_body,
            ): sample
            for sample in pending
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_sample), total=len(future_to_sample), desc="TurnBack eval"):
            sample = future_to_sample[future]
            try:
                record = future.result()
            except Exception as exc:  # pragma: no cover - defensive checkpoint for unexpected worker crashes.
                record = base_record(sample, args)
                record.update({"parser_status": "worker_error", "error": f"{type(exc).__name__}:{exc}"})
            append_jsonl(result_path, record)
            if record.get("completed") is True:
                completed_now[sample.sample_id] = record
            else:
                failures += 1
            if args.request_sleep:
                time.sleep(args.request_sleep)

    summary = summarize_completed(samples, completed_now)
    summary.update(
        {
            "created_at": datetime.now(UTC).isoformat(),
            "model": args.model,
            "base_url": args.base_url,
            "data_file": str(args.data_file),
            "result_path": str(result_path),
            "resume": args.resume,
            "jobs": args.jobs,
            "new_failures_or_incomplete": failures,
        }
    )
    write_json_atomic(summary_path, summary)
    print("Finished TurnBack vLLM evaluation")
    print(f"  completed: {summary['overall']['completed']}/{summary['overall']['total']}")
    print(f"  pending  : {summary['overall']['pending']}")
    print(f"  summary  : {summary_path}")
    return 0 if summary["overall"]["pending"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
