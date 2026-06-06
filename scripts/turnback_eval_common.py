#!/usr/bin/env python3
"""Shared helpers for the TurnBack vLLM evaluation scripts."""

from __future__ import annotations

import json
import math
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SIGNATURE_FIELDS = ("id", "task", "city", "difficulty", "route_id", "instructions_text", "route_geojson")


@dataclass(frozen=True)
class EvalSample:
    index: int
    sample_id: str
    city: str
    difficulty: str
    route_id: int | str
    prompt: str
    instructions_text: str
    route_geojson: dict[str, Any]
    raw: dict[str, Any]
    source_signature: str


def safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return safe.strip("_") or "value"


def stable_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_prompt(city: str, difficulty: str, route_id: int | str, instructions_text: str) -> str:
    city_text = str(city).replace("_", " ")
    return (
        "You are evaluating route-level geospatial cognition.\n\n"
        "Given the following forward pedestrian navigation instructions, generate reverse "
        "navigation instructions that guide a pedestrian from the destination back to the "
        "original starting point. Preserve street names when useful and write executable "
        "step-by-step navigation instructions.\n\n"
        f"City: {city_text}\n"
        f"Difficulty: {difficulty}\n"
        f"Route ID: {route_id}\n\n"
        "Forward navigation instructions:\n"
        f"{instructions_text.strip()}\n"
    )


def source_signature(record: dict[str, Any]) -> str:
    payload = {field: record.get(field) for field in SIGNATURE_FIELDS}
    return sha256_text(stable_dumps(payload))


def _sample_id(record: dict[str, Any], index: int) -> str:
    if record.get("id"):
        return str(record["id"])
    city = record.get("city", "unknown_city")
    difficulty = record.get("difficulty", "unknown_difficulty")
    route_id = record.get("route_id", index)
    try:
        route_text = f"{int(route_id):04d}"
    except (TypeError, ValueError):
        route_text = safe_name(str(route_id))
    return f"{city}__{difficulty}__{route_text}"


def make_sample(record: dict[str, Any], index: int) -> EvalSample:
    sample_id = _sample_id(record, index)
    city = str(record.get("city") or "unknown_city")
    difficulty = str(record.get("difficulty") or "unknown_difficulty")
    route_id: int | str = record.get("route_id", index)
    instructions_text = str(record.get("instructions_text") or "")
    route_geojson = record.get("route_geojson")
    if not isinstance(route_geojson, dict):
        raise ValueError(f"record {sample_id} has no route_geojson object")
    prompt = str(record.get("prompt") or build_prompt(city, difficulty, route_id, instructions_text))
    normalized = dict(record)
    normalized.setdefault("id", sample_id)
    normalized.setdefault("task", "route_reversal")
    return EvalSample(
        index=index,
        sample_id=sample_id,
        city=city,
        difficulty=difficulty,
        route_id=route_id,
        prompt=prompt,
        instructions_text=instructions_text,
        route_geojson=route_geojson,
        raw=normalized,
        source_signature=source_signature(normalized),
    )


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    bad_lines = 0
    if not path.exists():
        return records, bad_lines
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            if isinstance(item, dict):
                records.append(item)
            else:
                bad_lines += 1
    return records, bad_lines


def load_samples(path: str | Path) -> list[EvalSample]:
    data_path = Path(path)
    samples: list[EvalSample] = []
    with data_path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"expected JSON object on line {index + 1}: {data_path}")
            samples.append(make_sample(record, index))
    if not samples:
        raise ValueError(f"no samples found in {data_path}")
    return samples


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def filter_samples(
    samples: Iterable[EvalSample],
    *,
    cities: Iterable[str] | None = None,
    difficulties: Iterable[str] | None = None,
    ids: Iterable[str] | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> list[EvalSample]:
    city_filter = set(cities or [])
    difficulty_filter = set(difficulties or [])
    id_filter = set(ids or [])
    selected = [
        sample
        for sample in samples
        if (not city_filter or sample.city in city_filter)
        and (not difficulty_filter or sample.difficulty in difficulty_filter)
        and (not id_filter or sample.sample_id in id_filter)
    ]
    if offset:
        selected = selected[offset:]
    if limit is not None:
        selected = selected[:limit]
    return selected


def result_path_for(output_dir: str | Path, model: str, data_file: str | Path) -> Path:
    data_stem = safe_name(Path(data_file).stem)
    return Path(output_dir) / safe_name(model) / f"{data_stem}.jsonl"


def summary_path_for(output_dir: str | Path, model: str, data_file: str | Path) -> Path:
    data_stem = safe_name(Path(data_file).stem)
    return Path(output_dir) / safe_name(model) / f"{data_stem}_summary.json"


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        f.write("\n")
        f.flush()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def is_complete_record(record: dict[str, Any], sample: EvalSample) -> bool:
    if record.get("sample_id") != sample.sample_id:
        return False
    if record.get("source_signature") != sample.source_signature:
        return False
    if record.get("completed") is not True:
        return False
    if record.get("parser_status") != "ok":
        return False
    if record.get("error") not in (None, ""):
        return False
    try:
        similarity = float(record.get("similarity"))
    except (TypeError, ValueError):
        return False
    return math.isfinite(similarity)


def load_completed_records(result_path: str | Path, samples: Iterable[EvalSample]) -> tuple[dict[str, dict[str, Any]], int, int]:
    samples_by_id = {sample.sample_id: sample for sample in samples}
    records, bad_lines = read_jsonl(Path(result_path))
    completed: dict[str, dict[str, Any]] = {}
    observed_ids: set[str] = set()
    for record in records:
        sample_id = str(record.get("sample_id", ""))
        sample = samples_by_id.get(sample_id)
        if sample is None:
            continue
        observed_ids.add(sample_id)
        if is_complete_record(record, sample):
            completed[sample_id] = record
    return completed, len(observed_ids), bad_lines


def summarize_completed(samples: list[EvalSample], completed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    similarities = [float(record["similarity"]) for record in completed.values()]
    by_city: dict[str, dict[str, Any]] = {}
    by_difficulty: dict[str, dict[str, Any]] = {}
    for sample in samples:
        is_done = sample.sample_id in completed
        for bucket, key in ((by_city, sample.city), (by_difficulty, sample.difficulty)):
            entry = bucket.setdefault(key, {"completed": 0, "total": 0, "similarity_sum": 0.0})
            entry["total"] += 1
            if is_done:
                entry["completed"] += 1
                entry["similarity_sum"] += float(completed[sample.sample_id]["similarity"])
    for bucket in (by_city, by_difficulty):
        for entry in bucket.values():
            done = int(entry["completed"])
            total = int(entry["total"])
            entry["progress"] = done / total if total else 0.0
            entry["mean_similarity_done"] = entry["similarity_sum"] / done if done else None
            del entry["similarity_sum"]
    total = len(samples)
    done = len(completed)
    return {
        "overall": {
            "completed": done,
            "total": total,
            "pending": total - done,
            "progress": done / total if total else 0.0,
            "mean_similarity_done": sum(similarities) / len(similarities) if similarities else None,
            "min_similarity_done": min(similarities) if similarities else None,
            "max_similarity_done": max(similarities) if similarities else None,
        },
        "by_city": dict(sorted(by_city.items())),
        "by_difficulty": dict(sorted(by_difficulty.items())),
    }
