import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.turnback_eval_common import (
    append_jsonl,
    is_complete_record,
    load_completed_records,
    load_samples,
    result_path_for,
)


def _record(route_id: int = 7) -> dict:
    return {
        "id": f"Toronto_Canada__easy__{route_id:04d}",
        "task": "route_reversal",
        "city": "Toronto_Canada",
        "difficulty": "easy",
        "route_id": route_id,
        "instructions_text": "Head east for 10 meters.",
        "route_geojson": {
            "type": "FeatureCollection",
            "metadata": {"query": {"coordinates": [[0.0, 0.0], [0.001, 0.0]]}},
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {"type": "LineString", "coordinates": [[0.0, 0.0], [0.001, 0.0]]},
                }
            ],
        },
    }


def test_load_samples_builds_source_signature(tmp_path: Path):
    data_file = tmp_path / "turnback_10pct.jsonl"
    data_file.write_text(json.dumps(_record(), ensure_ascii=False) + "\n", encoding="utf-8")

    samples = load_samples(data_file)

    assert len(samples) == 1
    assert samples[0].sample_id == "Toronto_Canada__easy__0007"
    assert samples[0].city == "Toronto_Canada"
    assert len(samples[0].source_signature) == 64


def test_completion_rule_rejects_incomplete_bad_and_nonfinite_records(tmp_path: Path):
    data_file = tmp_path / "turnback_10pct.jsonl"
    data_file.write_text(json.dumps(_record(), ensure_ascii=False) + "\n", encoding="utf-8")
    sample = load_samples(data_file)[0]
    result_path = tmp_path / "results.jsonl"

    incomplete = {
        "sample_id": sample.sample_id,
        "source_signature": sample.source_signature,
        "completed": False,
        "parser_status": "api_error",
        "similarity": None,
        "error": "boom",
    }
    nonfinite = {
        "sample_id": sample.sample_id,
        "source_signature": sample.source_signature,
        "completed": True,
        "parser_status": "ok",
        "similarity": math.nan,
        "error": None,
    }
    complete = {
        "sample_id": sample.sample_id,
        "source_signature": sample.source_signature,
        "completed": True,
        "parser_status": "ok",
        "similarity": 91.25,
        "error": None,
    }

    assert not is_complete_record(incomplete, sample)
    assert not is_complete_record(nonfinite, sample)
    assert is_complete_record(complete, sample)

    append_jsonl(result_path, incomplete)
    result_path.write_text(result_path.read_text(encoding="utf-8") + "{bad json\n", encoding="utf-8")
    append_jsonl(result_path, complete)

    completed, observed, bad_lines = load_completed_records(result_path, [sample])

    assert observed == 1
    assert bad_lines == 1
    assert completed[sample.sample_id]["similarity"] == 91.25


def test_result_path_is_model_scoped(tmp_path: Path):
    path = result_path_for(tmp_path, "qwen3/4b thinking", "data/turnback_10pct.jsonl")
    assert path == tmp_path / "qwen3_4b_thinking" / "turnback_10pct.jsonl"
