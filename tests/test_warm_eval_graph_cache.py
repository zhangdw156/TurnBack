import json
from argparse import Namespace
from pathlib import Path

from scripts.turnback_eval_common import load_samples
from scripts.warm_eval_graph_cache import graph_cache_path, sample_center_latlon, warm_sample


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
            "metadata": {"query": {"coordinates": [[-79.0, 43.0], [-79.001, 43.002]]}},
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[-79.0, 43.0], [-79.001, 43.002]],
                    },
                }
            ],
        },
    }


def _sample(tmp_path: Path):
    data_file = tmp_path / "turnback_10pct.jsonl"
    data_file.write_text(json.dumps(_record(), ensure_ascii=False) + "\n", encoding="utf-8")
    return load_samples(data_file)[0]


def test_graph_cache_path_matches_evaluator_layout(tmp_path: Path):
    sample = _sample(tmp_path)

    path = graph_cache_path(sample, tmp_path / "cache" / "turnback_eval_graphs", "walk", 3500)

    assert path == (
        tmp_path
        / "cache"
        / "turnback_eval_graphs"
        / "Toronto_Canada"
        / "easy"
        / "7"
        / "walk"
        / "dist_3500"
        / "graph.graphml"
    )


def test_sample_center_latlon_uses_route_end_coordinate(tmp_path: Path):
    sample = _sample(tmp_path)

    assert sample_center_latlon(sample) == (43.002, -79.001)


def test_warm_sample_skips_existing_cache_without_network(tmp_path: Path):
    sample = _sample(tmp_path)
    graph_cache_dir = tmp_path / "cache" / "turnback_eval_graphs"
    cache_path = graph_cache_path(sample, graph_cache_dir, "walk", 3500)
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("cached", encoding="utf-8")
    args = Namespace(
        graph_cache_dir=graph_cache_dir,
        osmnx_cache_dir=tmp_path / "osmnx_cache",
        network_type="walk",
        dist=3500,
        refresh=False,
        dry_run=False,
        overpass_timeout=180,
    )

    result = warm_sample(sample, args)

    assert result.status == "cached"
    assert result.cache_path == str(cache_path)
    assert result.error is None


def test_warm_sample_dry_run_reports_missing_cache(tmp_path: Path):
    sample = _sample(tmp_path)
    args = Namespace(
        graph_cache_dir=tmp_path / "cache" / "turnback_eval_graphs",
        osmnx_cache_dir=tmp_path / "osmnx_cache",
        network_type="walk",
        dist=3500,
        refresh=False,
        dry_run=True,
        overpass_timeout=180,
    )

    result = warm_sample(sample, args)

    assert result.status == "missing"
    assert result.cache_path.endswith("/walk/dist_3500/graph.graphml")
    assert result.error is None
