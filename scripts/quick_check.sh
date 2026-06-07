#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"

if command -v uv >/dev/null 2>&1; then
  RUN=(uv run --no-sync)
else
  RUN=()
fi

"${RUN[@]}" ruff check src/path_builder tests scripts --select F,E9
"${RUN[@]}" python -m compileall src/path_builder scripts
"${RUN[@]}" pytest -q
"${RUN[@]}" python -m path_builder.cli --help >/dev/null
"${RUN[@]}" python -m path_builder.cli generate-routes --help >/dev/null
"${RUN[@]}" python -m path_builder.cli generate-reverse --help >/dev/null
"${RUN[@]}" python scripts/evaluate_vllm.py --help >/dev/null
"${RUN[@]}" python scripts/estimate_eval_progress.py --help >/dev/null
"${RUN[@]}" python scripts/warm_eval_graph_cache.py --help >/dev/null
bash scripts/prepare_eval_data.sh --help >/dev/null
bash scripts/run_vllm_eval.sh --help >/dev/null
bash hfsync/local_to_remote.sh --help >/dev/null
bash hfsync/remote_to_local.sh --help >/dev/null
