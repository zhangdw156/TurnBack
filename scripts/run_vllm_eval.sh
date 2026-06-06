#!/usr/bin/env bash
set -euo pipefail

# TurnBack route-reversal evaluator wrapper for OpenAI-compatible vLLM servers.
#
# Quick start from a fresh server clone:
#   uv sync
#   bash scripts/prepare_eval_data.sh
#   bash scripts/run_vllm_eval.sh \
#     --model qwen3-4b-thinking-2507 \
#     --base-url http://127.0.0.1:8000/v1 \
#     --api-key EMPTY \
#     --jobs 8
#
# Resume is enabled by default by scripts/evaluate_vllm.py. Completed samples
# are skipped only when their result record has matching sample id/source
# signature, parser_status=ok, and a finite similarity. Use --force only for an
# intentional full rerun.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-qwen3-4b-thinking-2507}"
BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
API_KEY="${OPENAI_API_KEY:-EMPTY}"
JOBS="${JOBS:-8}"
DATA_FILE="${REPO_ROOT}/data/turnback_10pct.jsonl"
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: $0 [options] [-- extra evaluator args]

Required in practice:
  --model MODEL          Served model name exposed by vLLM.
                         Default: ${MODEL}
  --base-url URL         OpenAI-compatible base URL.
                         Default: ${BASE_URL}
  --api-key KEY          API key; vLLM usually accepts EMPTY.
                         Default: ${API_KEY}

Common options:
  --jobs N               Concurrent sample workers. Default: ${JOBS}
  --data-file FILE       Dataset JSONL. Default: ${DATA_FILE}
  --limit N              Smoke-test first N selected samples.
  --city LIST            Comma-separated city filter.
  --difficulty LIST      Comma-separated difficulty filter.
  --force                Full rerun: delete existing result JSONL for this model/data file.
  --dry-run              Show selection/resume state and sample prompts; do not call vLLM.
  -h, --help             Show this help.

Everything after -- is passed to scripts/evaluate_vllm.py.

Examples:
  bash scripts/run_vllm_eval.sh --limit 3 --jobs 2
  bash scripts/run_vllm_eval.sh --jobs 16
  bash scripts/run_vllm_eval.sh --city Toronto_Canada --difficulty easy --jobs 8
  bash scripts/run_vllm_eval.sh --dry-run -- --print-sample-prompts 2
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="${2:?missing value for --model}"
      shift 2
      ;;
    --base-url)
      BASE_URL="${2:?missing value for --base-url}"
      shift 2
      ;;
    --api-key)
      API_KEY="${2:?missing value for --api-key}"
      shift 2
      ;;
    --jobs)
      JOBS="${2:?missing value for --jobs}"
      shift 2
      ;;
    --data-file)
      DATA_FILE="${2:?missing value for --data-file}"
      shift 2
      ;;
    --limit|--city|--difficulty|--id|--offset)
      EXTRA_ARGS+=("$1" "${2:?missing value for $1}")
      shift 2
      ;;
    --force|--dry-run|--save-prompts|--save-geojson|--refresh-graph-cache)
      EXTRA_ARGS+=("$1")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "${JOBS}" =~ ^[0-9]+$ ]] || [[ "${JOBS}" -lt 1 ]]; then
  echo "ERROR: --jobs must be a positive integer, got: ${JOBS}" >&2
  exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required. Install uv, then run: uv sync" >&2
  exit 2
fi
if [[ ! -f "${DATA_FILE}" ]]; then
  cat >&2 <<EOF
ERROR: data file does not exist:
  ${DATA_FILE}

Run first:
  bash scripts/prepare_eval_data.sh
EOF
  exit 2
fi

cd "${REPO_ROOT}"

echo "TurnBack vLLM wrapper"
echo "  model    : ${MODEL}"
echo "  base_url : ${BASE_URL}"
echo "  jobs     : ${JOBS}"
echo "  data     : ${DATA_FILE}"
echo "  resume   : enabled by default (pass --force only for full rerun)"

uv run python scripts/evaluate_vllm.py \
  --model "${MODEL}" \
  --base-url "${BASE_URL}" \
  --api-key "${API_KEY}" \
  --jobs "${JOBS}" \
  --data-file "${DATA_FILE}" \
  "${EXTRA_ARGS[@]}"
