#!/usr/bin/env bash
set -euo pipefail

# Download and install the lightweight TurnBack 10% evaluation dataset.
#
# Default behavior:
#   1. Download zhangdw/TurnBack-10pct from Hugging Face with `uv run hf`.
#   2. Keep a stable cache under ${TMPDIR:-/tmp}/turnback-data/ for resumable downloads.
#   3. Install data/turnback_10pct.jsonl and metadata/ into this repo's data/.

DATASET_REPO="${TURNBACK_DATASET_REPO:-zhangdw/TurnBack-10pct}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOWNLOAD_DIR=""
TARGET_DIR="${REPO_ROOT}/data"
FORCE=0
DEFAULT_DOWNLOAD_DIR=1

abs_path() {
  python3 - "$1" <<'PYABS'
import os
import sys
print(os.path.abspath(sys.argv[1]))
PYABS
}

usage() {
  cat <<EOF
Usage: $0 [options]

Download the sampled TurnBack route-reversal dataset from Hugging Face and
install it into the default evaluator path.

Options:
  --repo REPO_ID          Hugging Face dataset repo.
                          Default: ${DATASET_REPO}
  --download-dir DIR      Local dataset download/cache directory.
                          Default: stable dir under \${TMPDIR:-/tmp}
  --target-dir DIR        Destination data directory.
                          Default: ${TARGET_DIR}
  --force                 Replace an existing non-empty target directory.
  -h, --help              Show this help.

Examples:
  bash scripts/prepare_eval_data.sh
  bash scripts/prepare_eval_data.sh --force
  bash scripts/prepare_eval_data.sh --repo zhangdw/TurnBack-10pct

After running, the evaluator reads:
  data/turnback_10pct.jsonl
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      DATASET_REPO="${2:?Missing value for --repo}"
      shift 2
      ;;
    --download-dir)
      DOWNLOAD_DIR="$(abs_path "${2:?Missing value for --download-dir}")"
      DEFAULT_DOWNLOAD_DIR=0
      shift 2
      ;;
    --target-dir)
      TARGET_DIR="$(abs_path "${2:?Missing value for --target-dir}")"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if command -v uv >/dev/null 2>&1; then
  HF_CMD=(uv run hf)
elif command -v hf >/dev/null 2>&1; then
  HF_CMD=(hf)
else
  echo "ERROR: uv is required for the default workflow, or install the hf CLI." >&2
  exit 1
fi

if [[ -z "${DOWNLOAD_DIR}" ]]; then
  TMP_DOWNLOAD_PARENT="${TMPDIR:-/tmp}"
  TMP_DOWNLOAD_PARENT="${TMP_DOWNLOAD_PARENT%/}"
  [[ -n "${TMP_DOWNLOAD_PARENT}" ]] || TMP_DOWNLOAD_PARENT="/tmp"
  REPO_CACHE_NAME="${DATASET_REPO//\//__}"
  DOWNLOAD_DIR="${TMP_DOWNLOAD_PARENT}/turnback-data/${REPO_CACHE_NAME}"
fi
mkdir -p "${DOWNLOAD_DIR}"

echo "==> Repository root: ${REPO_ROOT}"
echo "==> HF dataset repo: ${DATASET_REPO}"
echo "==> Download/cache dir: ${DOWNLOAD_DIR}"
if [[ ${DEFAULT_DOWNLOAD_DIR} -eq 1 ]]; then
  echo "==> Download/cache dir is outside the repository and kept for resumable re-runs."
fi
echo "==> Target data dir: ${TARGET_DIR}"

echo "==> Downloading dataset with: ${HF_CMD[*]} download ${DATASET_REPO} --repo-type dataset"
"${HF_CMD[@]}" download "${DATASET_REPO}" \
  --repo-type dataset \
  --local-dir "${DOWNLOAD_DIR}"

SOURCE_JSONL="${DOWNLOAD_DIR}/data/turnback_10pct.jsonl"
if [[ ! -f "${SOURCE_JSONL}" ]]; then
  SOURCE_JSONL="${DOWNLOAD_DIR}/turnback_10pct.jsonl"
fi
if [[ ! -f "${SOURCE_JSONL}" ]]; then
  echo "ERROR: downloaded dataset does not contain data/turnback_10pct.jsonl" >&2
  exit 1
fi

if [[ -e "${TARGET_DIR}" ]]; then
  if [[ ${FORCE} -ne 1 ]]; then
    if find "${TARGET_DIR}" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
      cat >&2 <<EOF
ERROR: target directory already exists and is non-empty:
  ${TARGET_DIR}

Re-run with --force to replace it, or choose a different --target-dir.
EOF
      exit 1
    fi
  else
    echo "==> Removing existing target directory because --force was provided."
    rm -rf "${TARGET_DIR}"
  fi
fi

mkdir -p "${TARGET_DIR}"
cp "${SOURCE_JSONL}" "${TARGET_DIR}/turnback_10pct.jsonl"
if [[ -d "${DOWNLOAD_DIR}/metadata" ]]; then
  mkdir -p "${TARGET_DIR}/metadata"
  cp -a "${DOWNLOAD_DIR}/metadata/." "${TARGET_DIR}/metadata/"
fi
cat > "${TARGET_DIR}/source.json" <<EOF
{
  "dataset_repo": "${DATASET_REPO}",
  "download_dir": "${DOWNLOAD_DIR}",
  "installed_jsonl": "${TARGET_DIR}/turnback_10pct.jsonl"
}
EOF

echo "==> Installed TurnBack data: ${TARGET_DIR}/turnback_10pct.jsonl"
python3 - "${TARGET_DIR}/turnback_10pct.jsonl" <<'PYCOUNT'
import sys
from pathlib import Path
path = Path(sys.argv[1])
count = sum(1 for line in path.open(encoding="utf-8") if line.strip())
print(f"==> Records: {count}")
PYCOUNT

cat <<EOF

Next step:
  bash scripts/run_vllm_eval.sh \
    --model qwen3-4b-thinking-2507 \
    --base-url http://127.0.0.1:8000/v1 \
    --api-key EMPTY \
    --jobs 8
EOF
