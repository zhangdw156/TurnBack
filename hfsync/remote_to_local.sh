#!/usr/bin/env bash
set -euo pipefail

# Direction: Hugging Face bucket -> local TurnBack-eval results.
# Safe default: no local deletion. Pass --delete only when the remote prefix is
# the authoritative complete copy; it deletes local files absent remotely.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUCKET_ID="${HF_BUCKET_ID:-zhangdw/leo-benchmark}"
REMOTE_PREFIX="${HF_TURNBACK_PREFIX:-TurnBack-eval}"
HF_CLI_STRING="${HF_CLI:-uvx hf}"
LOCAL_DIR="${REPO_ROOT}/results"
DRY_RUN=0
DELETE=0
IGNORE_EXISTING=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: hfsync/remote_to_local.sh [options] [-- extra hf sync args]

Direction:
  HF bucket -> LOCAL TurnBack-eval results

Default sync pair:
  hf://buckets/zhangdw/leo-benchmark/TurnBack-eval/results -> results/

Safe defaults:
  - Does not delete local files absent remotely.
  - May update same-path local files if remote files differ.
  - Use --ignore-existing / --new-only for strictly create-only behavior.

Options:
  --results-dir DIR     Local results directory. Default: results/
  --dry-run             Print the sync plan without downloading.
  --delete              Delete LOCAL files absent remotely. Use carefully.
  --ignore-existing     Skip local files that already exist; only download new files.
  --new-only            Alias for --ignore-existing.
  --bucket BUCKET_ID    Bucket ID. Default: zhangdw/leo-benchmark.
  --prefix PREFIX       Remote prefix inside the bucket. Default: TurnBack-eval.
  -h, --help            Show this help.

Environment overrides:
  HF_BUCKET_ID          Same as --bucket.
  HF_TURNBACK_PREFIX    Same as --prefix.
  HF_CLI                Command used to run hf. Default: "uvx hf".

Examples:
  hfsync/remote_to_local.sh --dry-run
  hfsync/remote_to_local.sh --new-only
EOF
}

abs_path() {
  python3 - "$1" <<'PYABS'
import os
import sys
print(os.path.abspath(sys.argv[1]))
PYABS
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-dir)
      LOCAL_DIR="$(abs_path "${2:?Missing value for --results-dir}")"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --delete)
      DELETE=1
      shift
      ;;
    --ignore-existing|--new-only)
      IGNORE_EXISTING=1
      shift
      ;;
    --bucket)
      BUCKET_ID="${2:?Missing value for --bucket}"
      shift 2
      ;;
    --prefix)
      REMOTE_PREFIX="${2:?Missing value for --prefix}"
      shift 2
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
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

REMOTE_PREFIX="${REMOTE_PREFIX#/}"
REMOTE_PREFIX="${REMOTE_PREFIX%/}"
read -r -a HF_CMD <<< "${HF_CLI_STRING}"
mkdir -p "${LOCAL_DIR}"

if [[ -n "${REMOTE_PREFIX}" ]]; then
  REMOTE_URI="hf://buckets/${BUCKET_ID}/${REMOTE_PREFIX}/results"
else
  REMOTE_URI="hf://buckets/${BUCKET_ID}/results"
fi

CMD=("${HF_CMD[@]}" buckets sync "${REMOTE_URI}" "${LOCAL_DIR}")
[[ "${DELETE}" -eq 1 ]] && CMD+=(--delete)
[[ "${DRY_RUN}" -eq 1 ]] && CMD+=(--dry-run)
[[ "${IGNORE_EXISTING}" -eq 1 ]] && CMD+=(--ignore-existing)
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "==> Artifact: results"
echo "Direction: REMOTE -> LOCAL"
echo "Remote:    ${REMOTE_URI}"
echo "Local:     ${LOCAL_DIR}"
[[ "${DRY_RUN}" -eq 1 ]] && echo "Mode:      dry-run" || echo "Mode:      apply"
[[ "${DELETE}" -eq 1 ]] && echo "Delete:    enabled (LOCAL files absent remotely may be deleted)" || echo "Delete:    disabled"
[[ "${IGNORE_EXISTING}" -eq 1 ]] && echo "Existing:  skip local-existing files" || echo "Existing:  update same-path local files if changed"
printf 'Command:  '
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
