#!/usr/bin/env bash
# Run from any directory: bash dgx/submit.sh smoke|train configs/<name>.yaml [checkpoint]
set -euo pipefail
umask 077
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${1:?Pass smoke or train}
CONFIG=${2:?Pass repository-relative config path}
RESUME=${3:-}
[[ "$MODE" == smoke || "$MODE" == train ]] || { echo "Mode must be smoke or train" >&2; exit 1; }
[[ "$CONFIG" == configs/*.yaml && "$CONFIG" != *..* && -f "$ROOT/$CONFIG" ]] || { echo "Use a committed configs/*.yaml path" >&2; exit 1; }
source "$ROOT/dgx/cluster.local.sh"
: "${OUTPUT_ROOT:?}" "${PARTITION:?}" "${GPU_GRES:?}" "${DATA_DIR:?}" "${IMAGES_ROOT:?}" "${CONDA_SH:?}" "${CONDA_ENV:?}"
[[ -d "$DATA_DIR" && -d "$IMAGES_ROOT" ]] || { echo "Data directories are missing" >&2; exit 1; }
if [[ -n "$RESUME" ]]; then RESUME=$(realpath "$RESUME"); test -f "$RESUME"; fi
[[ -z $(git -C "$ROOT" status --porcelain) ]] || { echo "Commit or preserve checkout changes before submission" >&2; exit 1; }
COMMIT=$(git -C "$ROOT" rev-parse HEAD)
mkdir -p "$OUTPUT_ROOT/scheduler-logs" "$OUTPUT_ROOT/releases" "$OUTPUT_ROOT/attempts"
SNAPSHOT=$(mktemp -d "$OUTPUT_ROOT/releases/${COMMIT:0:12}-XXXXXX")
# Self-contained source snapshot: future pulls cannot alter a pending job.
git -C "$ROOT" archive "$COMMIT" | tar -x -C "$SNAPSHOT"
test -f "$SNAPSHOT/$CONFIG"
printf '%s\n' "$COMMIT" > "$SNAPSHOT/COMMIT"
opts=(--partition="$PARTITION" --gres="$GPU_GRES" --nodes=1 --ntasks=1
      --cpus-per-task="${CPUS:-4}" --mem="${HOST_MEMORY:-16G}" --time="${WALL_TIME:-00:15:00}"
      --chdir="$SNAPSHOT" --output="$OUTPUT_ROOT/scheduler-logs/%x-%j.out" --error="$OUTPUT_ROOT/scheduler-logs/%x-%j.err")
[[ -z ${QOS:-} ]] || opts+=(--qos="$QOS")
[[ -z ${ACCOUNT:-} ]] || opts+=(--account="$ACCOUNT")
export OUTPUT_ROOT DATA_DIR IMAGES_ROOT CONDA_SH CONDA_ENV
SUBMISSION=$(sbatch --parsable "${opts[@]}" "$SNAPSHOT/dgx/run.sbatch" "$MODE" "$SNAPSHOT/$CONFIG" "$RESUME")
printf '%s\t%s\t%s\t%s\n' "$SUBMISSION" "$COMMIT" "$CONFIG" "$SNAPSHOT" >> "$OUTPUT_ROOT/submissions.tsv"
printf 'Submitted %s from %s\n' "$SUBMISSION" "$SNAPSHOT"
