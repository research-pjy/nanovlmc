#!/usr/bin/env bash
# Activate qwen-vl yourself; this script never installs packages or calls Slurm.
set -euo pipefail
umask 077
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
MODE=${1:-smoke}
[[ "$MODE" == check || "$MODE" == smoke ]] || { echo "Usage: bash l40s/run.sh check|smoke" >&2; exit 1; }
DATA_DIR=${DATA_DIR:-/home/jayanth/datasets/nanovlm-caption-work/shortdesc-pilot-v1}
IMAGES_ROOT=${IMAGES_ROOT:-/home/jayanth/datasets/images}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/jayanth/nanovlm-runs}
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
mkdir -p "$OUTPUT_ROOT"
ATTEMPT=$(mktemp -d "$OUTPUT_ROOT/${MODE}-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
printf 'Logs and outputs: %s\n' "$ATTEMPT"
python -m nanovlm.cli.l40s "$MODE" --data-dir "$DATA_DIR" --images-root "$IMAGES_ROOT" \
    --config-dir "$ROOT/configs/l40s" --output-dir "$ATTEMPT/run" 2>&1 | tee "$ATTEMPT/console.log"
