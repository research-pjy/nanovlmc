#!/usr/bin/env bash
#SBATCH --job-name=run
#SBATCH --signal=USR1@180
set -euo pipefail
umask 077
MODE=${1:?}
CONFIG=${2:?}
RESUME=${3:-}
# Slurm may spool this script elsewhere. The wrapper sets --chdir to the snapshot.
ROOT=$PWD
test -f "$ROOT/COMMIT"
source "$ROOT/dgx/activate_env.sh"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
ATTEMPT="$OUTPUT_ROOT/attempts/${SLURM_JOB_ID:?}-r${SLURM_RESTART_COUNT:-0}"
mkdir "$ATTEMPT"
cp "$ROOT/COMMIT" "$ATTEMPT/COMMIT"
python -m pip freeze > "$ATTEMPT/packages.txt"
if [[ "$MODE" == smoke ]]; then
    srun --ntasks=1 python -m nanovlm preflight --config "$CONFIG" --data-dir "$DATA_DIR" \
        --images-root "$IMAGES_ROOT" --device cuda --report "$ATTEMPT/preflight.json" --decode-all
elif [[ "$MODE" == train ]]; then
    args=(--config "$CONFIG" --data-dir "$DATA_DIR" --images-root "$IMAGES_ROOT" --device cuda --output-dir "$ATTEMPT/training")
    [[ -z "$RESUME" ]] || args+=(--resume "$RESUME")
    srun --ntasks=1 python -m nanovlm train "${args[@]}"
    test -f "$ATTEMPT/training/COMPLETE"
else
    echo "Unknown mode: $MODE" >&2
    exit 1
fi
