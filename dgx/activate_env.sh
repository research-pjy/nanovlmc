#!/usr/bin/env bash
# Source from a noninteractive allocated job. No network or installation.
: "${CONDA_SH:?Set CONDA_SH to the existing conda.sh path}"
: "${CONDA_ENV:?Set CONDA_ENV (dgx-research-test on your DGX)}"
test -r "$CONDA_SH"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
