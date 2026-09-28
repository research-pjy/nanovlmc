# Copy to cluster.local.sh ON THE DGX; populated machine configuration is ignored.
# These are examples to verify against your site and current allocation.
export CONDA_SH="/absolute/path/to/conda/etc/profile.d/conda.sh"
export CONDA_ENV="dgx-research-test"
export DATA_DIR="/absolute/path/to/versioned/descriptions"
export IMAGES_ROOT="/absolute/path/to/coco/images"
export OUTPUT_ROOT="/absolute/path/to/outputs/nanovlm"
export PARTITION="longq"
export QOS="longq"
export GPU_GRES="gpu:a100:1"
export ACCOUNT=""
export CPUS=4
export HOST_MEMORY="16G"
# Set from measurements; default is deliberately a bounded diagnostic allocation.
export WALL_TIME="00:15:00"
