#!/usr/bin/env bash
# Usage: scripts/run_session.sh <gpu> <cpus> [run_session.py arguments]
#   <cpus>: logical CPUs for taskset; CPU c and c+4 share physical core c on this machine,
#   so use 0,1,4,5 with GPU 0 and 2,3,6,7 with GPU 1.
# AR_DETERMINISTIC=1 sets XLA_FLAGS=--xla_gpu_deterministic_ops=true.
set -euo pipefail
GPU=$1
CPUS=$2
shift 2
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
if [ "${AR_DETERMINISTIC:-0}" = 1 ]; then
    export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
fi
exec taskset -c "$CPUS" venv_ar/bin/python scripts/run_session.py "$@"
