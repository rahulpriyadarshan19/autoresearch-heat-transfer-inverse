#!/usr/bin/env bash
# Usage: scripts/smoke_test/run_smoke_gpu.sh <gpu> <cpus> <script> [arguments]
#   Runs one smoke test on one GPU. <cpus> as in scripts/run_session.sh: 0,1,4,5 with GPU 0, 2,3,6,7 with GPU 1.
set -euo pipefail
GPU=$1
CPUS=$2
shift 2
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
exec taskset -c "$CPUS" venv_ar/bin/python "$@"
