#!/usr/bin/env bash
# Usage: scripts/smoke_test/run_pilot.sh      (phase C needs the proxy: sudo proxy/install_proxy.sh)
# Runs the phases of smoke_ar_pilot.py (ID case2_ar_smoke_pilot) in order, then the plots.
# CPU c and c+4 share physical core c on this machine: GPU 0 gets 0,1,4,5 and GPU 1 gets 2,3,6,7.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=venv_ar/bin/python
S=scripts/smoke_test/smoke_ar_pilot.py
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
DIR=outputs/$($PY -c "from autoresearch.harness import run_id; print(run_id('case2_ar_smoke_pilot'))" 2>/dev/null)
mkdir -p "$DIR"
T0=$(date +%s)
CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 $PY $S --phase A --dir "$DIR"
CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 $PY $S --phase B --gpu 0 --dir "$DIR" > "$DIR/B_gpu0.log" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 $PY $S --phase B --gpu 1 --dir "$DIR" > "$DIR/B_gpu1.log" 2>&1 &
P1=$!
wait $P0
wait $P1
$PY $S --phase C --dir "$DIR"
$PY $S --phase summary --dir "$DIR"
$PY scripts/smoke_test/plot_smoke_ar_pilot.py "$DIR"
echo "pilot wall time $(( $(date +%s) - T0 )) s -> $DIR"
