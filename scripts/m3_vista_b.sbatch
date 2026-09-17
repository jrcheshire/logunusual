#!/bin/bash
#SBATCH --job-name=logunusual-m3-vista-b
#SBATCH --account=JPL-SPHEREx
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.log
#
# M3 on Vista, second job: the full seven-bin realization that m3_vista.sbatch's
# phase 5 never started (`/usr/bin/time` does not exist on the gh compute nodes).
# Reuses the gpu environment that job built in this checkout; summary.json carries
# the per-bin timings and memory, so no external timer is needed.
#
# TACC sets OMP_NUM_THREADS=1 and `nproc` reports that, not the allocation; the
# affinity mask is the core count, and threading is opened up to it.
#
# Submit from the repo root on Vista:
#     sbatch scripts/m3_vista_b.sbatch

set -u
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export PIXI_CACHE_DIR="/tmp/pixi-cache-${USER}"

cd "$SLURM_SUBMIT_DIR" || exit 1
OUT="$SCRATCH/logunusual_m3"
mkdir -p "$OUT"

NCORES=$(python3 -c 'import os; print(len(os.sched_getaffinity(0)))')
export OMP_NUM_THREADS="$NCORES"

echo "=== host $(hostname)  job ${SLURM_JOB_ID}  $(date -Is) ==="
echo "=== commit $(git rev-parse --short HEAD)  account JPL-SPHEREx  partition gh ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "=== affinity cores $NCORES  OMP_NUM_THREADS $OMP_NUM_THREADS ==="

echo; echo "##### device check #####"
pixi run --locked -e gpu python -u -c '
import jax
devs = jax.devices()
print("devices:", devs)
assert any(d.platform == "gpu" for d in devs), "no CUDA device"
'
rc=$?; [ $rc -eq 0 ] || { echo "DEVICE CHECK FAILED (rc=$rc)"; exit 1; }

echo; echo "##### full seven-bin realization -> $OUT #####"
time pixi run --locked -e gpu logunusual run \
    --config configs/v28_default.yaml \
    --output-dir "$OUT" \
    --realizations 0:1 \
    --overwrite
rc=$?
echo "rc=$rc"

echo; echo "##### summary.json #####"
cat "$OUT"/v28/realization_00000/summary.json 2>/dev/null || echo "(no summary written)"

echo
echo "=== done $(date -Is) ==="
du -sh "$OUT" 2>/dev/null
exit $rc
