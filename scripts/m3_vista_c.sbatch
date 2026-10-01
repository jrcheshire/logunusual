#!/bin/bash
#SBATCH --job-name=logunusual-m3-vista-c
#SBATCH --account=JPL-SPHEREx
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.log
#
# M3 on Vista, fourth job: the seven-bin realization re-measured after the sample-stage
# rebuild (per-slab streams on a thread pool, angular pre-cut) and the per-radius
# spline table. Baselines: job 1002399's 259 s on this partition, the laptop's 82 s on
# the same commit's code, and job 1014508's 512^3 field stage (6.27 s unblocked, each
# P -> P_G conversion 2.57 s, 98% host).
#
# Phases are cheap-first: the 512^3 `pkg` split and `wall` measure the spline table's
# gain on the card against 1014508 in seconds, before the long leg. Reuses the gpu env
# in this checkout (built by job 999776). No `/usr/bin/time` (absent on the compute
# nodes); `nproc` is not the allocation, the affinity mask is.
#
# Submit from the repo root on Vista:
#     sbatch scripts/m3_vista_c.sbatch

set -u
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export PIXI_CACHE_DIR="/tmp/pixi-cache-${USER}"

cd "$SLURM_SUBMIT_DIR" || exit 1
RUNS="runs/m3"
OUT="$SCRATCH/logunusual_m3"
mkdir -p "$RUNS" "$OUT"

NCORES=$(python3 -c 'import os; print(len(os.sched_getaffinity(0)))')
export OMP_NUM_THREADS="$NCORES"

echo "=== host $(hostname)  job ${SLURM_JOB_ID}  $(date -Is) ==="
echo "=== commit $(git rev-parse --short HEAD)  account JPL-SPHEREx  partition gh ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "=== affinity cores $NCORES  OMP_NUM_THREADS $OMP_NUM_THREADS ==="

echo; echo "##### phase 0: device check #####"
pixi run --locked -e gpu python -u -c '
import jax
devs = jax.devices()
print("devices:", devs)
assert any(d.platform == "gpu" for d in devs), "no CUDA device"
' || { echo "DEVICE CHECK FAILED"; exit 1; }

echo; echo "##### phase 1: P -> P_G split at 512^3 (spline table on Grace) #####"
pixi run --locked -e gpu python -u scripts/m3_device.py pkg \
    --bin bin05 --repeat 5 --out "$RUNS/vista_c_pkg_bin05.json"
echo "rc=$?"

echo; echo "##### phase 2: 512^3 field stage wall #####"
pixi run --locked -e gpu python -u scripts/m3_device.py wall \
    --bin bin05 --out "$RUNS/vista_c_wall_bin05.json"
echo "rc=$?"

echo; echo "##### phase 3: full seven-bin realization -> $OUT #####"
time pixi run --locked -e gpu logunusual run \
    --config configs/v28_default.yaml \
    --output-dir "$OUT" \
    --realizations 0:1 \
    --overwrite
rc=$?
echo "rc=$rc"

echo; echo "##### summary.json + layout check #####"
cat "$OUT"/v28/realization_00000/summary.json 2>/dev/null || echo "(no summary written)"
pixi run --locked -e gpu logunusual check "$OUT"/v28/realization_00000/catalog.parq
echo "check rc=$?"

echo
echo "=== done $(date -Is) ==="
du -sh "$OUT" 2>/dev/null
exit $rc
