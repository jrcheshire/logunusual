#!/bin/bash
#SBATCH --job-name=logunusual-m3
#SBATCH --partition=compute
#SBATCH --nodelist=deneb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log
#
# M3 phase 1: what the RTX 3050 can hold, how far CUDA drifts from CPU, and where
# the field stage's time goes. Read-only with respect to the package -- it measures
# the code as committed and writes only JSON under runs/m3/.
#
# Submit from the repo root on deneb:
#     sbatch scripts/m3_deneb.sbatch
#
# The card is 6 GB. At the measured 7.0 x N^3 float64 the three 512^3 bins need
# 7.5 GB, so an OOM in phase 1 is an expected RESULT, not a failure: the child
# records it and the job carries on. Phase 2 re-runs the edge bins with JAX allowed
# 95% of the card (the default pool is a 75% GROWTH CAP even unpreallocated), which
# is what decides whether bin04 at 448^3 is reachable in float64 at all.

set -u
export PATH="$HOME/.pixi/bin:$PATH"
export OMP_NUM_THREADS=16          # part of the pin for the CPU leg of phase 4
export PYTHONUNBUFFERED=1          # stream the log; never reconstruct it from a tail

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p runs/m3

echo "=== host $(hostname)  job ${SLURM_JOB_ID}  $(date -Is) ==="
echo "=== commit $(git rev-parse --short HEAD) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
echo

run() { echo; echo "##### $* #####"; pixi run -e gpu python -u "$@"; echo "exit=$?"; }

# ---- phase 0: fail fast if JAX cannot see the card ----------------------------
echo "##### phase 0: device check #####"
pixi run -e gpu python -u -c '
import jax
d = jax.devices()
print("devices:", d)
assert any(x.platform == "gpu" for x in d), "no CUDA device -- aborting"
s = jax.local_devices()[0].memory_stats() or {}
lim = s.get("bytes_limit")
print("allocator limit:", f"{lim / 2**30:.2f} GiB" if lim else "unreported")
' || { echo "PHASE 0 FAILED -- JAX has no CUDA device; nothing else would mean anything"; exit 1; }

# ---- phase 1: f64 footprint per production bin, default pool -----------------
run scripts/m3_device.py ladder --bins bin01 bin02 bin03 bin04 bin05 \
    --out runs/m3/ladder_f64_default.json

# ---- phase 2: the edge bins with 95% of the card -----------------------------
echo; echo "##### phase 2: same bins, XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 #####"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 pixi run -e gpu python -u \
    scripts/m3_device.py ladder --bins bin04 bin05 \
    --out runs/m3/ladder_f64_p95.json
echo "exit=$?"

# ---- phase 3: what float32 would buy -----------------------------------------
run scripts/m3_device.py ladder --bins bin04 bin05 --dtype f32 \
    --out runs/m3/ladder_f32.json

# ---- phase 4: CPU vs CUDA agreement, one process, one jaxlib -----------------
run scripts/m3_device.py ulp --n 64 128 256 --out runs/m3/ulp.json

# ---- phase 5: where the field stage's time goes ------------------------------
# bin03 at 384^3 fits the card in f64 on the arithmetic, so it is the leg that
# should produce numbers; bin05 at 512^3 is attempted and may OOM, which is a result.
run scripts/m3_device.py wall --bin bin03 --out runs/m3/wall_bin03.json
run scripts/m3_device.py wall --bin bin05 --out runs/m3/wall_bin05.json

echo
echo "=== done $(date -Is) ==="
ls -la runs/m3/
