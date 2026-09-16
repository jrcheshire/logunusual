#!/bin/bash
#SBATCH --job-name=logunusual-m3-vista
#SBATCH --account=JPL-SPHEREx
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.log
#
# M3's GPU leg on a real card. deneb's RTX 3050 answered the question it could
# (6 GB holds 3 of 7 bins in float64 and gives no speedup over the laptop CPU), so
# it stays a correctness box; the performance numbers come from here.
#
# Everything runs on the compute node, INCLUDING the environment build. A pixi/pip
# install is not login-node work: attempting it there hit the per-user thread cap
# (rayon "Resource temporarily unavailable") and that limit is the point, not an
# obstacle to tune around.
#
# Submit from the repo root on Vista:
#     sbatch scripts/m3_vista.sbatch

set -u
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
# Keep the package cache off Lustre; pixi warns and redirects anyway, and a compute
# node's /tmp is local.
export PIXI_CACHE_DIR="/tmp/pixi-cache-${USER}"

cd "$SLURM_SUBMIT_DIR" || exit 1
OUT="$SCRATCH/logunusual_m3"
mkdir -p runs/m3 "$OUT"

echo "=== host $(hostname)  job ${SLURM_JOB_ID}  $(date -Is) ==="
echo "=== commit $(git rev-parse --short HEAD)  account JPL-SPHEREx  partition gh ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "=== cores $(nproc)  ulimit -u $(ulimit -u) ==="
echo

# ---- phase 0: build the environment here, not on the login node --------------
# No `rm -rf`: `pixi clean` is the supported way to drop the environments.
echo "##### phase 0: environment #####"
pixi clean || echo "(pixi clean reported nothing to remove)"
pixi install --locked
rc=$?; echo "pixi install (default) rc=$rc"; [ $rc -eq 0 ] || exit 1
pixi install --locked -e gpu
rc=$?; echo "pixi install (gpu) rc=$rc"; [ $rc -eq 0 ] || exit 1
pixi run -e gpu setup-gpu
rc=$?; echo "setup-gpu rc=$rc"; [ $rc -eq 0 ] || exit 1

# ---- phase 1: fail fast if JAX cannot see the GH200 --------------------------
echo; echo "##### phase 1: device check #####"
pixi run -e gpu python -u -c '
import jax
devs = jax.devices()
print("devices:", devs)
assert any(d.platform == "gpu" for d in devs), "no CUDA device"
stats = jax.local_devices()[0].memory_stats() or {}
lim = stats.get("bytes_limit")
print("allocator limit:", "%.2f GiB" % (lim / 2**30) if lim else "unreported")
'
rc=$?; [ $rc -eq 0 ] || { echo "PHASE 1 FAILED (rc=$rc) -- see the traceback above"; exit 1; }

run() { echo; echo "##### $* #####"; pixi run -e gpu python -u "$@"; echo "rc=$?"; }

# ---- phase 2: footprint per bin ---------------------------------------------
# 512^3 needs ~12.9 GiB on the coefficient measured on deneb; the card has ~96 GB,
# so every bin should fit and the interesting number is the coefficient itself at
# sizes deneb could not reach.
run scripts/m3_device.py ladder --bins bin01 bin02 bin03 bin04 bin05 \
    --out runs/m3/vista_ladder_f64.json

# ---- phase 3: per-bin field-stage wall, the ROADMAP's table ------------------
for b in bin01 bin02 bin03 bin04 bin05 bin06 bin07; do
    run scripts/m3_device.py wall --bin "$b" --out "runs/m3/vista_wall_${b}.json"
done

# ---- phase 4: CPU vs CUDA, to pair against deneb's record --------------------
run scripts/m3_device.py ulp --n 64 128 256 --out runs/m3/vista_ulp.json

# ---- phase 5: one full seven-bin realization --------------------------------
# The sample stage is 80% of the laptop wall and nothing so far has measured it on
# a GPU node. Same mask as the M2 acceptance run (fsky 0.7127), so the 187 s / 651.4M
# galaxies / 15.04 GiB laptop figures are the comparison. ~15 GB lands in $SCRATCH.
echo; echo "##### phase 5: full seven-bin realization -> $OUT #####"
/usr/bin/time -v pixi run -e gpu logunusual run \
    --config configs/v28_default.yaml \
    --output-dir "$OUT" \
    --realizations 0:1 \
    --overwrite
echo "rc=$?"

echo; echo "##### summary.json #####"
cat "$OUT"/v28/realization_00000/summary.json 2>/dev/null || echo "(no summary written)"

echo
echo "=== done $(date -Is) ==="
ls -la runs/m3/
du -sh "$OUT" 2>/dev/null
