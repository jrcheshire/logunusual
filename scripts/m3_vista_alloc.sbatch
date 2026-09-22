#!/bin/bash
#SBATCH --job-name=logunusual-m3-alloc
#SBATCH --account=JPL-SPHEREx
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.log
#
# M3, third Vista job: the ALLOCATOR peaks the laptop cannot report.
#
# The laptop settled two things and could not settle a third. float32 halves the
# stage's LIVE arrays (7.02 -> 3.51 x N^3 f64) and jit leaves them untouched -- but
# what decides whether a bin fits a card is the device allocator's peak, which adds
# XLA intra-op scratch on top (50-60% in the f64 deneb table) and which a CPU backend
# does not report at all. This job measures it over f64/f32 x eager/jit.
#
# Two arms ride along because they are the same 15-second process start:
#   - `pkg` on Grace, the within-job control. The laptop split says the P -> P_G step
#     is 65% host work; the claim that this is what holds the GH200 to 1.2x currently
#     rests on scaling that host half by a cross-machine factor. Running the split ON
#     the node replaces the arithmetic with a measurement.
#   - `wall --jit` at 512^3, so "jit buys nothing" is stated about a card rather than
#     extrapolated from a CPU backend where the device half is already small.
#
# Reuses the gpu environment in this checkout (built by job 999776). Phases are
# ordered cheap-first: a device check and the 192^3 ladder point fail in seconds if
# the env or the branch is wrong, before anything long runs. Every phase streams.
#
# Submit from the repo root on Vista:
#     sbatch scripts/m3_vista_alloc.sbatch

set -u
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export PIXI_CACHE_DIR="/tmp/pixi-cache-${USER}"

cd "$SLURM_SUBMIT_DIR" || exit 1
OUT="runs/m3"
mkdir -p "$OUT"

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
s = jax.local_devices()[0].memory_stats() or {}
print("allocator limit: %.2f GiB" % (s.get("bytes_limit", 0) / 2**30))
' || { echo "DEVICE CHECK FAILED"; exit 1; }

echo; echo "##### phase 1: dtype knob reaches the device #####"
# Cheapest possible proof that the branch under test is the one running: the f32
# leg must come back float32 from a CUDA device, not from a silent CPU fallback.
pixi run --locked -e gpu python -u -c '
import numpy as np
from logunusual import field, pk
from logunusual.grid import Box
spec = pk.PowerSpectrum.from_tsv("tests/data/matterpower_camb_zeff=0.9.tsv")
for dt, want in (("f64", np.float64), ("f32", np.float32)):
    F = field.generate_fields(spec, 1.76, Box(64, 640.0), 1, psi_axes="xyz", dtype=dt)
    assert F.delta_g.dtype == want, (dt, F.delta_g.dtype)
    assert F.delta_g.devices().pop().platform == "gpu"
    print(dt, "->", F.delta_g.dtype, F.delta_g.devices())
' || { echo "DTYPE CHECK FAILED"; exit 1; }

echo; echo "##### phase 2: allocator ladder, f64/f32 x eager/jit #####"
for DT in f64 f32; do
  for JIT in "" "--jit"; do
    TAG="${DT}$([ -n "$JIT" ] && echo _jit || echo _eager)"
    echo; echo "--- ladder $TAG ---"
    pixi run --locked -e gpu python -u scripts/m3_device.py ladder \
        --bins bin01 bin02 bin03 bin04 bin05 --dtype "$DT" $JIT \
        --out "$OUT/vista_alloc_ladder_${TAG}.json"
    echo "rc=$?"
  done
done

echo; echo "##### phase 3: P -> P_G split on Grace (the host-bound control) #####"
for B in bin02 bin05; do
  echo; echo "--- pkg $B ---"
  pixi run --locked -e gpu python -u scripts/m3_device.py pkg \
      --bin "$B" --repeat 5 --out "$OUT/vista_alloc_pkg_${B}.json"
  echo "rc=$?"
done

echo; echo "##### phase 4: 512^3 wall, eager vs jit #####"
for JIT in "" "--jit"; do
  TAG=$([ -n "$JIT" ] && echo jit || echo eager)
  echo; echo "--- wall bin05 $TAG ---"
  pixi run --locked -e gpu python -u scripts/m3_device.py wall \
      --bin bin05 $JIT --out "$OUT/vista_alloc_wall_bin05_${TAG}.json"
  echo "rc=$?"
done

echo
echo "=== done $(date -Is) ==="
ls -la "$OUT"/vista_alloc_* 2>/dev/null
