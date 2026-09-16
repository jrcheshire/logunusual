#!/bin/bash
#SBATCH --job-name=logunusual-m3b
#SBATCH --partition=compute
#SBATCH --nodelist=deneb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.log
#
# M3 phase 2, correcting three things job 1956 got wrong or could not answer.
#
# 1956 measured the device allocator peak at 12.07 x N^3 float64, against the 7.5x
# that the live-array instrument (M1's G9) reports -- the card must provide ~60%
# more than G9 suggests. That moves the edge of the card DOWN a bin: bin03 at 384^3
# needs 5.09 GiB on that coefficient, which is over the 4.25 GiB default pool but
# under the 5.38 GiB that a 0.95 fraction gives. 1956's phase 2 only retried bin04
# and bin05, because its bin choice came from the old 7.0x number, so the one bin
# that might be rescued was never tested at 0.95. That is the headline question here.
#
# 2. The f32 leg of 1956 measured nothing: the probe cast to float32 only AFTER
#    grid_pkG, so its two full-grid FFTs ran in float64 and the probe OOMed in its
#    own f64 section. Fixed, along with a `1j *` promotion that widened float32
#    arrays to complex128, and the probe's self-check now runs as its own process
#    (peak_bytes_in_use has no reset, so two legs in one process give only a
#    one-sided bound).
#
# 3. Both wall legs of 1956 ran at the default pool and OOMed, so there is still no
#    timing for any bin. They run at 0.95 here, on bins that fit.

set -u
export PATH="$HOME/.pixi/bin:$PATH"
export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95   # 5.38 GiB of the 6 GiB card

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p runs/m3

echo "=== host $(hostname)  job ${SLURM_JOB_ID}  $(date -Is) ==="
echo "=== commit $(git rev-parse --short HEAD) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
echo

run() { echo; echo "##### $* #####"; pixi run -e gpu python -u "$@"; echo "exit=$?"; }

# No nested double quotes inside the f-string: this block is already inside a
# single-quoted shell argument, and escaping them there leaves the backslashes in
# the Python source (job 1957 died on exactly that, and blamed the GPU for it).
echo "##### phase 0: device check #####"
pixi run -e gpu python -u -c '
import jax
devs = jax.devices()
print("devices:", devs)
assert any(d.platform == "gpu" for d in devs), "no CUDA device"
stats = jax.local_devices()[0].memory_stats() or {}
lim = stats.get("bytes_limit")
print("allocator limit:", "%.2f GiB" % (lim / 2**30) if lim else "unreported")
' || { echo "PHASE 0 FAILED -- see the traceback above; it is NOT necessarily the GPU"; exit 1; }

# ---- phase 1: the whole f64 ladder at 0.95; bin03 is the question ------------
run scripts/m3_device.py ladder --bins bin01 bin02 bin03 bin04 bin05 \
    --out runs/m3/b_ladder_f64_p95.json

# ---- phase 2: f32, now that the probe is actually float32 -------------------
# --validate-n 256 puts the self-check on a grid big enough to exercise the FFT
# path properly while still fitting the card in f64 (1.51 GiB measured at 256^3).
run scripts/m3_device.py ladder --bins bin03 bin04 bin05 --dtype f32 \
    --validate-n 256 --out runs/m3/b_ladder_f32_p95.json

# ---- phase 3: where the time goes, on bins that fit -------------------------
run scripts/m3_device.py wall --bin bin01 --out runs/m3/b_wall_bin01.json
run scripts/m3_device.py wall --bin bin02 --out runs/m3/b_wall_bin02.json
run scripts/m3_device.py wall --bin bin03 --out runs/m3/b_wall_bin03.json

echo
echo "=== done $(date -Is) ==="
ls -la runs/m3/
