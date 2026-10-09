#!/bin/bash
#SBATCH --job-name=logunusual-m5
#SBATCH --partition=compute
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log
#
# M5 ensemble production on deneb (ROADMAP M5): `configs/v28_halofit.yaml`, two
# concurrent streams of realizations, each on 16 physical cores.
#
# Submit from the repo root on deneb, one realization range per stream:
#     sbatch scripts/m5_deneb.sbatch 0:1 50:51      # pilot
#     sbatch scripts/m5_deneb.sbatch 0:50 50:100    # full ensemble
#
# 64 CPUs can only be satisfied by deneb, so the request places the job without a
# node pin. No GPU: the 3050 holds 3 of the 7 bins. Memory: two processes at the
# measured 14.0 GB peak RSS (laptop, same config) plus margin.
#
# Inputs, relative to the checkout as the config names them: the 14 kh-10 tables in
# data/ and the fiducial mask as data/mask.h5. Each realization is its own process:
# run -> check -> sha256 into the manifest. An existing catalog is skipped (atomic
# rename: a file with the final name is complete), a failed realization is logged and
# the stream moves on, and three failures in a row stop the stream, so resubmitting
# the same command fills gaps. Bitwise reproduction of a catalog needs the same
# 16-core affinity (XLA-CPU's reduction order follows its thread pool).

set -u
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=1

RANGE_A=${1:?usage: sbatch scripts/m5_deneb.sbatch A0:A1 B0:B1}
RANGE_B=${2:?usage: sbatch scripts/m5_deneb.sbatch A0:A1 B0:B1}
CONFIG=${M5_CONFIG:-configs/v28_halofit.yaml}
OUT=${M5_OUT:-/work/jamie/logunusual/m5}
RUN_DIR="$OUT/$(awk '$1 == "run_name:" {print $2}' "$CONFIG")"
MANIFEST="$OUT/sha256_manifest.txt"

cd "${SLURM_SUBMIT_DIR:?}" || exit 1
mkdir -p "$OUT" runs/m5
touch "$MANIFEST"

echo "=== host $(hostname)  job ${SLURM_JOB_ID}  $(date '+%Y-%m-%dT%H:%M:%S') ==="
echo "=== commit $(git rev-parse --short HEAD)  $(git status --porcelain | wc -l) dirty paths ==="
echo "=== streams $RANGE_A (cores 0-15) and $RANGE_B (cores 16-31) -> $RUN_DIR ==="
df -h "$OUT" | tail -1

# ---- phase 0: CPU environment, built on the node that runs it ---------------
echo; echo "##### phase 0: environment #####"
pixi install --locked || { echo "pixi install FAILED"; exit 1; }
pixi run --locked setup || { echo "setup FAILED"; exit 1; }
pixi run --locked python -u -c '
import jax, logunusual
print("jax", jax.__version__, jax.devices(), "x64", jax.config.jax_enable_x64)
print("logunusual", logunusual.__version__)
' || { echo "IMPORT CHECK FAILED"; exit 1; }
pixi run --locked logunusual run --config "$CONFIG" --output-dir "$OUT" \
    --realizations 0:1 --dry-run || { echo "DRY RUN FAILED"; exit 1; }

# ---- phase 1: two streams ----------------------------------------------------
# The generator runs as a child of a small Python parent so that the child's peak RSS
# is read exactly (getrusage of the waited-for child), not sampled.
RUN_WITH_RSS='
import resource, subprocess, sys
rc = subprocess.call(sys.argv[1:])
unit = 1 if sys.platform == "darwin" else 1024  # ru_maxrss: bytes on macOS, kB on Linux
rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * unit
print("peak_rss_gb %.2f" % (rss / 1e9))
sys.exit(rc)
'

stream() {  # stream <label> <cores> <r0:r1>
    local label=$1 cores=$2 lo=${3%%:*} hi=${3##*:} fails=0 failed=0 r cat log
    local t0 t1 t2 t3 rc
    for ((r = lo; r < hi; r++)); do
        cat=$(printf '%s/realization_%05d/catalog.parq' "$RUN_DIR" "$r")
        log=$(printf 'runs/m5/r%05d.log' "$r")
        if [ -e "$cat" ]; then
            echo "[$label] r$r exists, skipping"
            continue
        fi
        t0=$(date +%s)
        taskset -c "$cores" pixi run --locked python -u -c "$RUN_WITH_RSS" \
            logunusual run --config "$CONFIG" --output-dir "$OUT" \
            --realizations "$r:$((r + 1))" > "$log" 2>&1
        rc=$?
        t1=$(date +%s)
        if [ $rc -eq 0 ] && pixi run --locked logunusual check "$cat" >> "$log" 2>&1; then
            t2=$(date +%s)
            (cd "$OUT" && sha256sum "${cat#"$OUT"/}") >> "$MANIFEST"
            t3=$(date +%s)
            fails=0
            echo "[$label] r$r ok: run $((t1 - t0)) s, check $((t2 - t1)) s," \
                "sha256 $((t3 - t2)) s, $(grep peak_rss_gb "$log")," \
                "$(du -m "$cat" | cut -f1) MB"
        else
            fails=$((fails + 1))
            failed=$((failed + 1))
            echo "[$label] r$r FAILED (run rc=$rc) after $(($(date +%s) - t0)) s; see $log"
            if [ $fails -ge 3 ]; then
                echo "[$label] three failures in a row, stopping"
                return 1
            fi
        fi
    done
    [ $failed -eq 0 ]  # nonzero if any realization in the range failed
}

echo; echo "##### phase 1: streams #####"
stream A 0-15 "$RANGE_A" &
PID_A=$!
stream B 16-31 "$RANGE_B" &
PID_B=$!
# Array write rate every 30 s while either stream runs (/proc/diskstats field 10:
# 512-byte sectors written to md0, the /work array this node serves).
SECT0=$(awk '$3 == "md0" {print $10}' /proc/diskstats 2>/dev/null)
SECT0=${SECT0:-0}
while kill -0 $PID_A 2>/dev/null || kill -0 $PID_B 2>/dev/null; do
    sleep 30
    SECT1=$(awk '$3 == "md0" {print $10}' /proc/diskstats 2>/dev/null)
    SECT1=${SECT1:-0}
    echo "--- $(date +%T) md0 write $(((SECT1 - SECT0) * 512 / 30 / 1000000)) MB/s"
    SECT0=$SECT1
done
wait $PID_A; RC_A=$?
wait $PID_B; RC_B=$?

echo
echo "=== streams done $(date '+%Y-%m-%dT%H:%M:%S'): A rc=$RC_A  B rc=$RC_B ==="
echo "=== catalogs $(find "$RUN_DIR" -name catalog.parq | wc -l)  manifest $(wc -l < "$MANIFEST") ==="
du -sh "$RUN_DIR"
[ $RC_A -eq 0 ] && [ $RC_B -eq 0 ]
