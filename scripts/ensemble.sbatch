#!/bin/bash
#SBATCH --job-name=logunusual-ensemble
##SBATCH --partition=<partition>
##SBATCH --account=<account>
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log
#
# Ensemble production on one CPU node: two concurrent streams of realizations, each
# pinned to 16 cores. Set the partition/account lines for your cluster.
#
# Submit from the repo root, one realization range per stream:
#     ENSEMBLE_OUT=/path/to/out sbatch scripts/ensemble.sbatch 0:1 50:51     # pilot
#     ENSEMBLE_OUT=/path/to/out sbatch scripts/ensemble.sbatch 0:50 50:100   # 100
# ENSEMBLE_CONFIG overrides the config (default configs/v28_halofit.yaml), whose input
# paths are relative to the checkout. Memory: two processes at ~14 GB peak RSS each
# for the default bins, plus margin.
#
# Each realization is its own process: run -> `logunusual check` -> sha256 into
# $ENSEMBLE_OUT/sha256_manifest.txt. An existing catalog is skipped (a file with the
# final name is complete), a failed realization is logged to runs/ensemble/ and the
# stream moves on, and three failures in a row stop the stream, so resubmitting the
# same command fills gaps. Bitwise reproduction of a catalog needs the same 16-core
# affinity (XLA-CPU's reduction order follows its thread pool).

set -u
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=1

RANGE_A=${1:?usage: sbatch scripts/ensemble.sbatch A0:A1 B0:B1}
RANGE_B=${2:?usage: sbatch scripts/ensemble.sbatch A0:A1 B0:B1}
CONFIG=${ENSEMBLE_CONFIG:-configs/v28_halofit.yaml}
OUT=${ENSEMBLE_OUT:?set ENSEMBLE_OUT to the output directory}
RUN_DIR="$OUT/$(awk '$1 == "run_name:" {print $2}' "$CONFIG")"
MANIFEST="$OUT/sha256_manifest.txt"

cd "${SLURM_SUBMIT_DIR:?}" || exit 1
mkdir -p "$OUT" runs/ensemble
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
        log=$(printf 'runs/ensemble/r%05d.log' "$r")
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
wait $PID_A; RC_A=$?
wait $PID_B; RC_B=$?

echo
echo "=== streams done $(date '+%Y-%m-%dT%H:%M:%S'): A rc=$RC_A  B rc=$RC_B ==="
echo "=== catalogs $(find "$RUN_DIR" -name catalog.parq | wc -l)  manifest $(wc -l < "$MANIFEST") ==="
du -sh "$RUN_DIR"
[ $RC_A -eq 0 ] && [ $RC_B -eq 0 ]
