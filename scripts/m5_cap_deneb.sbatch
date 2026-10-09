#!/bin/bash
#SBATCH --job-name=logunusual-m5cap
#SBATCH --partition=compute
#SBATCH --nodelist=deneb
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --mem=52G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log
#
# M5 velocity-source cap measurement on deneb (ROADMAP M5, `scripts/m5_cap.py`). One
# process at a time: the 512^3 bins need ~38 GB (extrapolated from 192^3 / 256^3 on
# the laptop), so two concurrent streams would not fit deneb's 56 GB.
#
# Submit from the repo root on deneb:
#     sbatch scripts/m5_cap_deneb.sbatch
# Optional overrides: M5_CAP_NU="5 4.5 4" M5_CAP_RADIUS="1 2" M5_CAP_OUT=runs/m5/cap.jsonl
#
# Phases, most decision-relevant first; each skips rows already in the output (one
# JSON line per (realization, bin)), so resubmitting continues where a job stopped.
#   1. the nine realizations that failed the guard in job 2082, bins 1-3
#   2. realizations 0-19, bins 1-3
#   3. realizations 0-19, bins 4-7

set -u
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=1

NU=${M5_CAP_NU:-5 4.5 4}
RADIUS=${M5_CAP_RADIUS:-1 2}
OUT=${M5_CAP_OUT:-runs/m5/cap.jsonl}
FAILED="2 11 21 24 29 48 49 55 82"
SEEDS="$(seq -s ' ' 0 19)"

cd "${SLURM_SUBMIT_DIR:?}" || exit 1
mkdir -p "$(dirname "$OUT")"

echo "=== host $(hostname)  job ${SLURM_JOB_ID}  $(date '+%Y-%m-%dT%H:%M:%S') ==="
echo "=== commit $(git rev-parse --short HEAD)  $(git status --porcelain | wc -l) dirty paths ==="
echo "=== nu: $NU  radius: $RADIUS  -> $OUT ==="

echo; echo "##### phase 0: environment #####"
pixi install --locked || { echo "pixi install FAILED"; exit 1; }
pixi run --locked setup || { echo "setup FAILED"; exit 1; }
pixi run --locked python -u -c '
import jax, logunusual
print("jax", jax.__version__, jax.devices(), "x64", jax.config.jax_enable_x64)
print("logunusual", logunusual.__version__)
' || { echo "IMPORT CHECK FAILED"; exit 1; }

phase() {
    local name=$1 bins=$2 seeds=$3
    echo; echo "##### $name: bins $bins, realizations $seeds  $(date '+%H:%M:%S') #####"
    # word splitting of the lists is intended
    # shellcheck disable=SC2086
    pixi run --locked python -u scripts/m5_cap.py --bins $bins --realizations $seeds \
        --nu $NU --radius $RADIUS --out "$OUT" || { echo "$name FAILED"; exit 1; }
}

phase "phase 1 (job 2082 failures)" "1 2 3" "$FAILED"
phase "phase 2" "1 2 3" "$SEEDS"
phase "phase 3" "4 5 6 7" "$SEEDS"

echo; echo "=== done $(date '+%Y-%m-%dT%H:%M:%S'): $(wc -l < "$OUT") rows in $OUT ==="
