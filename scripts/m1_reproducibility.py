"""G8: same seed -> same catalog, across two fresh processes.

    pixi run python scripts/m1_reproducibility.py [--n 64] [--L 500]

Runs the field stage and one catalog in two subprocesses and compares the bytes. On
Linux (XLA CPU deterministic in the umbrella record) the result must be identical; on
macOS-arm64 the field stage can differ in the last bit, so the script REPORTS the
number of differing cells and the max relative difference rather than asserting.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

CHILD = r"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
from logunusual import field, sample
from logunusual.grid import Box
from logunusual.pk import PowerSpectrum
n, L, pk, out = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
box = Box(n, L)
spec = PowerSpectrum.from_tsv(pk)
F = field.generate_fields(spec, 1.76, box, 12345, keep_matter=True)
cat = sample.sample_catalog(F, 3e-3, 777, f=0.855, rsd=True)
np.savez(out, delta_g=np.asarray(F.delta_g), psi_z=np.asarray(F.psi_z),
         xyz=cat.xyz, cell=cat.cell)
"""


def run(n, L, pk, out):
    subprocess.run([sys.executable, "-c", CHILD, str(n), str(L), pk, out], check=True)
    return np.load(out)


def compare(a, b):
    rep = {}
    for key in a.files:
        x, y = a[key], b[key]
        same_shape = x.shape == y.shape
        n_diff = int(np.count_nonzero(x != y)) if same_shape else -1
        max_rel = 0.0
        if same_shape and n_diff and np.issubdtype(x.dtype, np.floating):
            max_rel = float(np.max(np.abs(x - y) / np.maximum(np.abs(x), 1e-300)))
        rep[key] = {
            "shape_equal": same_shape,
            "n_diff": n_diff,
            "max_rel": max_rel,
            "sha256_a": hashlib.sha256(x.tobytes()).hexdigest()[:16],
            "sha256_b": hashlib.sha256(y.tobytes()).hexdigest()[:16],
        }
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--L", type=float, default=500.0)
    ap.add_argument("--pk", default="tests/data/matterpower_camb_zeff=0.9.tsv")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as d:
        a = run(args.n, args.L, args.pk, str(Path(d) / "a.npz"))
        b = run(args.n, args.L, args.pk, str(Path(d) / "b.npz"))
        rep = compare(a, b)
    rep["platform"] = {
        "system": platform.system(),
        "machine": platform.machine(),
        "node": platform.node(),
        "cpu_count": os.cpu_count(),
    }
    print(json.dumps(rep, indent=1))
    identical = all(v["n_diff"] == 0 for k, v in rep.items() if k != "platform")
    print("IDENTICAL" if identical else "DIFFERS (see per-array report)")
    if platform.system() == "Linux" and not identical:
        sys.exit(1)


if __name__ == "__main__":
    main()
