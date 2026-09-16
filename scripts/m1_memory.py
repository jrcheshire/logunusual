"""G9: peak live JAX-array bytes of the field stage, with the buffer count as referent.

    pixi run python scripts/m1_memory.py [--n 256 512] [--L 1000]

One subprocess per N (a JAX process keeps no resettable peak counter); each child runs a
warm-up at a small N first (compilation allocates), then the measured run. A polling
thread sums `jax.live_arrays()` bytes every millisecond (so transients inside a step,
e.g. the xi / xi_G / P_G triple in `grid_pkG`, are caught while they are Python-visible
arrays) and the step trace records the resting set after each step. What this sees:
device buffers held by JAX arrays. What it does NOT see: XLA intra-op scratch inside an
FFT or exp (umbrella memory `reference_measure_allocation_not_rss`), and the numpy white
noise before it is moved to the device. Host `ru_maxrss` is printed as a cross-check
only (macOS reads low).
"""

import argparse
import json
import subprocess
import sys

CHILD = r"""
import os, sys, resource, json
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax
import numpy as np
from logunusual import field
from logunusual.grid import Box
from logunusual.pk import PowerSpectrum
n, L, pk = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
spectrum = PowerSpectrum.from_tsv(pk)

def live_bytes():
    return sum(int(a.nbytes) for a in jax.live_arrays())

field.generate_fields(spectrum, 1.76, Box(32, L), 0)  # warm-up (compile), discarded
import threading, time
trace = []
state = {"label": "start", "peak": 0, "peak_label": "start", "stop": False}
def cb(label):
    trace.append((label, live_bytes()))
    state["label"] = label
def poll():
    while not state["stop"]:
        b = live_bytes()
        if b > state["peak"]:
            state["peak"] = b
            state["peak_label"] = "during step after '" + state["label"] + "'"
        time.sleep(0.001)
th = threading.Thread(target=poll, daemon=True); th.start()
F = field.generate_fields(spectrum, 1.76, Box(n, L), 1, keep_matter=True, trace=cb)
state["stop"] = True; th.join()
peak, peak_label = state["peak"], state["peak_label"]
n3_f64 = n**3 * 8
report = {
    "n": n, "cell_bytes_f64": n3_f64,
    "trace": [(l, b, round(b / n3_f64, 2)) for l, b in trace],
    "peak_bytes": peak, "peak_in_n3_f64_units": round(peak / n3_f64, 2),
    "peak_after_step": peak_label,
    "ru_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
    "device": str(jax.devices()[0]),
}
print(json.dumps(report))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[256])
    ap.add_argument("--L", type=float, default=1000.0)
    ap.add_argument("--pk", default="tests/data/matterpower_camb_zeff=0.9.tsv")
    args = ap.parse_args()
    for n in args.n:
        out = subprocess.run(
            [sys.executable, "-c", CHILD, str(n), str(args.L), args.pk],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        rep = json.loads(out.strip().splitlines()[-1])
        print(
            f"\nN = {n}: peak live JAX bytes {rep['peak_bytes'] / 2**30:.2f} GiB "
            f"= {rep['peak_in_n3_f64_units']} x (N^3 float64) {rep['peak_after_step']}"
            f"; host ru_maxrss {rep['ru_maxrss_bytes'] / 2**30:.2f} GiB; "
            f"{rep['device']}"
        )
        for label, b, units in rep["trace"]:
            print(f"   {label:12s} {b / 2**30:6.2f} GiB  ({units} N^3 f64)")
    print(
        "\nNot seen by this instrument: XLA intra-op scratch and host-side numpy "
        "arrays."
    )


if __name__ == "__main__":
    main()
